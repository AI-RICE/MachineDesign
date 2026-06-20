"""H0 vs H1 separability study for the 5-phase SynRM.

Question: is sequential (design-then-control) optimization as good as joint
optimization of rotor geometry and dq1/dq3 current setpoints?

  H0 (sequential): stage1 maximizes torque over geometry + dq1 angle with dq3=0;
                   stage2 freezes the geometry and optimizes the currents
                   (dq1 AND dq3) under the peak-current constraint.
  H1 (joint):      a LOCAL trust-region joint BO over geometry + currents around
                   the H0 endpoint P0. Because stage2 already made the currents
                   optimal at P0 (dT/dc = 0), any torque gained here comes only
                   from re-shaping the geometry for the dq3-inclusive operating
                   point -> it isolates the geometry x composition interaction.
                   Gain beyond the FEA noise floor => H1 (coupled); else H0.

Objective: maximize ANSYS mean torque s.t. peak phase current <= I_MAX. The
peak is enforced analytically by the setpoint parameterization (every candidate
sits exactly on the peak budget), computed from the imposed-current waveform.

Setpoint parameterization (3 free dims, on the peak-current boundary):
  phi1  = dq1 angle (epsI1)
  rho   = Im3/Im1   (3rd-harmonic injection ratio)
  delta = epsI3 - 3*phi1  (relative dq3 phase; peak depends only on rho,delta)
  Im1   = I_MAX / max_theta|cos(theta) + rho*cos(3 theta + delta)|  (peak = I_MAX)

Usage (bayes):
  export ANSYSEM_ROOT242=/data/Ansys/v242/Linux64/
  ./venv_5f/bin/python notebooks/h0h1_study.py --stage stage1 [--mock]
  ./venv_5f/bin/python notebooks/h0h1_study.py --stage stage2 [--mock]
  ./venv_5f/bin/python notebooks/h0h1_study.py --stage h1     [--mock]
Each stage checkpoints every eval to results_h0h1/<stage>.npz and resumes.
"""

import argparse
import json
import os

import numpy as np
import torch
from botorch.acquisition import LogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

from machine_design import HacklGenerator_SixLambdas, analyze_results
from machine_design.design2 import Design2

torch.set_default_dtype(torch.float64)

# --------------------------------------------------------------------------- #
# Setpoint parameterization + analytic peak-current constraint
# --------------------------------------------------------------------------- #
I_MAX = 10.0  # A, peak phase current budget
_THETA = np.linspace(0.0, 2.0 * np.pi, 4000, endpoint=False)

# free current params (phi1, rho, delta) and their physical ranges
PHI1_MAX = np.pi / 2.0
RHO_MAX = 0.5
DELTA_MAX = 2.0 * np.pi


def peak_per_unit(rho, delta):
    """max_theta |cos(theta) + rho*cos(3 theta + delta)| -- peak per unit Im1."""
    return float(np.max(np.abs(np.cos(_THETA) + rho * np.cos(3.0 * _THETA + delta))))


def setpoint_from_params(phi1, rho, delta, i_max=I_MAX):
    """(phi1,rho,delta) -> (Id1,Iq1,Id3,Iq3) with peak phase current == i_max."""
    g = peak_per_unit(rho, delta)
    Im1 = i_max / g
    Im3 = rho * Im1
    eps1 = phi1
    eps3 = delta + 3.0 * phi1
    Id1, Iq1 = Im1 * np.cos(eps1), Im1 * np.sin(eps1)
    Id3, Iq3 = Im3 * np.cos(eps3), Im3 * np.sin(eps3)
    return Id1, Iq1, Id3, Iq3, Im1, Im3


def actual_waveform_peak(Id1, Iq1, Id3, Iq3):
    """Peak of the realized phase-A waveform (sanity check of the analytic peak)."""
    Im1, eps1 = np.hypot(Id1, Iq1), np.arctan2(Iq1, Id1)
    Im3, eps3 = np.hypot(Id3, Iq3), np.arctan2(Iq3, Id3)
    wave = Im1 * np.cos(_THETA + eps1 - np.pi) + Im3 * np.cos(3.0 * _THETA + eps3 - np.pi)
    return float(np.max(np.abs(wave)))


PENALTY = 0.0  # torque assigned to an infeasible geometry

# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #

def geom_bounds_arrays(generator):
    lb, ub = generator.bounds
    return np.asarray(lb, float), np.asarray(ub, float)


def geom_norm_to_phys(gn, lb, ub):
    return lb + np.asarray(gn) * (ub - lb)


def geom_phys_to_norm(gp, lb, ub):
    return (np.asarray(gp) - lb) / (ub - lb)


def rand_feasible_geom_norm(generator, lb, ub, max_tries=200):
    """A uniformly-random barrier-feasible geometry, returned normalized to [0,1]^12."""
    for _ in range(max_tries):
        p = generator.random_parameters()  # (phis_in, phis_out, lam_in, lam_out)
        flat = np.concatenate([np.atleast_1d(x).ravel() for x in p])
        generator.set_parameters(generator.X_to_params(flat))
        barriers = generator.split_barriers(generator.generate_barriers())
        if generator.feasible_barriers(barriers):
            return geom_phys_to_norm(flat, lb, ub)
    raise RuntimeError("could not draw a feasible geometry")


def build_barriers(generator, geom_norm, lb, ub):
    """Return feasible barrier polylines for a normalized geometry, or None."""
    phys = geom_norm_to_phys(np.clip(geom_norm, 0.0, 1.0), lb, ub)
    generator.set_parameters(generator.X_to_params(phys))
    barriers = generator.split_barriers(generator.generate_barriers())
    return barriers if generator.feasible_barriers(barriers) else None


# --------------------------------------------------------------------------- #
# Objective: ANSYS torque (or a mock surrogate with a built-in interaction)
# --------------------------------------------------------------------------- #

def mock_torque(geom_norm, phi1, rho, delta):
    """Cheap synthetic objective for pipeline validation. Has a deliberate
    geometry x composition interaction: the dq3 term prefers a *different*
    geometry than the fundamental, so H1 should find a gain over H0."""
    gn = np.asarray(geom_norm)
    shape1 = np.exp(-np.sum((gn - 0.30) ** 2) / 0.5)  # fundamental likes ~0.30
    shape2 = np.exp(-np.sum((gn - 0.70) ** 2) / 0.5)  # 3rd harmonic likes ~0.70
    t_fund = 35.0 * np.sin(2.0 * phi1) * shape1
    t_3rd = 25.0 * rho * shape2 * (0.7 + 0.3 * np.cos(delta))
    return t_fund + t_3rd


def eval_torque(geom_norm, phi1, rho, delta, ctx):
    """Returns (torque, feasible). Infeasible geometry -> (PENALTY, False)."""
    if ctx["mock"]:
        if np.any(geom_norm < 0) or np.any(geom_norm > 1):
            return PENALTY, False
        return mock_torque(geom_norm, phi1, rho, delta), True

    gen, lb, ub = ctx["gen"], ctx["lb"], ctx["ub"]
    barriers = build_barriers(gen, geom_norm, lb, ub)
    if barriers is None:
        return PENALTY, False
    Id1, Iq1, Id3, Iq3, _, _ = setpoint_from_params(phi1, rho, delta)
    design = ctx["design"]
    design.add_rotor()
    for b in barriers:
        design.add_rotor_barrier(b)
    try:
        Tor = design.compute(Id1, Iq1, Id3, Iq3, NUM_CORES=ctx["num_cores"])
        tmean, _, _ = analyze_results(np.asarray(Tor, float))
        ok = np.isfinite(tmean)
        tmean = float(tmean) if ok else PENALTY
    except Exception as e:  # noqa: BLE001
        print(f"    [eval] FEA exception: {e}")
        tmean, ok = PENALTY, False
    design.delete_rotor()
    return tmean, ok


# --------------------------------------------------------------------------- #
# Per-stage decode: normalized search vector u in [0,1]^dim -> setpoint
# --------------------------------------------------------------------------- #

def make_decoder(stage, fixed_geom=None, center=None, tr_geom=0.15, tr_cur=0.20):
    if stage == "stage1":
        dim = 13  # 12 geometry + phi1 ; rho=0
        def decode(u):
            return np.asarray(u[:12]), u[12] * PHI1_MAX, 0.0, 0.0
    elif stage == "stage2":
        dim = 3  # phi1, rho, delta ; geometry fixed
        def decode(u):
            return np.asarray(fixed_geom), u[0] * PHI1_MAX, u[1] * RHO_MAX, u[2] * DELTA_MAX
    elif stage == "h1":
        dim = 15  # 12 geometry + 3 currents, in a trust region around `center`
        gc = np.asarray(center["geom"])           # normalized geometry of P0
        cc = np.asarray(center["cur_norm"])        # normalized (phi1,rho,delta) of P0
        def decode(u):
            geom = np.clip(gc + (2.0 * np.asarray(u[:12]) - 1.0) * tr_geom, 0.0, 1.0)
            cur = np.clip(cc + (2.0 * np.asarray(u[12:15]) - 1.0) * tr_cur, 0.0, 1.0)
            return geom, cur[0] * PHI1_MAX, cur[1] * RHO_MAX, cur[2] * DELTA_MAX
    else:
        raise ValueError(stage)
    return dim, decode


# --------------------------------------------------------------------------- #
# Single-objective BO loop (LogEI, q=1), resumable
# --------------------------------------------------------------------------- #

def run_bo(stage, dim, decode, ctx, n_init, n_total, ckpt, seed, init_u=None):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if os.path.exists(ckpt):
        d = np.load(ckpt)
        U = list(d["U"])
        Y = list(d["Y"])
        print(f"[{stage}] resumed at {len(Y)} evals from {ckpt}")
    else:
        U, Y = [], []

    # initial design
    if len(Y) < n_init:
        seeds = []
        if init_u is not None:
            seeds.extend(list(init_u))
        # geometry stages: draw feasible-geometry seeds; others: uniform
        while len(seeds) < n_init:
            if stage == "stage1":
                gn = rand_feasible_geom_norm(ctx["gen"], ctx["lb"], ctx["ub"]) if not ctx["mock"] \
                    else np.random.rand(12)
                seeds.append(np.concatenate([gn, np.random.rand(1)]))
            else:
                seeds.append(np.random.rand(dim))
        for u in seeds[len(Y):n_init]:
            gn, phi1, rho, delta = decode(u)
            y, ok = eval_torque(gn, phi1, rho, delta, ctx)
            U.append(np.asarray(u, float)); Y.append(y)
            print(f"[{stage}] init {len(Y)}/{n_init}  T={y:.4f}  feas={ok}")
            np.savez(ckpt, U=np.array(U), Y=np.array(Y))

    bounds = torch.stack([torch.zeros(dim), torch.ones(dim)])
    while len(Y) < n_total:
        train_U = torch.tensor(np.array(U))
        train_Y = torch.tensor(np.array(Y)).unsqueeze(-1)
        gp = SingleTaskGP(train_U, train_Y, outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        acqf = LogExpectedImprovement(gp, best_f=train_Y.max())
        cand, _ = optimize_acqf(acqf, bounds=bounds, q=1, num_restarts=10, raw_samples=256)
        u = cand.detach().numpy().ravel()
        gn, phi1, rho, delta = decode(u)
        y, ok = eval_torque(gn, phi1, rho, delta, ctx)
        U.append(u); Y.append(y)
        np.savez(ckpt, U=np.array(U), Y=np.array(Y))
        print(f"[{stage}] {len(Y)}/{n_total}  T={y:.4f}  best={max(Y):.4f}  feas={ok}")

    U, Y = np.array(U), np.array(Y)
    best = int(np.argmax(Y))
    return U, Y, best


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
OUT = "results_h0h1"


def open_design(ctx, project):
    if ctx["mock"]:
        return None
    path = os.path.join(os.getcwd(), "data", f"{project}.aedt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    kw = dict(version=ctx["aedt_version"], non_graphical=True, new_desktop=False, close_on_exit=True)
    if os.path.exists(path):
        return Design2.load(path, **kw)
    return Design2.create(project, "Design01", path, **kw)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=["stage1", "stage2", "h1", "probe"])
    p.add_argument("--mock", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-cores", type=int, default=4)
    p.add_argument("--aedt-version", default="2024.2")
    p.add_argument("--n-init-s1", type=int, default=24)
    p.add_argument("--n-evals-s1", type=int, default=120)
    p.add_argument("--n-init-s2", type=int, default=8)
    p.add_argument("--n-evals-s2", type=int, default=32)
    p.add_argument("--n-init-h1", type=int, default=20)
    p.add_argument("--n-evals-h1", type=int, default=50)
    p.add_argument("--tr-geom", type=float, default=0.15, help="h1 geometry trust-region half-width (normalized)")
    p.add_argument("--tr-cur", type=float, default=0.20, help="h1 current trust-region half-width (normalized)")
    p.add_argument("--tag", default="", help="suffix for h1 checkpoint/summary, e.g. 'wide'")
    args = p.parse_args()

    os.makedirs(OUT, exist_ok=True)
    gen_design = None if args.mock else open_design({"mock": False, "aedt_version": args.aedt_version}, "SynRM5f_study")
    # the generator needs a design for rotor_r_min/max; build a throwaway in mock
    if args.mock:
        class _D:  # minimal stand-in carrying the derived radii
            rotor_r_min, rotor_r_max = 12.5, 39.275
        gen = HacklGenerator_SixLambdas(_D(), 0.7, offset=0.35)
    else:
        gen = HacklGenerator_SixLambdas(gen_design, 0.7, offset=0.35)
    lb, ub = geom_bounds_arrays(gen)

    ctx = dict(mock=args.mock, gen=gen, lb=lb, ub=ub, design=gen_design,
               num_cores=args.num_cores, aedt_version=args.aedt_version)

    if args.stage == "probe":
        # Re-evaluate the H0 endpoint P0 = (G*, c*) and report mean torque AND
        # ripple (which the optimization so far ignored). Grounds the R_max choice.
        s2 = json.load(open(f"{OUT}/stage2_best.json"))
        gn = np.array(s2["geom_norm"])
        barriers = build_barriers(gen, gn, lb, ub)
        assert barriers is not None, "G* came out infeasible?!"
        Id1, Iq1, Id3, Iq3, _, _ = setpoint_from_params(s2["phi1"], s2["rho"], s2["delta"])
        gen_design.add_rotor()
        for b in barriers:
            gen_design.add_rotor_barrier(b)
        Tor = gen_design.compute(Id1, Iq1, Id3, Iq3, NUM_CORES=args.num_cores)
        Tmean, _, Tripple = analyze_results(np.asarray(Tor, float))
        gen_design.delete_rotor()
        print(f"[probe] P0  T_mean={Tmean:.4f} Nm  ripple={Tripple:.3f} %  "
              f"(rho={s2['rho']:.3f}, T_seq stored={s2['T_seq']:.4f})")
        gen_design.save_project()
        gen_design.close_project()
        return

    if args.stage == "stage1":
        dim, decode = make_decoder("stage1")
        U, Y, best = run_bo("stage1", dim, decode, ctx, args.n_init_s1, args.n_evals_s1,
                            f"{OUT}/stage1.npz", args.seed)
        gn, phi1, _, _ = decode(U[best])
        json.dump({"T": float(Y[best]), "geom_norm": gn.tolist(), "phi1": float(phi1)},
                  open(f"{OUT}/stage1_best.json", "w"), indent=2)
        print(f"[stage1] DONE  G* found, T(dq3=0)={Y[best]:.4f} Nm")

    elif args.stage == "stage2":
        s1 = json.load(open(f"{OUT}/stage1_best.json"))
        dim, decode = make_decoder("stage2", fixed_geom=np.array(s1["geom_norm"]))
        U, Y, best = run_bo("stage2", dim, decode, ctx, args.n_init_s2, args.n_evals_s2,
                            f"{OUT}/stage2.npz", args.seed)
        u = U[best]
        cur_norm = [u[0], u[1], u[2]]
        gn, phi1, rho, delta = decode(u)
        Id1, Iq1, Id3, Iq3, Im1, Im3 = setpoint_from_params(phi1, rho, delta)
        peak = actual_waveform_peak(Id1, Iq1, Id3, Iq3)
        json.dump({"T_seq": float(Y[best]), "geom_norm": s1["geom_norm"],
                   "cur_norm": cur_norm, "phi1": float(phi1), "rho": float(rho),
                   "delta": float(delta), "Im1": float(Im1), "Im3": float(Im3),
                   "peak_check_A": peak}, open(f"{OUT}/stage2_best.json", "w"), indent=2)
        print(f"[stage2] DONE  T_seq={Y[best]:.4f} Nm  rho={rho:.3f}  peak={peak:.3f} A (budget {I_MAX})")

    elif args.stage == "h1":
        s2 = json.load(open(f"{OUT}/stage2_best.json"))
        center = {"geom": np.array(s2["geom_norm"]), "cur_norm": np.array(s2["cur_norm"])}
        dim, decode = make_decoder("h1", center=center, tr_geom=args.tr_geom, tr_cur=args.tr_cur)
        tag = f"_{args.tag}" if args.tag else ""
        p0 = 0.5 * np.ones(dim)  # u=0.5 maps to the trust-region centre = P0
        U, Y, best = run_bo("h1", dim, decode, ctx, args.n_init_h1, args.n_evals_h1,
                            f"{OUT}/h1{tag}.npz", args.seed, init_u=[p0])
        t_seq = s2["T_seq"]
        t_h1 = float(Y[best])
        dT = t_h1 - t_seq
        gn, phi1, rho, delta = decode(U[best])
        geom_move = float(np.linalg.norm(gn - center["geom"]))
        json.dump({"T_seq": t_seq, "T_h1": t_h1, "dT": dT, "dT_pct": 100 * dT / t_seq,
                   "geom_move_norm": geom_move, "rho_h1": float(rho),
                   "tr_geom": args.tr_geom, "tr_cur": args.tr_cur},
                  open(f"{OUT}/h1{tag}_summary.json", "w"), indent=2)
        print(f"[h1] DONE  T_seq={t_seq:.4f}  T_h1={t_h1:.4f}  dT={dT:.4f} Nm "
              f"({100*dT/t_seq:.2f}%)  geom_move={geom_move:.3f}")

    if gen_design is not None:
        gen_design.save_project()
        gen_design.close_project()


if __name__ == "__main__":
    main()
