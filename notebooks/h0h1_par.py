"""Parallel, CONSTRAINED-EI H0/H1 driver (worker-pool, ports RD pattern).

Currents are COMPLETELY FREE: the 4 dq components (Id1,Iq1,Id3,Iq3) are BO
variables -- no pinning, no rho/Delta structure. Constraints:
  * peak phase current <= I_MAX : ANALYTIC (known from the imposed waveform) ->
    enforced as a hard feasibility filter on candidates (no FEA wasted).
  * peak phase voltage <= V_MAX : FEA-evaluated, DESIGN-DEPENDENT -> constraint GP.
  * torque ripple      <= R_MAX : FEA-evaluated, design-dependent -> constraint GP.

Acquisition = qLogEI x P(feasible): a ModelListGP models [torque, ripple, Vpeak];
qLogExpectedImprovement maximizes torque subject to the (ripple, voltage)
constraints via their GPs, with best_f = best *feasible* observed torque. This
lets dq3 do whatever helps -- flatten the current waveform (current-limited) OR
flatten the voltage waveform (voltage-limited) -- instead of a fixed penalty.

Stages: probe_speed | stage1 (geom + Id1,Iq1, dq3=0) | stage2 (4 dq, fixed geom)
        | joint (geom + 4 dq, trust region) | crosscheck.
"""

import argparse
import json
import math
import os
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import h0h1_study as H  # noqa: E402  (geometry helpers, Design2, analyze_results, _THETA)

from botorch.acquisition.logei import qLogExpectedImprovement  # noqa: E402
from botorch.acquisition.objective import GenericMCObjective  # noqa: E402
from botorch.fit import fit_gpytorch_mll  # noqa: E402
from botorch.models import ModelListGP, SingleTaskGP  # noqa: E402
from botorch.models.transforms import Standardize  # noqa: E402
from botorch.optim import optimize_acqf  # noqa: E402
from botorch.sampling.normal import SobolQMCNormalSampler  # noqa: E402
from gpytorch.mlls import SumMarginalLogLikelihood  # noqa: E402

torch.set_default_dtype(torch.float64)

OUT = "results_h0h1_par"
I_MAX = 10.0       # A, peak phase current limit
BIG_RIPPLE = 999.0
BIG_VOLT = 9999.0
R_STATOR = 19.0   # phase resistance consistent with Nc=113 (the earlier 0.19 decoupled R from the winding — a mistake)
# End-winding leakage inductance [H/phase] = 2.4 mH (measured value; supersedes the earlier
# ~1.4 mH Pyrhonen analytic estimate for D=79mm, Nc=113, q=2, p=2). 2D FEA cannot capture it
# (end turns are out of plane), so it is added ANALYTICALLY in post-processing as
# psi += Lew*I and V_d -= h*w*Lew*Iq / V_q += h*w*Lew*Id -- NOT via deriv()/ddt() in an FEA
# output var, which is unsupported by the AEDT 2024.2 parser and breaks get_solution_data.
LEW_H = 2.4e-3
POLE_PAIRS = 2

# dq current box: dq1 in the motoring quadrant, dq3 injection either sign.
ICUR_LB = np.array([0.0, 0.0, -0.3 * I_MAX, -0.3 * I_MAX])
ICUR_UB = np.array([I_MAX, I_MAX, 0.3 * I_MAX, 0.3 * I_MAX])


# --------------------------------------------------------------------------- #
# Analytic waveform peaks (current is known exactly; voltage comes from FEA)
# --------------------------------------------------------------------------- #
def peak_current_from_dq(Id1, Iq1, Id3, Iq3):
    Im1, a1 = math.hypot(Id1, Iq1), math.atan2(Iq1, Id1)
    Im3, a3 = math.hypot(Id3, Iq3), math.atan2(Iq3, Id3)
    th = H._THETA
    return float(np.max(np.abs(Im1 * np.cos(th + a1) + Im3 * np.cos(3.0 * th + a3))))


def combined_voltage_peak(vd1, vq1, vd3, vq3, w_elec=0.0, dq=None, Lew=0.0):
    # Optional end-winding leakage (analytic, dodges the ddt() parser issue): in the
    # dq frame of harmonic h the leakage adds V_d -= h*w*Lew*Iq, V_q += h*w*Lew*Id.
    if Lew and w_elec and dq is not None:
        vd1 = vd1 - w_elec * Lew * dq[1]; vq1 = vq1 + w_elec * Lew * dq[0]
        vd3 = vd3 - 3.0 * w_elec * Lew * dq[3]; vq3 = vq3 + 3.0 * w_elec * Lew * dq[2]
    Vm1, p1 = math.hypot(vd1, vq1), math.atan2(vq1, vd1)
    Vm3, p3 = math.hypot(vd3, vq3), math.atan2(vq3, vd3)
    th = H._THETA
    return float(np.max(np.abs(Vm1 * np.cos(th + p1) + Vm3 * np.cos(3.0 * th + p3))))


def voltage_speed_boundary(psi_d1, psi_q1, Id1, Iq1, R, v_max, Lew=0.0):
    # end-winding leakage adds Lew*I to the (speed-independent) flux linkage
    psi_d1 = psi_d1 + Lew * Id1
    psi_q1 = psi_q1 + Lew * Iq1
    a = psi_d1**2 + psi_q1**2
    b = 2.0 * R * (Iq1 * psi_d1 - Id1 * psi_q1)
    c = R**2 * (Id1**2 + Iq1**2) - v_max**2
    if a <= 0:
        return None
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    return (-b + math.sqrt(disc)) / (2.0 * a)


# --------------------------------------------------------------------------- #
# Geometry generator (widened bounds)
# --------------------------------------------------------------------------- #
class WideSixLambdas(H.HacklGenerator_SixLambdas):
    def __init__(self, design, r_stator_end, **kwargs):
        super().__init__(design, r_stator_end, **kwargs)
        self.phis_inner_min = np.maximum(self.phis_inner_min - 4.0, 1.0)
        self.phis_outer_min = np.maximum(self.phis_outer_min - 4.0, 1.0)
        self.lam_inner_max = np.minimum(self.lam_inner_max + 0.15, 0.75)
        self.lam_outer_max = np.minimum(self.lam_outer_max + 0.15, 0.75)


def make_generator(design, wide):
    return (WideSixLambdas if wide else H.HacklGenerator_SixLambdas)(design, 0.7, offset=0.35)


# --------------------------------------------------------------------------- #
# Per-stage decoder: normalized u -> (geom_norm, dq=[Id1,Iq1,Id3,Iq3])
# --------------------------------------------------------------------------- #
def make_decoder(stage, fixed_geom=None, center=None, tr_geom=0.35, tr_cur=0.35):
    span = ICUR_UB - ICUR_LB
    if stage == "stage1":
        dim = 14  # 12 geom + Id1,Iq1 ; dq3 = 0
        def decode(u):
            return np.asarray(u[:12]), np.array([u[12] * I_MAX, u[13] * I_MAX, 0.0, 0.0])
    elif stage == "stage2":
        dim = 4   # Id1,Iq1,Id3,Iq3 ; geometry fixed
        def decode(u):
            return np.asarray(fixed_geom), ICUR_LB + np.asarray(u[:4]) * span
    elif stage == "joint":
        dim = 16  # 12 geom + 4 dq, trust region around P0
        gc, cc = np.asarray(center["geom"]), np.asarray(center["cur_norm"])
        def decode(u):
            geom = np.clip(gc + (2.0 * np.asarray(u[:12]) - 1.0) * tr_geom, 0.0, 1.0)
            curn = np.clip(cc + (2.0 * np.asarray(u[12:16]) - 1.0) * tr_cur, 0.0, 1.0)
            return geom, ICUR_LB + curn * span
    elif stage == "full16":
        dim = 16  # 12 geom + 4 dq, FULL boxes (global; for min-loss study)
        def decode(u):
            return np.asarray(u[:12]), ICUR_LB + np.asarray(u[12:16]) * span
    else:
        raise ValueError(stage)
    return dim, decode


def cur_to_norm(dq):
    return (np.asarray(dq) - ICUR_LB) / (ICUR_UB - ICUR_LB)


def mock_torque(gn, dq):
    Id1, Iq1, Id3, Iq3 = dq
    Im1, Im3 = math.hypot(Id1, Iq1), math.hypot(Id3, Iq3)
    shape1 = np.exp(-np.sum((gn - 0.30) ** 2) / 0.5)
    shape2 = np.exp(-np.sum((gn - 0.70) ** 2) / 0.5)
    return 3.5 * Im1 * shape1 + 2.5 * Im3 * shape2


# --------------------------------------------------------------------------- #
# Feasibility (cEI handles ripple/voltage via GPs; current is analytic+hard)
# --------------------------------------------------------------------------- #
def feasible(T, R, V, Ipk, r_max, v_max):
    return (np.asarray(T) > 0) & (np.asarray(R) <= r_max) & (np.asarray(V) <= v_max) \
        & (np.asarray(Ipk) <= I_MAX + 1e-6)


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #
def open_isolated_design(tag, wid, version, slots=40, phases=5):
    from machine_design.design2 import Design2, Design2_60
    if int(phases) == 3:
        from machine_design.design import Design_60
        cls = Design_60                                  # 3-phase, common 60-slot stator
    else:
        cls = Design2_60 if int(slots) == 60 else Design2
    path = os.path.join(os.getcwd(), "data", f"{tag}_w{wid}.aedt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    kw = dict(version=version, non_graphical=True, new_desktop=True, close_on_exit=True)
    return cls.load(path, **kw) if os.path.exists(path) else \
        cls.create(f"{tag}_w{wid}", "Design01", path, **kw)


def set_speed(design, fhz):
    if fhz is None:
        return
    design.m2d["f"] = f"{fhz}Hz"
    design.m2d["RotSpeed"] = f"{60.0 * fhz / POLE_PAIRS}rpm"


def eval_one(decode, u, gen, lb, ub, design, num_cores, mock):
    """Returns (T, ripple, Vpeak, Ipeak, feasible_geom)."""
    gn, dq = decode(u)
    Ipk = peak_current_from_dq(*dq)
    if mock:
        if np.any(gn < 0) or np.any(gn > 1):
            return H.PENALTY, BIG_RIPPLE, BIG_VOLT, Ipk, False
        t = mock_torque(gn, dq)
        r = 2.0 + 0.15 * abs(dq[1]) + 5.0 * math.hypot(dq[2], dq[3])     # realistic ~3-5%
        v = 200.0 + 10.0 * t + 20.0 * dq[0] - 30.0 * dq[2]               # dq3 (Id3<0) lowers V
        return t, r, v, Ipk, True
    barriers = H.build_barriers(gen, gn, lb, ub)
    if barriers is None:
        return H.PENALTY, BIG_RIPPLE, BIG_VOLT, Ipk, False
    design.add_rotor()
    for b in barriers:
        design.add_rotor_barrier(b)
    try:
        res = design.compute(*[float(x) for x in dq], NUM_CORES=num_cores)
        tor, m = res["Tor"], res["means"]
        tmean, _, ripple = H.analyze_results(np.asarray(tor, float))
        # 3-phase machines (Design_60) return no dq-voltage means -> no-limit eval
        vpk = combined_voltage_peak(m["V_d1"], m["V_q1"], m["V_d3"], m["V_q3"]) if "V_d1" in m else 0.0
        if not (np.isfinite(tmean) and np.isfinite(ripple) and np.isfinite(vpk)):
            tmean, ripple, vpk = H.PENALTY, BIG_RIPPLE, BIG_VOLT
    except Exception as e:  # noqa: BLE001
        print(f"  [worker] FEA exception: {e}", flush=True)
        tmean, ripple, vpk = H.PENALTY, BIG_RIPPLE, BIG_VOLT
    design.delete_rotor()
    return tmean, ripple, vpk, Ipk, True


def decoder_from_meta(meta):
    stage = meta["stage"]
    if stage == "stage1":
        return make_decoder("stage1")[1]
    if stage in ("stage2", "probe_eval"):
        return make_decoder("stage2", fixed_geom=np.array(meta["fixed_geom"]))[1]
    if stage == "joint":
        center = {"geom": np.array(meta["center_geom"]), "cur_norm": np.array(meta["center_cur"])}
        return make_decoder("joint", center=center, tr_geom=meta["tr_geom"], tr_cur=meta["tr_cur"])[1]
    if stage == "full16":
        return make_decoder("full16")[1]
    raise ValueError(stage)


def worker_main(args):
    U = np.load(args.shard)["U"]
    meta = json.load(open(args.shard.replace(".npz", ".json")))
    decode = decoder_from_meta(meta)
    mock = meta["mock"]
    design = None if mock else open_isolated_design(args.tag, args.worker_id, args.aedt_version, slots=meta.get("slots", 40), phases=meta.get("phases", 5))
    if not mock:
        set_speed(design, meta.get("fhz"))
        gen = make_generator(design, meta["wide"])
    else:
        class _D:
            rotor_r_min, rotor_r_max = 12.5, 39.275
        gen = make_generator(_D(), meta["wide"])
    lb, ub = H.geom_bounds_arrays(gen)
    for idx in range(args.worker_id, len(U), args.n_workers):
        t, r, v, ipk, ok = eval_one(decode, U[idx], gen, lb, ub, design, 1, mock)
        np.savez(os.path.join(args.resdir, f"res_{idx}.npz"), t=t, r=r, v=v, ipk=ipk, ok=ok)
        print(f"  [w{args.worker_id}] idx {idx}: T={t:.3f} rip={r:.2f} Vpk={v:.1f} Ipk={ipk:.2f}", flush=True)
    if design is not None:
        design.close_project()


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def eval_batch(U, meta, n_workers, tag):
    os.makedirs(OUT, exist_ok=True)
    shard = f"{OUT}/shard_{tag}.npz"
    np.savez(shard, U=np.array(U))
    json.dump(meta, open(f"{OUT}/shard_{tag}.json", "w"))
    resdir = f"{OUT}/res_{tag}"
    os.makedirs(resdir, exist_ok=True)
    for f in os.listdir(resdir):
        os.remove(os.path.join(resdir, f))
    k = min(n_workers, len(U))
    procs = [subprocess.Popen([sys.executable, os.path.abspath(__file__), "--worker",
             "--worker-id", str(w), "--n-workers", str(k), "--shard", shard,
             "--resdir", resdir, "--tag", tag, "--aedt-version", meta["aedt_version"]]) for w in range(k)]
    for p in procs:
        p.wait()
    n = len(U)
    T = np.full(n, H.PENALTY); R = np.full(n, BIG_RIPPLE); V = np.full(n, BIG_VOLT); Ipk = np.full(n, 99.0)
    for idx in range(n):
        f = f"{resdir}/res_{idx}.npz"
        if os.path.exists(f):
            d = np.load(f)
            T[idx], R[idx], V[idx], Ipk[idx] = float(d["t"]), float(d["r"]), float(d["v"]), float(d["ipk"])
    return T, R, V, Ipk


def cand_ok(u, decode, gen, lb, ub, mock):
    """Hard, ANALYTIC feasibility for a candidate: geometry valid + peak current <= I_MAX."""
    gn, dq = decode(u)
    if peak_current_from_dq(*dq) > I_MAX + 1e-6:
        return False
    if mock:
        return bool(np.all(gn >= 0) and np.all(gn <= 1))
    return H.build_barriers(gen, gn, lb, ub) is not None


def fit_clist(U, T, R, V):
    # Sanitize any non-finite values already in the data (a degenerate solve can
    # yield NaN ripple) -> map to penalties so the GPs never see NaN/inf.
    X = torch.tensor(np.array(U))
    Tn = np.nan_to_num(np.asarray(T, float), nan=H.PENALTY, posinf=H.PENALTY, neginf=H.PENALTY)
    Rn = np.nan_to_num(np.asarray(R, float), nan=BIG_RIPPLE, posinf=BIG_RIPPLE, neginf=BIG_RIPPLE)
    Vn = np.nan_to_num(np.asarray(V, float), nan=BIG_VOLT, posinf=BIG_VOLT, neginf=BIG_VOLT)
    gp_t = SingleTaskGP(X, torch.tensor(Tn).unsqueeze(-1), outcome_transform=Standardize(1))
    gp_r = SingleTaskGP(X, torch.tensor(Rn).unsqueeze(-1), outcome_transform=Standardize(1))
    gp_v = SingleTaskGP(X, torch.tensor(Vn).unsqueeze(-1), outcome_transform=Standardize(1))
    model = ModelListGP(gp_t, gp_r, gp_v)
    fit_gpytorch_mll(SumMarginalLogLikelihood(model.likelihood, model))
    return model


def bo_stage(stage, dim, decode, meta, args, ckpt, gen, lb, ub, init_u=None, init_geom_feasible=False):
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    if os.path.exists(ckpt):
        d = np.load(ckpt)
        U, T, R, V, Ipk = list(d["U"]), list(d["T"]), list(d["R"]), list(d["V"]), list(d["Ipk"])
        print(f"[{stage}] resume at {len(T)} evals", flush=True)
    else:
        U, T, R, V, Ipk = [], [], [], [], []

    def draw_feasible():
        for _ in range(500):
            if init_geom_feasible and not args.mock:
                gn = H.rand_feasible_geom_norm(gen, lb, ub)
                u = np.concatenate([gn, np.random.rand(dim - 12)])
            else:
                u = np.random.rand(dim)
            if cand_ok(u, decode, gen, lb, ub, args.mock):
                return u
        return np.random.rand(dim)  # give up -> let it be scored infeasible

    if len(T) < args.n_init:
        seeds = list(init_u) if init_u is not None else []
        while len(seeds) < args.n_init:
            seeds.append(draw_feasible())
        batch = seeds[len(T):args.n_init]
        Tb, Rb, Vb, Ib = eval_batch(batch, meta, args.n_workers, f"{stage}_init")
        U += list(batch); T += list(Tb); R += list(Rb); V += list(Vb); Ipk += list(Ib)
        np.savez(ckpt, U=np.array(U), T=np.array(T), R=np.array(R), V=np.array(V), Ipk=np.array(Ipk))
        nf = int(feasible(T, R, V, Ipk, args.r_max, args.v_max).sum())
        print(f"[{stage}] init {len(T)} evals, {nf} feasible", flush=True)

    bounds = torch.stack([torch.zeros(dim), torch.ones(dim)])
    obj = GenericMCObjective(lambda Z, X=None: Z[..., 0])
    cons = [lambda Z: Z[..., 1] - args.r_max, lambda Z: Z[..., 2] - args.v_max]
    rnd = 0
    while len(T) < args.n_total:
        model = fit_clist(U, T, R, V)
        feas = feasible(T, R, V, Ipk, args.r_max, args.v_max)
        best_f = float(np.max(np.array(T)[feas])) if feas.any() else float(np.min(T))
        acqf = qLogExpectedImprovement(model, best_f=best_f,
                                       sampler=SobolQMCNormalSampler(torch.Size([128])),
                                       objective=obj, constraints=cons)
        # collect q current+geometry-feasible candidates (analytic, no FEA)
        picks = []
        for _ in range(12):
            cand, _ = optimize_acqf(acqf, bounds=bounds, q=args.q, num_restarts=10, raw_samples=256)
            for c in cand.detach().numpy():
                if cand_ok(c, decode, gen, lb, ub, args.mock):
                    picks.append(c)
            if len(picks) >= args.q:
                break
        while len(picks) < args.q:
            picks.append(draw_feasible())
        batch = picks[:args.q]
        Tb, Rb, Vb, Ib = eval_batch(batch, meta, args.n_workers, f"{stage}_r{rnd}")
        U += list(batch); T += list(Tb); R += list(Rb); V += list(Vb); Ipk += list(Ib)
        np.savez(ckpt, U=np.array(U), T=np.array(T), R=np.array(R), V=np.array(V), Ipk=np.array(Ipk))
        feas = feasible(T, R, V, Ipk, args.r_max, args.v_max)
        bf = float(np.max(np.array(T)[feas])) if feas.any() else float("nan")
        print(f"[{stage}] {len(T)}/{args.n_total} feas={int(feas.sum())} best_feasible_T={bf:.3f}", flush=True)
        rnd += 1

    feas = feasible(T, R, V, Ipk, args.r_max, args.v_max)
    if feas.any():
        b = int(np.where(feas)[0][np.argmax(np.array(T)[feas])])
    else:
        b = int(np.argmax(np.nan_to_num(np.asarray(T, float), nan=-1e18)))  # no feasible point
    return np.array(U), np.array(T), np.array(R), np.array(V), np.array(Ipk), b, bool(feas[b])


def base_meta(args):
    return dict(mock=args.mock, wide=args.wide, aedt_version=args.aedt_version,
                fhz=(args.fhz if args.fhz > 0 else None))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["stage1", "stage2", "joint", "crosscheck", "probe_speed"])
    p.add_argument("--worker", action="store_true")
    p.add_argument("--worker-id", type=int, default=0)
    p.add_argument("--n-workers", type=int, default=3)
    p.add_argument("--shard"); p.add_argument("--resdir"); p.add_argument("--tag", default="t")
    p.add_argument("--mock", action="store_true")
    p.add_argument("--wide", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--q", type=int, default=3)
    p.add_argument("--n-init", type=int, default=36)
    p.add_argument("--n-total", type=int, default=120)
    p.add_argument("--r-max", type=float, default=5.0)
    p.add_argument("--v-max", type=float, default=400.0)
    p.add_argument("--fhz", type=float, default=0.0)
    p.add_argument("--aedt-version", default="2024.2")
    p.add_argument("--tr-geom", type=float, default=0.35)
    p.add_argument("--tr-cur", type=float, default=0.35)
    p.add_argument("--out", default="results_h0h1_par")
    args = p.parse_args()

    if args.worker:
        worker_main(args)
        return

    global OUT
    OUT = args.out
    os.makedirs(OUT, exist_ok=True)
    meta = base_meta(args)
    if args.mock:
        class _D:
            rotor_r_min, rotor_r_max = 12.5, 39.275
        gen, gen_design = make_generator(_D(), args.wide), None
    else:
        gen_design = open_isolated_design("seedgen", 99, args.aedt_version)
        set_speed(gen_design, meta["fhz"])
        gen = make_generator(gen_design, args.wide)
    lb, ub = H.geom_bounds_arrays(gen)

    if args.stage == "probe_speed":
        s1 = json.load(open(f"{OUT}/stage1_best.json"))
        gn = np.array(s1["geom_norm"]); dq = np.array(s1["dq"])
        barriers = H.build_barriers(gen, gn, lb, ub)
        gen_design.add_rotor()
        for b in barriers:
            gen_design.add_rotor_barrier(b)
        res = gen_design.compute(*[float(x) for x in dq], NUM_CORES=args.n_workers)
        gen_design.delete_rotor()
        m = res["means"]
        f0 = args.fhz if args.fhz > 0 else 50.0
        wstar = voltage_speed_boundary(m["Flux_d1"], m["Flux_q1"], dq[0], dq[1], R_STATOR, args.v_max, LEW_H)
        out = {"v_max": args.v_max, "psi_d1": m["Flux_d1"], "psi_q1": m["Flux_q1"], "Id1": dq[0], "Iq1": dq[1],
               "V1_at_f0": math.hypot(m["V_d1"], m["V_q1"]), "f0": f0}
        if wstar:
            fstar = wstar / (2 * math.pi)
            out.update({"f_star_hz": fstar, "rpm_star": 60.0 * fstar / POLE_PAIRS})
            print(f"[probe_speed] |V1|({f0}Hz)={out['V1_at_f0']:.1f}V -> f*={fstar:.1f}Hz "
                  f"({out['rpm_star']:.0f} rpm) at V_MAX={args.v_max}", flush=True)
        else:
            print(f"[probe_speed] |V1|({f0}Hz)={out['V1_at_f0']:.1f}V; no boundary <= V_MAX", flush=True)
        json.dump(out, open(f"{OUT}/speed.json", "w"), indent=2)

    elif args.stage == "stage1":
        dim, decode = make_decoder("stage1")
        U, T, R, V, Ipk, b, fb = bo_stage("stage1", dim, decode, {**meta, "stage": "stage1"},
                                          args, f"{OUT}/stage1.npz", gen, lb, ub, init_geom_feasible=True)
        gn, dq = decode(U[b])
        json.dump({"T": float(T[b]), "ripple": float(R[b]), "Vpk": float(V[b]), "Ipk": float(Ipk[b]),
                   "feasible": fb, "geom_norm": gn.tolist(), "dq": [float(x) for x in dq]},
                  open(f"{OUT}/stage1_best.json", "w"), indent=2)
        print(f"[stage1] DONE T={T[b]:.3f} rip={R[b]:.2f} Vpk={V[b]:.1f} Ipk={Ipk[b]:.2f} feas={fb}", flush=True)

    elif args.stage == "stage2":
        s1 = json.load(open(f"{OUT}/stage1_best.json"))
        dim, decode = make_decoder("stage2", fixed_geom=np.array(s1["geom_norm"]))
        # Warm-start: seed with stage1's optimal currents so T_seq >= stage1 (the
        # 4-D current BO must not under-converge below the dq3=0 optimum).
        seed = np.clip(cur_to_norm(np.array(s1["dq"])), 0.0, 1.0)
        U, T, R, V, Ipk, b, fb = bo_stage("stage2", dim, decode,
                                          {**meta, "stage": "stage2", "fixed_geom": s1["geom_norm"]},
                                          args, f"{OUT}/stage2.npz", gen, lb, ub, init_u=[seed])
        gn, dq = decode(U[b])
        json.dump({"T_seq": float(T[b]), "ripple": float(R[b]), "Vpk": float(V[b]), "Ipk": float(Ipk[b]),
                   "feasible": fb, "geom_norm": s1["geom_norm"], "cur_norm": cur_to_norm(dq).tolist(),
                   "dq": [float(x) for x in dq]}, open(f"{OUT}/stage2_best.json", "w"), indent=2)
        print(f"[stage2] DONE T_seq={T[b]:.3f} rip={R[b]:.2f} Vpk={V[b]:.1f} dq={np.round(dq,2)} feas={fb}", flush=True)

    elif args.stage == "joint":
        s2 = json.load(open(f"{OUT}/stage2_best.json"))
        center = {"geom": np.array(s2["geom_norm"]), "cur_norm": np.array(s2["cur_norm"])}
        dim, decode = make_decoder("joint", center=center, tr_geom=args.tr_geom, tr_cur=args.tr_cur)
        m = {**meta, "stage": "joint", "center_geom": s2["geom_norm"], "center_cur": s2["cur_norm"],
             "tr_geom": args.tr_geom, "tr_cur": args.tr_cur}
        U, T, R, V, Ipk, b, fb = bo_stage("joint", dim, decode, m, args, f"{OUT}/joint.npz",
                                          gen, lb, ub, init_u=[0.5 * np.ones(dim)])
        gn, dq = decode(U[b])
        json.dump({"T_seq": s2["T_seq"], "T_joint": float(T[b]), "ripple": float(R[b]), "Vpk": float(V[b]),
                   "feasible": fb, "dT": float(T[b]) - s2["T_seq"],
                   "dT_pct": 100 * (float(T[b]) - s2["T_seq"]) / s2["T_seq"],
                   "geom_move_norm": float(np.linalg.norm(gn - center["geom"])),
                   "geom_joint": gn.tolist(), "dq": [float(x) for x in dq]},
                  open(f"{OUT}/joint_best.json", "w"), indent=2)
        print(f"[joint] DONE T_joint={T[b]:.3f} dT={float(T[b])-s2['T_seq']:.3f} Vpk={V[b]:.1f} feas={fb}", flush=True)

    if gen_design is not None:
        gen_design.close_project()


if __name__ == "__main__":
    main()
