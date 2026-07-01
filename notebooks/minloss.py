"""No-limit (efficiency) anchor for the H0/H1 study: minimize copper loss at a
prescribed torque, with NO current or voltage ceiling. This is the meaningful
'no limitation' regime (max-torque is unbounded without limits).

  min_{g,i}  J = Id1^2+Iq1^2+Id3^2+Iq3^2   (proportional to sum-of-squares of the
                                            current waveform = copper loss)
  s.t.  T(g,i) >= T_TARGET,   ripple(g,i) <= R_MAX,   f = 50 Hz

The loss J is ANALYTIC in the dq setpoint (no FEA); torque and ripple are FEA and
modelled by GPs. Reuses the h0h1_par worker pool / FEA eval. mode=dq1 fixes dq3=0
(H0 baseline geometry); mode=joint frees all four dq (H1).
"""
import argparse
import json
import os

import numpy as np
import torch

import h0h1_par as P
import h0h1_study as H
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective
from botorch.fit import fit_gpytorch_mll
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms import Standardize
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.mlls import SumMarginalLogLikelihood


def loss_of(u, decode):
    _, dq = decode(u)
    return float(np.sum(np.square(dq)))          # copper loss (R-scaled), analytic


def geom_ok(u, decode, gen, lb, ub):
    gn, _ = decode(u)
    return H.build_barriers(gen, gn, lb, ub) is not None   # geometry only -- NO current cap


def fit_jtr(U, J, T, R):
    X = torch.tensor(np.array(U))
    gj = SingleTaskGP(X, torch.tensor(np.nan_to_num(J, nan=1e6, posinf=1e6)).unsqueeze(-1), outcome_transform=Standardize(1))
    gt = SingleTaskGP(X, torch.tensor(np.nan_to_num(T, nan=H.PENALTY, posinf=H.PENALTY, neginf=H.PENALTY)).unsqueeze(-1), outcome_transform=Standardize(1))
    gr = SingleTaskGP(X, torch.tensor(np.nan_to_num(R, nan=P.BIG_RIPPLE, posinf=P.BIG_RIPPLE)).unsqueeze(-1), outcome_transform=Standardize(1))
    m = ModelListGP(gj, gt, gr)
    fit_gpytorch_mll(SumMarginalLogLikelihood(m.likelihood, m))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dq1", "joint"], default="dq1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--t-target", type=float, default=20.0)
    ap.add_argument("--r-max", type=float, default=5.0)
    ap.add_argument("--fhz", type=float, default=50.0)
    ap.add_argument("--q", type=int, default=8)
    ap.add_argument("--n-workers", type=int, default=8)
    ap.add_argument("--n-init", type=int, default=36)
    ap.add_argument("--n-total", type=int, default=120)
    ap.add_argument("--wide", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--aedt-version", default="2024.2")
    ap.add_argument("--slots", type=int, default=40, help="stator slot count (40=Design2, 60=Design2_60)")
    ap.add_argument("--phases", type=int, default=5, choices=[3, 5],
                    help="3=Design_60 (3-phase 60-slot), 5=Design2/Design2_60")
    args = ap.parse_args()
    P.OUT = args.out
    os.makedirs(args.out, exist_ok=True)
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    stage = "stage1" if args.mode == "dq1" else "full16"
    dim, decode = P.make_decoder(stage)
    meta = dict(stage=stage, mock=False, wide=args.wide, aedt_version=args.aedt_version,
                fhz=args.fhz, slots=args.slots, phases=args.phases)

    gd = P.open_isolated_design("seedgen", 99, args.aedt_version, slots=args.slots, phases=args.phases)
    P.set_speed(gd, args.fhz)
    gen = P.make_generator(gd, args.wide)
    lb, ub = H.geom_bounds_arrays(gen)

    def draw():
        for _ in range(500):
            gn = H.rand_feasible_geom_norm(gen, lb, ub)
            u = np.concatenate([gn, np.random.rand(dim - 12)])
            if geom_ok(u, decode, gen, lb, ub):
                return u
        return np.random.rand(dim)

    def evalU(batch, tag):
        Tb, Rb, _Vb, _Ib = P.eval_batch(batch, meta, args.n_workers, tag)
        Jb = np.array([loss_of(u, decode) for u in batch])
        return Tb, Rb, Jb

    ckpt = f"{args.out}/{args.mode}.npz"
    if os.path.exists(ckpt):
        d = np.load(ckpt); U, T, R, J = list(d["U"]), list(d["T"]), list(d["R"]), list(d["J"])
        print(f"[minloss-{args.mode}] resume at {len(T)}", flush=True)
    else:
        U, T, R, J = [], [], [], []

    if len(T) < args.n_init:
        seeds = [draw() for _ in range(args.n_init - len(T))]
        Tb, Rb, Jb = evalU(seeds, f"{args.mode}_init")
        U += seeds; T += list(Tb); R += list(Rb); J += list(Jb)
        np.savez(ckpt, U=np.array(U), T=np.array(T), R=np.array(R), J=np.array(J))
        print(f"[minloss-{args.mode}] init {len(T)} evals", flush=True)

    bounds = torch.stack([torch.zeros(dim), torch.ones(dim)])
    # PENALIZED scalar objective (not hard constraints): the optimum of min-loss
    # s.t. T>=target sits ON the lower torque boundary, where constrained-EI's
    # P(feasible)~0.5 shuns it. A penalty drives the optimum to exactly T=target,
    # least loss. dJ/dT ~ 2 (reluctance torque ~ I^2 ~ J), so lambda=20 >> dominates.
    LAM_T, LAM_R = 20.0, 20.0
    obj = GenericMCObjective(lambda Z, X=None:
        -Z[..., 0]
        - LAM_T * torch.clamp(args.t_target - Z[..., 1], min=0.0)
        - LAM_R * torch.clamp(Z[..., 2] - args.r_max, min=0.0))

    def feas(T, R):
        return (np.asarray(T) >= args.t_target) & (np.asarray(R) <= args.r_max)

    def score(Jv, Tv, Rv):
        return (-np.asarray(Jv)
                - LAM_T * np.clip(args.t_target - np.asarray(Tv), 0, None)
                - LAM_R * np.clip(np.asarray(Rv) - args.r_max, 0, None))

    rnd = 0
    while len(T) < args.n_total:
        m = fit_jtr(U, J, T, R)
        best = float(np.max(score(J, T, R)))
        acqf = qLogExpectedImprovement(m, best_f=best, sampler=SobolQMCNormalSampler(torch.Size([128])),
                                       objective=obj)
        picks = []
        for _ in range(12):
            cand, _ = optimize_acqf(acqf, bounds=bounds, q=args.q, num_restarts=10, raw_samples=256)
            for c in cand.detach().numpy():
                if geom_ok(c, decode, gen, lb, ub):
                    picks.append(c)
            if len(picks) >= args.q:
                break
        while len(picks) < args.q:
            picks.append(draw())
        batch = picks[:args.q]
        Tb, Rb, Jb = evalU(batch, f"{args.mode}_r{rnd}")
        U += batch; T += list(Tb); R += list(Rb); J += list(Jb)
        np.savez(ckpt, U=np.array(U), T=np.array(T), R=np.array(R), J=np.array(J))
        f = feas(T, R)
        bj = float(np.min(np.array(J)[f])) if f.any() else float("nan")
        print(f"[minloss-{args.mode}] {len(T)}/{args.n_total} feas={int(f.sum())} best_loss={bj:.3f}", flush=True)
        rnd += 1

    f = feas(T, R)
    b = int(np.where(f)[0][np.argmin(np.array(J)[f])]) if f.any() else int(np.argmin(J))
    gn, dq = decode(U[b])
    best = {"mode": args.mode, "t_target": args.t_target, "loss": float(J[b]), "Irms_equiv": float(np.sqrt(J[b])),
            "T": float(T[b]), "ripple": float(R[b]), "Ipk": float(P.peak_current_from_dq(*dq)),
            "feasible": bool(f[b]), "geom_norm": [float(x) for x in gn], "dq": [float(x) for x in dq]}
    json.dump(best, open(f"{args.out}/{args.mode}_best.json", "w"), indent=2)
    print(f"[minloss-{args.mode}] DONE loss={J[b]:.2f} (|I|={best['Irms_equiv']:.2f}) T={T[b]:.2f} "
          f"rip={R[b]:.2f} Ipk={best['Ipk']:.2f} dq={[round(x, 2) for x in dq]}", flush=True)
    gd.close_project()


if __name__ == "__main__":
    main()
