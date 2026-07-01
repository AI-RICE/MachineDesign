"""Nested (bilevel) geometry optimization over an equal-weight multi-point profile.

OUTER: 12-dim GP-BO over the SixLambda rotor geometry (60-slot 5-phase, wide=False).
INNER: per operating point k, minimize copper loss with dq1+dq3 FREED, s.t.
       T >= T_k, ripple <= 5%, Ipk <= 10, at speed f_k.  NO voltage limit (gen-1;
       the P(feasible) voltage constraint is layered on in a later generation).
OBJECTIVE: F(geom) = (1/K) sum_k loss_k*(geom)   (equal-weighted cycle copper loss)

Profile (from the H0/H1 regime points, equal weight):
  P1  20 Nm @ 50   Hz  (cruise; current limit non-binding)
  P2  35 Nm @ 50   Hz  (acceleration; current-limited)
  P3  30 Nm @ 63 Hz    (high speed; voltage-limited later, current-limited now; 63Hz
                        is the R=19 voltage-limited f*, the 71.2Hz value was at wrong R=0.19)

Reuses h0h1_par FEA eval + worker-pool farming. Each worker evaluates a whole
geometry (build rotor once, run the K inner current-optimizations sequentially on
the fixed rotor) and returns F. The inner is a small 4-dim penalized-objective BO.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch

import h0h1_par as P
import h0h1_study as H
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood

ICUR_LB = np.array([0.0, 0.0, -3.0, -3.0])     # dq box: Id1,Iq1 >=0 ; dq3 signed
ICUR_UB = np.array([10.0, 10.0, 3.0, 3.0])
LAM = 20.0                                       # penalty weight (physical units)
R_MAX = 5.0
I_MAX = 10.0
PEN_GEOM = 1.0e3                                 # per-infeasible-point penalty on F


def _score(loss, T, rip, ipk, Tk):
    return (-loss - LAM * max(0.0, Tk - T) - LAM * max(0.0, rip - R_MAX)
            - LAM * max(0.0, ipk - I_MAX))


def inner_map(design, demands, n_map, ncores, seed):
    """ONE current->(T,ripple,Ipk,loss) map for the fixed rotor, then derive the
    min-loss feasible current for EACH torque demand by thresholding the same map.

    Torque & ripple are speed-independent (quasi-static, verified in H0/H1), so all
    points share this single map regardless of their speed. Deriving every demand
    from one evaluated set GUARANTEES monotonicity (lower demand -> <= loss) and
    costs ~3x fewer solves than per-point optimization. The 4-dim BO drives toward
    the min-loss feasible point at the HARDEST demand; its Sobol init + exploration
    populate the lower-current region that serves the easier demands.
    """
    dim = 4
    bounds = torch.stack([torch.zeros(dim), torch.ones(dim)])
    eng = torch.quasirandom.SobolEngine(dim, scramble=True, seed=seed)
    Tmax = max(demands)
    n_init = max(8, n_map // 2)
    U, LOSS, T, RIP, IPK, SC = [], [], [], [], [], []

    def ev(u):
        u = np.clip(np.asarray(u, float), 0.0, 1.0)
        dq = ICUR_LB + u * (ICUR_UB - ICUR_LB)
        try:
            res = design.compute(*[float(x) for x in dq], NUM_CORES=ncores)
            tm, _, rp = H.analyze_results(np.asarray(res["Tor"], float))
            if not (np.isfinite(tm) and np.isfinite(rp)):
                tm, rp = H.PENALTY, P.BIG_RIPPLE
        except Exception as e:  # noqa: BLE001
            print(f"    [map] FEA exc: {e}", flush=True)
            tm, rp = H.PENALTY, P.BIG_RIPPLE
        ipk = float(P.peak_current_from_dq(*dq))
        loss = float(np.sum(np.square(dq)))
        U.append(u); LOSS.append(loss); T.append(tm); RIP.append(rp); IPK.append(ipk)
        SC.append(_score(loss, tm, rp, ipk, Tmax))     # drive toward feasible@Tmax, min loss

    for u in eng.draw(n_init).numpy().tolist():
        ev(u)
    while len(U) < n_map:
        X = torch.tensor(np.array(U)); Y = torch.tensor(np.array(SC)).unsqueeze(-1)
        gp = SingleTaskGP(X, Y, outcome_transform=Standardize(1))
        fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
        acqf = qLogExpectedImprovement(gp, best_f=float(np.max(SC)),
                                       sampler=SobolQMCNormalSampler(torch.Size([64])))
        cand, _ = optimize_acqf(acqf, bounds=bounds, q=1, num_restarts=5, raw_samples=128)
        ev(cand.detach().numpy()[0])

    LOSS = np.array(LOSS); T = np.array(T); RIP = np.array(RIP); IPK = np.array(IPK)
    base = (RIP <= R_MAX) & (IPK <= I_MAX)
    out = []
    for d in demands:
        feas = base & (T >= d)
        if feas.any():
            idxs = np.where(feas)[0]; bi = int(idxs[np.argmin(LOSS[idxs])]); ok = True
        else:
            bi = int(np.argmax(SC)); ok = False           # least-penalized point
        dq = ICUR_LB + U[bi] * (ICUR_UB - ICUR_LB)
        out.append(dict(T_target=float(d), loss=float(LOSS[bi]), dq=[float(x) for x in dq],
                        T=float(T[bi]), ripple=float(RIP[bi]), ipk=float(IPK[bi]), feasible=ok))
    return out


def eval_geometry(design, gen, lb, ub, gn, points, n_inner, ncores, seed):
    barriers = H.build_barriers(gen, np.asarray(gn, float), lb, ub)
    if barriers is None:
        return dict(F=PEN_GEOM * (len(points) + 1), feasible=False, points=[])
    design.add_rotor()
    for b in barriers:
        design.add_rotor_barrier(b)
    # ONE shared, speed-independent torque/ripple map; eval at the first point's speed
    P.set_speed(design, points[0][1])
    demands = [Tk for (Tk, _fk) in points]
    res = inner_map(design, demands, n_inner, ncores, seed)
    design.delete_rotor()
    pts = [dict(fhz=fk, **r) for r, (_Tk, fk) in zip(res, points)]
    n_inf = sum(1 for p in pts if not p["feasible"])
    F = float(np.mean([p["loss"] for p in pts])) + PEN_GEOM * n_inf
    return dict(F=F, feasible=(n_inf == 0), points=pts)


def worker_main(args):
    U = np.load(args.shard)["U"]
    meta = json.load(open(args.shard.replace(".npz", ".json")))
    points = meta["points"]
    design = P.open_isolated_design(args.tag, args.worker_id, args.aedt_version,
                                    slots=meta["slots"], phases=meta["phases"])
    gen = P.make_generator(design, meta["wide"]); lb, ub = H.geom_bounds_arrays(gen)
    for idx in range(args.worker_id, len(U), args.n_workers):
        r = eval_geometry(design, gen, lb, ub, U[idx], points, meta["n_inner"],
                          meta["ncores"], meta["seed"] * 100000 + idx * 17)
        json.dump(r, open(f"{args.resdir}/res_{idx}.json", "w"))
        tag = " ".join(f"P{k+1}={p['loss']:.1f}{'' if p['feasible'] else 'x'}"
                       for k, p in enumerate(r["points"]))
        print(f"  [w{args.worker_id}] geom {idx}: F={r['F']:.3f} feas={r['feasible']} {tag}", flush=True)
    design.close_project()


def eval_batch(U, meta, n_workers, tag, out):
    shard = f"{out}/shard_{tag}.npz"; np.savez(shard, U=np.array(U))
    json.dump(meta, open(f"{out}/shard_{tag}.json", "w"))
    resdir = f"{out}/res_{tag}"; os.makedirs(resdir, exist_ok=True)
    for f in os.listdir(resdir):
        os.remove(os.path.join(resdir, f))
    k = min(n_workers, len(U))
    procs = [subprocess.Popen([sys.executable, os.path.abspath(__file__), "--worker",
             "--worker-id", str(w), "--n-workers", str(k), "--shard", shard,
             "--resdir", resdir, "--tag", tag, "--aedt-version", meta["aedt_version"]])
             for w in range(k)]
    for p in procs:
        p.wait()
    F = np.full(len(U), PEN_GEOM * (len(meta["points"]) + 1)); details = [None] * len(U)
    for idx in range(len(U)):
        f = f"{resdir}/res_{idx}.json"
        if os.path.exists(f):
            d = json.load(open(f)); F[idx] = d["F"]; details[idx] = d
    return F, details


def save_best(U, F, best_det, out, points):
    bi = int(np.argmin(F))
    rec = dict(F=float(F[bi]), geom_norm=[float(x) for x in U[bi]],
               points=points, detail=best_det)
    json.dump(rec, open(f"{out}/nested_best.json", "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--n-workers", type=int, default=8)
    ap.add_argument("--shard"); ap.add_argument("--resdir"); ap.add_argument("--tag", default="t")
    ap.add_argument("--aedt-version", default="2024.2")
    ap.add_argument("--out", default="results/nested")
    ap.add_argument("--points", default="20:50,35:50,30:63", help="T:fhz comma-list")
    ap.add_argument("--n-init", type=int, default=16)
    ap.add_argument("--n-total", type=int, default=60)
    ap.add_argument("--q", type=int, default=8)
    ap.add_argument("--n-inner", type=int, default=24)
    ap.add_argument("--ncores", type=int, default=1)
    ap.add_argument("--slots", type=int, default=60)
    ap.add_argument("--phases", type=int, default=5)
    ap.add_argument("--wide", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.worker:
        return worker_main(args)

    P.OUT = args.out
    os.makedirs(args.out, exist_ok=True)
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    points = [[float(x) for x in p.split(":")] for p in args.points.split(",")]
    meta = dict(points=points, n_inner=args.n_inner, ncores=args.ncores, slots=args.slots,
                phases=args.phases, wide=args.wide, aedt_version=args.aedt_version, seed=args.seed)
    print(f"[nested] profile: {points}  (equal weight, dq3 freed, Ipk<=10, no V limit)", flush=True)

    gd = P.open_isolated_design("nseed", 98, args.aedt_version, slots=args.slots, phases=args.phases)
    P.set_speed(gd, points[0][1])
    gen = P.make_generator(gd, args.wide); lb, ub = H.geom_bounds_arrays(gen)
    dim = 12; bounds = torch.stack([torch.zeros(dim), torch.ones(dim)])

    def geom_ok(gn):
        return H.build_barriers(gen, np.asarray(gn, float), lb, ub) is not None

    def draw():
        for _ in range(500):
            gn = H.rand_feasible_geom_norm(gen, lb, ub)
            if geom_ok(gn):
                return np.asarray(gn, float)
        return np.random.rand(dim)

    ckpt = f"{args.out}/nested.npz"
    best_det = None
    if os.path.exists(ckpt):
        d = np.load(ckpt, allow_pickle=True); U = list(d["U"]); F = list(d["F"])
        print(f"[nested] resume at {len(F)}", flush=True)
    else:
        U, F = [], []

    if len(F) < args.n_init:
        seeds = [draw() for _ in range(args.n_init - len(F))]
        Fb, det = eval_batch(seeds, meta, args.n_workers, "nest_init", args.out)
        U += seeds; F += list(Fb)
        bi = int(np.argmin(Fb))
        if det[bi] is not None:
            best_det = det[bi]
        np.savez(ckpt, U=np.array(U), F=np.array(F))
        save_best(U, F, best_det, args.out, points)
        print(f"[nested] init {len(F)} evals  best_F={np.min(F):.3f}", flush=True)

    rnd = 0
    while len(F) < args.n_total:
        X = torch.tensor(np.array(U)); Y = torch.tensor(-np.array(F)).unsqueeze(-1)   # maximize -F
        gp = SingleTaskGP(X, Y, outcome_transform=Standardize(1))
        fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
        acqf = qLogExpectedImprovement(gp, best_f=float(np.max(-np.array(F))),
                                       sampler=SobolQMCNormalSampler(torch.Size([128])))
        picks = []
        for _ in range(12):
            cand, _ = optimize_acqf(acqf, bounds=bounds, q=args.q, num_restarts=10, raw_samples=256)
            for c in cand.detach().numpy():
                if geom_ok(c):
                    picks.append(c)
            if len(picks) >= args.q:
                break
        while len(picks) < args.q:
            picks.append(draw())
        batch = picks[:args.q]
        Fb, det = eval_batch(batch, meta, args.n_workers, f"nest_r{rnd}", args.out)
        U += batch; F += list(Fb)
        bi = int(np.argmin(Fb))
        if det[bi] is not None and Fb[bi] <= np.min(F):
            best_det = det[bi]
        np.savez(ckpt, U=np.array(U), F=np.array(F))
        save_best(U, F, best_det, args.out, points)
        nfeas = sum(1 for x in F if x < PEN_GEOM)
        print(f"[nested] {len(F)}/{args.n_total} feas={nfeas} best_F={np.min(F):.3f}", flush=True)
        rnd += 1

    bi = int(np.argmin(F))
    print(f"[nested] DONE best_F={F[bi]:.3f}  geom#{bi}", flush=True)
    if best_det is not None:
        for k, p in enumerate(best_det["points"]):
            print(f"  P{k+1} T>={p['T_target']}@{p['fhz']}Hz: loss={p['loss']:.2f} "
                  f"T={p['T']:.1f} rip={p['ripple']:.2f}% Ipk={p['ipk']:.2f} "
                  f"dq={[round(x,2) for x in p['dq']]} feas={p['feasible']}", flush=True)


if __name__ == "__main__":
    main()
