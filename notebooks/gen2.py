"""Gen-2: joint-GP composite constrained BILEVEL optimizer (OUR METHOD).
See Reluctance5f/docs/sota/bilevel-composite-bo.md and [[sota/bilevel-composite-bo]].

Contrast with gen-1 (nested.py, the naive nested black box that discards every
(g,I)->(T,ripple) FEA point): here ONE joint GP models the intermediate responses
T(g,I) and ripple(g,I) over the full (geometry x current) space, POOLING all FEA. Each
operating point's min-loss is derived ON THE POSTERIOR (loss = |I|^2 is analytic), so a
good current for one geometry informs its neighbours. The outer proposes geometries by
approximate Thompson sampling of the bilevel value function F(g) = mean_k loss_k*(g);
we then FEA-evaluate the chosen geometries' per-demand inner-optimal currents and fold
them back into the one GP.

Constraints (this build, = gen-1 for a fair F comparison): T>=T_k, ripple<=R_MAX,
Ipk<=I_MAX. Torque/ripple are speed-independent, so ONE (T,ripple)(g,I) surface serves
all K demands (threshold per demand). Voltage P(feasible) is the next add (a 3rd GP output
V(g,I) + LogCEI factor).

Profile default: P1 20Nm, P2 35Nm, P3 30Nm (speeds only matter once voltage is added).
"""
import argparse
import json
import math
import os
import subprocess
import sys

import numpy as np
import torch

import h0h1_par as P
import h0h1_study as H
from botorch.fit import fit_gpytorch_mll
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms import Standardize
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import SumMarginalLogLikelihood
from gpytorch.priors import LogNormalPrior

ICUR_LB = np.array([0.0, 0.0, -3.0, -3.0])
ICUR_UB = np.array([10.0, 10.0, 3.0, 3.0])
R_MAX = 5.0
I_MAX = 10.0
DIM_G = 12
DIM_I = 4
DIM = DIM_G + DIM_I
BIG_LOSS = 1.0e3
SQRT2, SQRT3 = math.sqrt(2.0), math.sqrt(3.0)


# --------------------------------------------------------------------------- #
# joint GP over (g, I)  ->  T,  ripple   (DSP sqrt(D)-scaled lengthscale prior)
# --------------------------------------------------------------------------- #
def dsp_gp(X, Y):
    d = X.shape[-1]
    prior = LogNormalPrior(loc=SQRT2 + math.log(d) * 0.5, scale=SQRT3)
    kern = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=d, lengthscale_prior=prior,
                                    lengthscale_constraint=GreaterThan(2.5e-2, transform=None,
                                                                       initial_value=prior.mode)))
    return SingleTaskGP(X, Y, covar_module=kern, outcome_transform=Standardize(m=1))


def fit_joint(XI, T, PTP):
    """Joint GP over (g,i) modelling torque T and ABSOLUTE torque ripple PTP (Nm ptp).
    We model ptp, NOT ripple% (=ptp/mean): ripple% is unbounded/non-smooth near zero
    torque and a GP cannot fit it (CV RMSE > signal sd); ptp is smooth. See
    docs/surrogate-validation.md."""
    X = torch.tensor(np.asarray(XI))
    gt = dsp_gp(X, torch.tensor(np.nan_to_num(T, nan=H.PENALTY)).unsqueeze(-1))
    gp = dsp_gp(X, torch.tensor(np.nan_to_num(PTP, nan=1.0e3)).unsqueeze(-1))
    m = ModelListGP(gt, gp)
    fit_gpytorch_mll(SumMarginalLogLikelihood(m.likelihood, m))
    return m


def ptp_of(R, T):
    """absolute torque ripple (Nm) from ripple% and mean torque"""
    return np.asarray(R, float) * np.asarray(T, float) / 100.0


def ipk_of(dq):
    return float(P.peak_current_from_dq(*dq))


# --------------------------------------------------------------------------- #
# FEA evaluation of a batch of (geometry, current) jobs, grouped by geometry
# --------------------------------------------------------------------------- #
def eval_batch(jobs, meta, n_workers, tag, out):
    """jobs: list of (gid, gn(list,12), dq(list,4)). Returns dict gid->list of
    (dq, T, ripple, ipk). Workers group by gid (build each rotor once)."""
    shard = f"{out}/shard_{tag}.json"
    json.dump({"jobs": [[int(g), list(map(float, gn)), list(map(float, dq))] for g, gn, dq in jobs],
               "meta": meta}, open(shard, "w"))
    resdir = f"{out}/res_{tag}"; os.makedirs(resdir, exist_ok=True)
    for f in os.listdir(resdir):
        os.remove(os.path.join(resdir, f))
    k = min(n_workers, max(1, len(jobs)))
    procs = [subprocess.Popen([sys.executable, os.path.abspath(__file__), "--worker",
             "--worker-id", str(w), "--n-workers", str(k), "--shard", shard,
             "--resdir", resdir, "--tag", tag, "--aedt-version", meta["aedt_version"]])
             for w in range(k)]
    for p in procs:
        p.wait()
    res = {}
    for w in range(k):
        f = f"{resdir}/res_w{w}.json"
        if os.path.exists(f):
            for gid_s, rows in json.load(open(f)).items():
                res.setdefault(int(gid_s), []).extend(rows)
    return res


def worker_main(args):
    d = json.load(open(args.shard)); meta = d["meta"]
    jobs = d["jobs"]
    design = P.open_isolated_design(args.tag, args.worker_id, args.aedt_version,
                                    slots=meta["slots"], phases=meta["phases"])
    gen = P.make_generator(design, meta["wide"]); lb, ub = H.geom_bounds_arrays(gen)
    P.set_speed(design, meta["fhz"])                       # one speed: T/ripple speed-indep
    # this worker's geometries = those whose gid % k == worker_id (contiguous rotor reuse)
    by_g = {}
    for gid, gn, dq in jobs:
        by_g.setdefault(gid, (gn, []))[1].append(dq)
    out = {}
    for gid in sorted(by_g):
        if gid % args.n_workers != args.worker_id:
            continue
        gn, dqs = by_g[gid]
        barriers = H.build_barriers(gen, np.asarray(gn, float), lb, ub)
        rows = []; nok = 0
        if barriers is not None:
            design.add_rotor()
            for b in barriers:
                design.add_rotor_barrier(b)
            for dq in dqs:
                ok = True
                try:
                    r = design.compute(*[float(x) for x in dq], NUM_CORES=meta["ncores"])
                    if r is None:
                        raise ValueError("compute returned None (solve produced no data)")
                    tm, _, rp = H.analyze_results(np.asarray(r["Tor"], float))
                    if not (np.isfinite(tm) and np.isfinite(rp)):
                        raise ValueError("non-finite T/ripple")
                except Exception as e:  # noqa: BLE001
                    print(f"  [w{args.worker_id}] FEA FAIL g{gid}: {e}", flush=True)
                    tm, rp, ok = H.PENALTY, P.BIG_RIPPLE, False
                nok += int(ok)
                # ok flag (5th field): failed solves are DROPPED at collection so a silent
                # solver-license failure never poisons the joint GP with T=0/ripple=999.
                rows.append([dq, float(tm), float(rp), ipk_of(dq), int(ok)])
            design.delete_rotor()
        out[gid] = rows
        print(f"  [w{args.worker_id}] g{gid}: {nok}/{len(rows)} ok", flush=True)
    json.dump(out, open(f"{args.resdir}/res_w{args.worker_id}.json", "w"))
    design.close_project()


# --------------------------------------------------------------------------- #
# posterior inner: min-loss per demand from a (T,ripple) sample over I-candidates
# --------------------------------------------------------------------------- #
LAM = 20.0  # penalty weight (physical units), same as minloss


def inner_from_surface(Tsurf, PTPsurf, Icand, ipk_cand, loss_cand, demands):
    """Tsurf: predicted torque; PTPsurf: predicted ABSOLUTE ripple (Nm ptp) over the
    I-candidates. Ripple% <= R_MAX is enforced smoothly as ptp <= (R_MAX/100)*T_k
    (conservative: feasible => T>=T_k, so actual ripple% = ptp/T <= ptp/T_k <= R_MAX%).
    PENALIZED per-demand min (mirrors minloss). Returns (F_penalized, idxs, feas flags)."""
    pen_ipk = LAM * np.clip(ipk_cand - I_MAX, 0, None)
    Fs, idxs, feas = [], [], []
    for Tk in demands:
        ptp_max = (R_MAX / 100.0) * Tk
        pen = (loss_cand + pen_ipk + LAM * np.clip(Tk - Tsurf, 0, None)
               + LAM * np.clip(PTPsurf - ptp_max, 0, None))
        j = int(np.argmin(pen))
        Fs.append(float(pen[j])); idxs.append(j)
        feas.append(bool(Tsurf[j] >= Tk and PTPsurf[j] <= ptp_max and ipk_cand[j] <= I_MAX))
    return float(np.mean(Fs)), idxs, feas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--n-workers", type=int, default=16)
    ap.add_argument("--shard"); ap.add_argument("--resdir"); ap.add_argument("--tag", default="t")
    ap.add_argument("--aedt-version", default="2024.2")
    ap.add_argument("--out", default="results/gen2")
    ap.add_argument("--demands", default="20,35,30", help="torque demands [Nm], comma list")
    ap.add_argument("--fhz", type=float, default=50.0)
    ap.add_argument("--n-seed", type=int, default=64, help="seed (g,I) FEA points")
    ap.add_argument("--n-rounds", type=int, default=8)
    ap.add_argument("--q", type=int, default=8, help="geometries proposed per round")
    ap.add_argument("--n-gcand", type=int, default=256, help="outer geometry candidate pool")
    ap.add_argument("--n-icand", type=int, default=512, help="inner current candidate pool (64 was too coarse; see surrogate-validation.md)")
    ap.add_argument("--ncores", type=int, default=1)
    ap.add_argument("--slots", type=int, default=60)
    ap.add_argument("--phases", type=int, default=5)
    ap.add_argument("--wide", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.worker:
        return worker_main(args)

    os.makedirs(args.out, exist_ok=True)
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    demands = [float(x) for x in args.demands.split(",")]
    meta = dict(slots=args.slots, phases=args.phases, wide=args.wide, fhz=args.fhz,
                ncores=args.ncores, aedt_version=args.aedt_version)
    print(f"[gen2] demands={demands} joint-GP over (g,I) dim={DIM}", flush=True)

    gd = P.open_isolated_design("g2seed", 97, args.aedt_version, slots=args.slots, phases=args.phases)
    gen = P.make_generator(gd, args.wide); lb, ub = H.geom_bounds_arrays(gen)

    def feasible_g():
        for _ in range(2000):
            gn = np.asarray(H.rand_feasible_geom_norm(gen, lb, ub), float)
            if H.build_barriers(gen, gn, lb, ub) is not None:
                return gn
        return np.random.rand(DIM_G)

    def sobol_I(n, seed):
        u = torch.quasirandom.SobolEngine(DIM_I, scramble=True, seed=seed).draw(n).numpy()
        return u  # unit-cube; decode with ICUR_LB/UB where needed

    def dq_of(u_i):
        return ICUR_LB + np.asarray(u_i) * (ICUR_UB - ICUR_LB)

    # ---- pooled data: X in unit cube (g_norm[0,1]^12 , I_norm[0,1]^4), plus T,R ----
    ckpt = f"{args.out}/gen2.npz"
    if os.path.exists(ckpt):
        z = np.load(ckpt, allow_pickle=True)
        Xg = list(z["Xg"]); Xi = list(z["Xi"]); T = list(z["T"]); R = list(z["R"])
        print(f"[gen2] resume {len(T)} FEA points", flush=True)
    else:
        Xg, Xi, T, R = [], [], [], []

    def run_jobs(pairs, tag):
        """pairs: list of (gn[12], u_i[4]); FEA-evaluate; append to pool.
        Currents are grouped by UNIQUE geometry so each rotor is built once."""
        g_list = []; g_key = {}; jobs = []
        for gn, ui in pairs:
            kk = tuple(np.round(np.asarray(gn, float), 6))
            if kk not in g_key:
                g_key[kk] = len(g_list); g_list.append(np.asarray(gn, float))
            jobs.append((g_key[kk], list(map(float, gn)), list(dq_of(ui))))
        res = eval_batch(jobs, meta, args.n_workers, tag, args.out)
        n_ok = n_fail = 0
        for gid, rows in res.items():
            gn = g_list[gid]
            for row in rows:
                dq, tm, rp = row[0], row[1], row[2]
                ok = row[4] if len(row) > 4 else 1
                if not ok:                        # drop failed solves — never poison the GP
                    n_fail += 1; continue
                ui = (np.asarray(dq, float) - ICUR_LB) / (ICUR_UB - ICUR_LB)
                Xg.append(gn); Xi.append(ui); T.append(tm); R.append(rp); n_ok += 1
        np.savez(ckpt, Xg=np.array(Xg), Xi=np.array(Xi), T=np.array(T), R=np.array(R))
        if n_fail:
            print(f"[gen2] {tag}: {n_ok} ok, {n_fail} failed solves dropped", flush=True)

    # ---- seed: space-filling (g, I); top up if solves were dropped ----
    st = 0
    while len(T) < args.n_seed and st < 5:
        need = args.n_seed - len(T)
        gseed = [feasible_g() for _ in range(need)]
        iseed = sobol_I(need, args.seed + 1 + st)
        run_jobs(list(zip(gseed, iseed)), f"seed{st}")
        print(f"[gen2] seed pass {st}: {len(T)}/{args.n_seed} FEA points", flush=True)
        st += 1

    # candidate pools (fixed across rounds)
    Icand_u = sobol_I(args.n_icand, args.seed + 7)
    Icand_dq = np.array([dq_of(u) for u in Icand_u])
    ipk_cand = np.array([ipk_of(dq) for dq in Icand_dq])
    loss_cand = np.sum(Icand_dq ** 2, axis=1)

    def best_confirmed():
        """From FEA-CONFIRMED data (grouped by geometry): the best hard-feasible mean-loss
        (headline; inf/BIG if no geometry meets all K demands), the best PENALIZED F
        (progress metric, always finite), and the best geometry."""
        Xgn = np.array(Xg); key = {}
        for idx in range(len(T)):
            key.setdefault(tuple(np.round(Xgn[idx], 6)), []).append(idx)
        bestHard, bestPen, bestg, n_feas = float("inf"), float("inf"), None, 0
        for g, idxs in key.items():
            dq = np.array([dq_of(Xi[i]) for i in idxs])
            tt = np.array([T[i] for i in idxs]); rr = np.array([R[i] for i in idxs])
            ll = np.sum(dq ** 2, axis=1); ip = np.array([ipk_of(dq[j]) for j in range(len(dq))])
            base = (rr <= R_MAX) & (ip <= I_MAX)
            penc = LAM * np.clip(rr - R_MAX, 0, None) + LAM * np.clip(ip - I_MAX, 0, None)
            Fk_hard, Fk_pen = [], []
            for Tk in demands:
                fe = base & (tt >= Tk)
                Fk_hard.append(float(np.min(ll[fe])) if fe.any() else BIG_LOSS)
                Fk_pen.append(float(np.min(ll + penc + LAM * np.clip(Tk - tt, 0, None))))
            Fh, Fp = float(np.mean(Fk_hard)), float(np.mean(Fk_pen))
            if Fh < BIG_LOSS:
                n_feas += 1
            if Fh < bestHard or (bestHard >= BIG_LOSS and Fp < bestPen):
                bestHard, bestg = min(Fh, bestHard), np.array(g)
            bestPen = min(bestPen, Fp)
        return bestHard, bestPen, bestg, n_feas

    for rnd in range(args.n_rounds):
        m = fit_joint(np.hstack([np.array(Xg), np.array(Xi)]), np.array(T),
                      ptp_of(np.array(R), np.array(T)))
        Gcand = np.array([feasible_g() for _ in range(args.n_gcand)])
        # build the (Gcand x Icand) query grid once
        GX = np.repeat(Gcand, args.n_icand, axis=0)
        IX = np.tile(Icand_u, (args.n_gcand, 1))
        Xq = torch.tensor(np.hstack([GX, IX]))
        with torch.no_grad():
            post = m.posterior(Xq)
            mu = post.mean.numpy(); sd = np.sqrt(post.variance.numpy())
        muT = mu[:, 0].reshape(args.n_gcand, args.n_icand)
        muPTP = mu[:, 1].reshape(args.n_gcand, args.n_icand)
        sdT = sd[:, 0].reshape(args.n_gcand, args.n_icand)
        sdPTP = sd[:, 1].reshape(args.n_gcand, args.n_icand)

        # approximate-Thompson: q draws, each picks argmin_g sampled F(g)
        picks = []
        for dd in range(args.q):
            epsT = np.random.randn(args.n_gcand, args.n_icand)
            epsP = np.random.randn(args.n_gcand, args.n_icand)
            Ts = muT + sdT * epsT; PTPs = np.clip(muPTP + sdPTP * epsP, 0, None)
            Fg = np.array([inner_from_surface(Ts[gi], PTPs[gi], Icand_u, ipk_cand, loss_cand, demands)[0]
                           for gi in range(args.n_gcand)])
            gi = int(np.argmin(Fg))
            picks.append(gi)
        picks = list(dict.fromkeys(picks))  # dedup, keep order

        # for each picked geometry: per-demand inner-optimal currents under the POSTERIOR MEAN
        pairs = []
        for gi in picks:
            _, idxs, _ = inner_from_surface(muT[gi], muPTP[gi], Icand_u, ipk_cand, loss_cand, demands)
            for j in sorted(set(idxs)):           # penalized argmin per demand (always valid)
                pairs.append((Gcand[gi], Icand_u[j]))
        run_jobs(pairs, f"r{rnd}")
        bH, bP, _, nf = best_confirmed()
        hs = f"{bH:.3f}" if bH < BIG_LOSS else "none-feasible"
        print(f"[gen2] round {rnd}: +{len(pairs)} FEA (total {len(T)}), "
              f"best_feasible_F={hs}  best_penalized_F={bP:.3f}  feas_geoms={nf}", flush=True)

    bH, bP, bg, nf = best_confirmed()
    hs = f"{bH:.3f}" if bH < BIG_LOSS else "none-feasible"
    print(f"[gen2] DONE best_feasible_F={hs} best_penalized_F={bP:.3f} feas_geoms={nf} "
          f"after {len(T)} FEA points", flush=True)
    if bg is not None:
        json.dump({"F_feasible": (bH if bH < BIG_LOSS else None), "F_penalized": bP,
                   "geom_norm": [float(x) for x in bg], "demands": demands,
                   "n_fea": len(T), "feas_geoms": nf}, open(f"{args.out}/gen2_best.json", "w"), indent=2)


if __name__ == "__main__":
    main()
