"""Gen-3: pathwise-Thompson multi-objective bilevel optimizer (OUR METHOD, v3).
See Reluctance5f/docs/sota/moo-bo-acquisitions.md and [[sota/moo-bo-acquisitions]].

vs gen-2 (gen2.py): (1) COHERENT Matheron/RFF sample paths replace gen-2's incoherent
`mu + sd*eps` (the fragility fix); (2) RIPPLE AS AN OBJECTIVE (Pareto option A) -- the inner
minimises loss s.t. sampled T>=T_k and Ipk<=I_MAX only, ripple read off at the loss-optimal
current; (3) HV-greedy batch over the union of per-path Pareto fronts; (4) variance monitoring
(per-round Monte-Carlo HV spread) + post-run S-sweep (diag_gen3.py).

PROCESS SPLIT (important, 2026-07-04): the GP fit + Matheron sampling + selection run in a
SEPARATE clean process (gen3_select.py) that imports no PyAEDT. gen-3's main process holds an
open AEDT/gRPC session for geometry generation, and torch's Matheron/autograd machinery
segfaults when run in that same process (torch vs ANSYS native BLAS/OpenMP). So main only:
generate geometry candidates (AEDT) -> hand GP data + candidates to gen3_select as a subprocess
-> FEA-evaluate the returned picks (workers, own AEDT). Selection is torch-isolated.

Profile default (500 W machine): P1 city 4, P2 accel(current-limited) 8, P3 high-speed 6 Nm.
"""
import argparse
import json
import math
import os
import subprocess
import sys

import numpy as np
import torch

import gen2  # FEA side only: eval_batch/worker dispatch, ipk_of; imports torch (import-safe w/ AEDT)
import h0h1_par as P
import h0h1_study as H

# --- machine electrical spec (gen-3 owns this; gen2's frozen constants left untouched) ---
# 500 W machine: peak phase current 1.3 A, end-winding leakage Lew = 2.4 mH.
I_MAX = 1.3                                     # A, peak phase current limit (was 10.0)
ICUR_LB = np.array([0.0, 0.0, -0.3 * I_MAX, -0.3 * I_MAX])   # fundamental [0,Imax], 3rd harm [±0.3 Imax]
ICUR_UB = np.array([I_MAX, I_MAX, 0.3 * I_MAX, 0.3 * I_MAX])
DIM_G, DIM_I = gen2.DIM_G, gen2.DIM_I
LAM = gen2.LAM
SELECT_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gen3_select.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/gen3")
    ap.add_argument("--demands", default="4,8,6")
    ap.add_argument("--speeds", default="25,16,63", help="per-demand ELECTRICAL Hz (voltage)")
    ap.add_argument("--v-max", type=float, default=400.0, help="peak phase voltage limit [V]")
    ap.add_argument("--turns-free", action="store_true",
                    help="treat winding turns as a free per-design analytic transformer "
                         "(feasible iff one Nc satisfies all points' I and V limits)")
    ap.add_argument("--nc-base", type=float, default=113.0, help="turns the FEA/flux is built at")
    ap.add_argument("--lew", type=float, default=None,
                    help="end-winding leakage [H] for the voltage bound; default h0h1_par.LEW_H")
    ap.add_argument("--fhz", type=float, default=50.0)
    ap.add_argument("--n-seed", type=int, default=64)
    ap.add_argument("--n-rounds", type=int, default=8)
    ap.add_argument("--q", type=int, default=8, help="geometries FEA-evaluated per round")
    ap.add_argument("--n-paths", type=int, default=256, help="coherent Thompson sample paths / round")
    ap.add_argument("--n-gcand", type=int, default=256)
    ap.add_argument("--n-icand", type=int, default=512)
    ap.add_argument("--n-workers", type=int, default=8)
    ap.add_argument("--ncores", type=int, default=1)
    ap.add_argument("--slots", type=int, default=60)
    ap.add_argument("--phases", type=int, default=5)
    ap.add_argument("--wide", action="store_true")
    ap.add_argument("--aedt-version", default="2024.2")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    demands = [float(x) for x in args.demands.split(",")]
    omegas = [2.0 * math.pi * float(f) for f in args.speeds.split(",")]  # elec rad/s per demand
    assert len(omegas) == len(demands), "--speeds must have one entry per --demands"
    meta = dict(slots=args.slots, phases=args.phases, wide=args.wide, fhz=args.fhz,
                ncores=args.ncores, aedt_version=args.aedt_version)
    print(f"[gen3] pathwise-Thompson MOO (torch-isolated select); demands={demands} "
          f"dim={DIM_G + DIM_I} n_paths={args.n_paths} q={args.q}", flush=True)

    gd = P.open_isolated_design("g3seed", 97, args.aedt_version, slots=args.slots, phases=args.phases)
    gen = P.make_generator(gd, args.wide); lb, ub = H.geom_bounds_arrays(gen)

    def feasible_g():
        for _ in range(2000):
            gn = np.asarray(H.rand_feasible_geom_norm(gen, lb, ub), float)
            if H.build_barriers(gen, gn, lb, ub) is not None:
                return gn
        return np.random.rand(DIM_G)

    def sobol_I(n, seed):
        return torch.quasirandom.SobolEngine(DIM_I, scramble=True, seed=seed).draw(n).numpy()

    def dq_of(u):
        return ICUR_LB + np.asarray(u) * (ICUR_UB - ICUR_LB)

    ckpt = f"{args.out}/gen3.npz"
    Xg, Xi, T, R, FL = [], [], [], [], []   # FL = flux linkages [Fd1,Fq1,Fd3,Fq3] per point
    if os.path.exists(ckpt):
        z = np.load(ckpt, allow_pickle=True)
        Xg, Xi, T, R = list(z["Xg"]), list(z["Xi"]), list(z["T"]), list(z["R"])
        FL = list(z["FL"]) if "FL" in z else [[0.0, 0.0, 0.0, 0.0]] * len(T)
        print(f"[gen3] resume {len(T)} FEA points", flush=True)

    def save():
        np.savez(ckpt, Xg=np.array(Xg), Xi=np.array(Xi), T=np.array(T), R=np.array(R),
                 FL=np.array(FL))

    def run_jobs(pairs, tag):
        g_list, g_key, jobs = [], {}, []
        for gn, ui in pairs:
            kk = tuple(np.round(np.asarray(gn, float), 6))
            if kk not in g_key:
                g_key[kk] = len(g_list); g_list.append(np.asarray(gn, float))
            jobs.append((g_key[kk], list(map(float, gn)), list(dq_of(ui))))
        res = gen2.eval_batch(jobs, meta, args.n_workers, tag, args.out)
        n_ok = n_fail = 0
        for gid, rows in res.items():
            gn = g_list[gid]
            for row in rows:
                dq, tm, rp = row[0], row[1], row[2]
                if len(row) > 4 and not row[4]:
                    n_fail += 1; continue
                ui = (np.asarray(dq, float) - ICUR_LB) / (ICUR_UB - ICUR_LB)
                flux = list(map(float, row[5:9])) if len(row) >= 9 else [0.0, 0.0, 0.0, 0.0]
                Xg.append(gn); Xi.append(ui); T.append(tm); R.append(rp); FL.append(flux); n_ok += 1
        save()
        if n_fail:
            print(f"[gen3] {tag}: {n_ok} ok, {n_fail} failed solves dropped", flush=True)

    # ---- seed ----
    st = 0
    while len(T) < args.n_seed and st < 5:
        need = args.n_seed - len(T)
        run_jobs(list(zip([feasible_g() for _ in range(need)], sobol_I(need, args.seed + 1 + st))),
                 f"seed{st}")
        print(f"[gen3] seed pass {st}: {len(T)}/{args.n_seed}", flush=True)
        st += 1

    # fixed inner current candidate pool (ipk/loss precomputed here so the select subprocess
    # needs no peak-current code / PyAEDT)
    Icand_u = sobol_I(args.n_icand, args.seed + 7)
    Icand_dq = np.array([dq_of(u) for u in Icand_u])
    ipk_cand = np.array([gen2.ipk_of(dq) for dq in Icand_dq])
    loss_cand = np.sum(Icand_dq ** 2, axis=1)
    theta = np.asarray(H._THETA, float)

    def run_select(Gcand, n_paths, tag):
        """Write the job, run the torch-isolated selection subprocess, return its json dict."""
        jobf = f"{args.out}/seljob_{tag}.npz"; outf = f"{args.out}/selout_{tag}.json"
        np.savez(jobf, Xg=np.array(Xg), Xi=np.array(Xi), T=np.array(T), R=np.array(R),
                 FL=np.array(FL), Gcand=np.array(Gcand), Icand_u=Icand_u, Icand_dq=Icand_dq,
                 ipk_cand=ipk_cand, loss_cand=loss_cand, icur_lb=ICUR_LB, icur_ub=ICUR_UB,
                 demands=np.array(demands), omegas=np.array(omegas), theta=theta,
                 i_max=I_MAX, lam=LAM, r_stator=P.R_STATOR,
                 lew=(args.lew if args.lew is not None else P.LEW_H), v_max=args.v_max,
                 turns_free=int(args.turns_free), nc_base=args.nc_base,
                 n_paths=n_paths, q=args.q, seed=args.seed)
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_THREADING_LAYER="SEQUENTIAL",
                   KMP_DUPLICATE_LIB_OK="TRUE")
        subprocess.run([sys.executable, SELECT_PY, jobf, outf], check=True, env=env)
        return json.load(open(outf))

    varlog = []
    last_front = []
    for rnd in range(args.n_rounds):
        Gcand = [feasible_g() for _ in range(args.n_gcand)]
        sel = run_select(Gcand, args.n_paths, f"r{rnd}")
        vl = sel["varlog"]; vl["rnd"] = rnd; varlog.append(vl); last_front = sel["front"]
        pairs = []
        for gi, idxs in sel["picks"]:
            for j in sorted(set(idxs)):
                pairs.append((Gcand[gi], Icand_u[j]))
        run_jobs(pairs, f"r{rnd}")
        json.dump(varlog, open(f"{args.out}/gen3_varlog.json", "w"), indent=2)
        pf = [round(x, 2) for x in vl.get("p_feasible_picks", [])]
        print(f"[gen3] round {rnd}: +{len(pairs)} FEA (total {len(T)})  "
              f"inc_front={vl.get('inc_front_size')} feas_frac={vl.get('mean_feasible_frac'):.2f} "
              f"n_pos_ehvi={vl.get('n_positive_ehvi')} p_feas_picks={pf} "
              f"|front|={len(last_front)}", flush=True)

    # ---- final confirmed front (front-only select over ALL data; no Matheron) ----
    sel = run_select([feasible_g()], 0, "final")   # 1 dummy Gcand (unused when n_paths=0)
    front = sorted(sel["front"])
    out = {"demands": demands, "v_max": args.v_max, "turns_free": bool(args.turns_free),
           "nc_base": args.nc_base, "n_fea": len(T),
           "front_cycle_loss": [f[0] for f in front],
           "front_max_ptp_nm": [f[1] for f in front],
           "front_ripple_pct_at_worst": [f[2] for f in front]}
    if args.turns_free:                              # rows carry the required winding window
        out["front_nc_lo"] = [f[3] for f in front]
        out["front_nc_hi"] = [f[4] for f in front]
    json.dump(out, open(f"{args.out}/gen3_front.json", "w"), indent=2)
    print(f"[gen3] DONE {len(T)} FEA points; confirmed front of {len(front)} points", flush=True)
    tag = "cycle_loss, max_ptp_Nm, ripple%" + (", Nc[lo,hi]" if args.turns_free else "")
    print(f"[gen3] front ({tag}): "
          + "  ".join((f"({f[0]:.2f},{f[1]:.2f},{f[2]:.1f},[{f[3]:.0f}-{f[4]:.0f}])" if args.turns_free
                       else f"({f[0]:.2f},{f[1]:.2f},{f[2]:.1f})") for f in front), flush=True)


if __name__ == "__main__":
    main()
