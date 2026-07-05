"""Coarse-only 1-D smoothness scan of ripple(g). At a feasible (g0,i0), vary each of a
few geometry params over a small range in fine steps (fixed current, coarse FEA) and
report ripple vs param + the max jump between adjacent steps. Decides: is ripple(g)
rough-but-CONTINUOUS (small adjacent jumps -> fixable by kernel/lengthscale/data) or
DISCONTINUOUS (big abrupt jumps -> unmodelable, use FEA-gate). Coarse-only is valid
because paircheck showed coarse ripple ~= fine ripple.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

import h0h1_par as P
import h0h1_study as H

ICUR_LB = np.array([0.0, 0.0, -3.0, -3.0])
ICUR_UB = np.array([10.0, 10.0, 3.0, 3.0])
OUTDIR = "results/scancheck"
DIMS = [0, 3, 6, 9]     # geometry dims to scan
NSTEP = 16
HALF = 0.12             # scan +/- this (normalized) around g0[d]


def dq_of(u):
    return ICUR_LB + np.asarray(u, float) * (ICUR_UB - ICUR_LB)


def worker_main(args):
    d = json.load(open(args.shard)); jobs = d["jobs"]; meta = d["meta"]
    design = P.open_isolated_design(args.tag, args.worker_id, args.aedt_version, slots=60, phases=5)
    gen = P.make_generator(design, False); lb, ub = H.geom_bounds_arrays(gen)
    P.set_speed(design, 50.0)
    out = {}
    for jid, gn, dq in jobs:
        if jid % args.n_workers != args.worker_id:
            continue
        barriers = H.build_barriers(gen, np.asarray(gn, float), lb, ub)
        if barriers is None:
            out[jid] = None; continue
        design.add_rotor()
        for b in barriers:
            design.add_rotor_barrier(b)
        try:
            r = design.compute(*[float(x) for x in dq], NUM_CORES=meta["ncores"])  # coarse default 3mm/101
            if r is None:
                raise ValueError("None")
            tm, _, rp = H.analyze_results(np.asarray(r["Tor"], float))
            out[jid] = [float(tm), float(rp)]
        except Exception as e:  # noqa: BLE001
            print(f"  [w{args.worker_id}] j{jid} FAIL: {e}", flush=True); out[jid] = None
        design.delete_rotor()
    json.dump(out, open(f"{args.resdir}/res_w{args.worker_id}.json", "w"))
    design.close_project()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--n-workers", type=int, default=20)
    ap.add_argument("--shard"); ap.add_argument("--resdir"); ap.add_argument("--tag", default="scan")
    ap.add_argument("--aedt-version", default="2024.2")
    ap.add_argument("--ncores", type=int, default=1)
    args = ap.parse_args()
    if args.worker:
        return worker_main(args)

    os.makedirs(OUTDIR, exist_ok=True)
    z = np.load("results/gen2_v2/gen2.npz", allow_pickle=True)
    Xg = np.array(z["Xg"]); Xi = np.array(z["Xi"]); T = np.array(z["T"]); R = np.array(z["R"])
    ipk = np.array([P.peak_current_from_dq(*dq_of(Xi[i])) for i in range(len(T))])
    feas = np.where((T >= 20) & (R <= 5) & (ipk <= 10))[0]
    b = feas[np.argmin(R[feas])]
    g0 = Xg[b].copy(); dq0 = dq_of(Xi[b])
    print(f"[scan] base geom feasible pt: T={T[b]:.1f} rip={R[b]:.2f}% dq={[round(x,2) for x in dq0]}", flush=True)

    jobs, layout = [], []
    jid = 0
    for d in DIMS:
        vals = np.linspace(max(0.0, g0[d] - HALF), min(1.0, g0[d] + HALF), NSTEP)
        for s, v in enumerate(vals):
            gn = g0.copy(); gn[d] = v
            jobs.append([jid, list(map(float, gn)), list(map(float, dq0))])
            layout.append((d, s, float(v))); jid += 1
    shard = f"{OUTDIR}/shard.json"
    json.dump({"jobs": jobs, "meta": {"ncores": args.ncores}}, open(shard, "w"))
    resdir = f"{OUTDIR}/res"; os.makedirs(resdir, exist_ok=True)
    for f in os.listdir(resdir):
        os.remove(os.path.join(resdir, f))
    k = min(args.n_workers, len(jobs))
    procs = [subprocess.Popen([sys.executable, os.path.abspath(__file__), "--worker",
             "--worker-id", str(w), "--n-workers", str(k), "--shard", shard, "--resdir", resdir,
             "--tag", "scan", "--aedt-version", args.aedt_version]) for w in range(k)]
    for p in procs:
        p.wait()

    res = {}
    for w in range(k):
        p = f"{resdir}/res_w{w}.json"
        if os.path.exists(p):
            for j_s, v in json.load(open(p)).items():
                res[int(j_s)] = v
    print(f"\n[scan] {sum(1 for v in res.values() if v)}/{len(jobs)} coarse solves ok\n", flush=True)
    for d in DIMS:
        seq = [(s, v, res.get(jid)) for jid, (dd, s, v) in enumerate(layout) if dd == d]
        rips = [r[1] if r else None for _, _, r in seq]
        vals = [v for _, v, _ in seq]
        print(f"=== dim {d}: ripple%% across g[{d}]={vals[0]:.2f}..{vals[-1]:.2f} ===", flush=True)
        print("  " + "  ".join("%.2f" % x if x is not None else "NA" for x in rips), flush=True)
        good = [x for x in rips if x is not None]
        jumps = [abs(rips[i + 1] - rips[i]) for i in range(len(rips) - 1) if rips[i] is not None and rips[i + 1] is not None]
        if jumps:
            print("  range %.2f  max adjacent jump %.2f  (big jump => discontinuous)"
                  % (max(good) - min(good), max(jumps)), flush=True)
    json.dump({"layout": layout, "res": {str(k2): v for k2, v in res.items()}}, open(f"{OUTDIR}/scan.json", "w"))
    print("\n[scan] DONE", flush=True)


if __name__ == "__main__":
    main()
