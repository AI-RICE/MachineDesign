"""Paired coarse-vs-fine ripple check (gates multi-fidelity for ripple).

Re-evaluates a spread of pooled (g,i) at CONVERGED FEA (0.5mm rotor+airgap mesh,
PointPer=401) and compares torque & ripple to the stored coarse values (3mm/PointPer=101).
Decides: is coarse ripple a CORRELATED proxy for fine ripple (=> MF viable, coarse
carries signal) or NOISE (=> MF cannot rescue ripple, need fine data directly)?
Also confirms torque is fidelity-robust (expected). Worker pool, one fine solve per (g,i).
"""
import argparse
import json
import os
import subprocess
import sys

import h0h1_par as P
import h0h1_study as H
import numpy as np

ICUR_LB = np.array([0.0, 0.0, -3.0, -3.0])
ICUR_UB = np.array([10.0, 10.0, 3.0, 3.0])
OUTDIR = "results/paircheck"


def dq_of(u):
    return ICUR_LB + np.asarray(u, float) * (ICUR_UB - ICUR_LB)


def worker_main(args):
    d = json.load(open(args.shard)); jobs = d["jobs"]; meta = d["meta"]
    design = P.open_isolated_design(args.tag, args.worker_id, args.aedt_version, slots=60, phases=5)
    gen = P.make_generator(design, False); lb, ub = H.geom_bounds_arrays(gen)
    P.set_speed(design, 50.0)
    design.m2d["PointPer"] = "401"          # fine time resolution
    out = {}
    for gid, gn, dq in jobs:
        if gid % args.n_workers != args.worker_id:
            continue
        barriers = H.build_barriers(gen, np.asarray(gn, float), lb, ub)
        if barriers is None:
            out[gid] = None; continue
        design.add_rotor()
        for b in barriers:
            design.add_rotor_barrier(b)
        try:
            r = design.compute(*[float(x) for x in dq], NUM_CORES=meta["ncores"],
                               mesh_length=0.5, airgap_mesh=0.5)   # converged mesh
            if r is None:
                raise ValueError("compute None")
            tm, _, rp = H.analyze_results(np.asarray(r["Tor"], float))
            out[gid] = [float(tm), float(rp)]
            print(f"  [w{args.worker_id}] g{gid} FINE: T={tm:.2f} rip={rp:.2f}%", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  [w{args.worker_id}] g{gid} FINE FAIL: {e}", flush=True); out[gid] = None
        design.delete_rotor()
    json.dump(out, open(f"{args.resdir}/res_w{args.worker_id}.json", "w"))
    design.close_project()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--n-workers", type=int, default=10)
    ap.add_argument("--shard"); ap.add_argument("--resdir"); ap.add_argument("--tag", default="pair")
    ap.add_argument("--aedt-version", default="2024.2")
    ap.add_argument("--npts", type=int, default=10)
    ap.add_argument("--ncores", type=int, default=1)
    args = ap.parse_args()
    if args.worker:
        return worker_main(args)

    os.makedirs(OUTDIR, exist_ok=True)
    z = np.load("results/gen2_v2/gen2.npz", allow_pickle=True)
    Xg = np.array(z["Xg"]); Xi = np.array(z["Xi"]); Tc = np.array(z["T"]); Rc = np.array(z["R"])
    # decision-relevant, meaningful-torque points; stratified across coarse ripple
    ok = np.where(Tc >= 15.0)[0]
    ok = ok[np.argsort(Rc[ok])]
    sel = ok[np.linspace(0, len(ok) - 1, args.npts).round().astype(int)]
    print(f"[pair] selected {len(sel)} pts, coarse ripple range {Rc[sel].min():.1f}-{Rc[sel].max():.1f}%", flush=True)

    jobs = [[int(i), list(map(float, Xg[i])), list(map(float, dq_of(Xi[i])))] for i in sel]
    shard = f"{OUTDIR}/shard.json"
    json.dump({"jobs": jobs, "meta": {"ncores": args.ncores}}, open(shard, "w"))
    resdir = f"{OUTDIR}/res"; os.makedirs(resdir, exist_ok=True)
    for f in os.listdir(resdir):
        os.remove(os.path.join(resdir, f))
    k = min(args.n_workers, len(jobs))
    procs = [subprocess.Popen([sys.executable, os.path.abspath(__file__), "--worker",
             "--worker-id", str(w), "--n-workers", str(k), "--shard", shard, "--resdir", resdir,
             "--tag", "pair", "--aedt-version", args.aedt_version]) for w in range(k)]
    for p in procs:
        p.wait()

    Tf, Rf, Tcs, Rcs = [], [], [], []
    fine = {}
    for w in range(k):
        p = f"{resdir}/res_w{w}.json"
        if os.path.exists(p):
            for gid_s, v in json.load(open(p)).items():
                if v is not None:
                    fine[int(gid_s)] = v
    for i in sel:
        if int(i) in fine:
            tf, rf = fine[int(i)]
            Tf.append(tf); Rf.append(rf); Tcs.append(Tc[i]); Rcs.append(Rc[i])
    Tf, Rf, Tcs, Rcs = map(np.array, (Tf, Rf, Tcs, Rcs))
    print(f"\n[pair] {len(Tf)}/{len(sel)} fine solves succeeded", flush=True)
    if len(Tf) >= 3:
        ptp_c = Rcs * Tcs / 100.0; ptp_f = Rf * Tf / 100.0
        print("coarse vs fine, per point (T_c/T_f Nm, rip_c/rip_f %%):")
        for j in range(len(Tf)):
            print("  T %.1f/%.1f   rip %.2f/%.2f" % (Tcs[j], Tf[j], Rcs[j], Rf[j]), flush=True)
        print("\n=== correlations (coarse proxy for fine?) ===")
        print("  corr(T_coarse, T_fine)       %.3f  (expect high: torque fidelity-robust)" % np.corrcoef(Tcs, Tf)[0, 1])
        print("  corr(ripple%%_c, ripple%%_f)   %.3f  <== the MF-viability number" % np.corrcoef(Rcs, Rf)[0, 1])
        print("  corr(ptp_c, ptp_f)           %.3f" % np.corrcoef(ptp_c, ptp_f)[0, 1])
        print("  ripple%% bias (mean f - c)    %.2f  ; RMS(f-c) %.2f" % (np.mean(Rf - Rcs), np.sqrt(np.mean((Rf - Rcs) ** 2))))
        json.dump({"T_c": Tcs.tolist(), "T_f": Tf.tolist(), "rip_c": Rcs.tolist(), "rip_f": Rf.tolist()},
                  open(f"{OUTDIR}/pair.json", "w"), indent=2)
    print("[pair] DONE", flush=True)


if __name__ == "__main__":
    main()
