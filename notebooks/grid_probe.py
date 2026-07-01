"""OAT feasibility/sensitivity grid around the voltage-regime P0.

Evaluates P0 and P0 +- step in each of the 16 params (12 geometry + 4 dq currents)
= 33 FEA. Maps the local feasibility boundary (ripple<=5, V<=800, Ipk<=10) and the
torque sensitivity dT/dparam in every direction -- exactly the local neighborhood
the joint BO skipped (it piled samples at the trust-region edges).

Geometry perturbations hold the currents at c* (so a feasible geom direction that
raises torque is a local H1 signal; one that only breaks feasibility is not).

Driver spawns N workers, each an isolated 1-core AEDT desktop, round-robin shard.
  export ANSYSEM_ROOT242=/data/Ansys/v242/Linux64/
  ./venv_5f/bin/python notebooks/grid_probe.py --n-workers 12 --step 0.08
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import h0h1_par as P  # WideSixLambdas, open_isolated_design, build_barriers, peaks, ICUR box
from machine_design import analyze_results

OUT = "results_grid"


def build_points(geom0, cur0_norm, step):
    span = P.ICUR_UB - P.ICUR_LB
    dq0 = P.ICUR_LB + cur0_norm * span
    pts = [("P0", geom0.copy(), dq0.copy())]
    for i in range(12):
        for s in (step, -step):
            g = geom0.copy(); g[i] = float(np.clip(g[i] + s, 0.0, 1.0))
            pts.append((f"g{i}{'+' if s > 0 else '-'}", g, dq0.copy()))
    for i in range(4):
        for s in (step, -step):
            c = cur0_norm.copy(); c[i] = float(np.clip(c[i] + s, 0.0, 1.0))
            pts.append((f"c{i}{'+' if s > 0 else '-'}", geom0.copy(), P.ICUR_LB + c * span))
    return pts


def worker(args):
    d = np.load(args.shard)
    G, DQ = d["G"], d["DQ"]
    design = P.open_isolated_design("grid", args.worker_id, args.aedt_version)
    if args.fhz > 0:
        P.set_speed(design, args.fhz)
    gen = P.make_generator(design, wide=True)
    lb, ub = P.H.geom_bounds_arrays(gen)
    for idx in range(args.worker_id, len(G), args.n_workers):
        gn, dq = G[idx], DQ[idx]
        ipk = P.peak_current_from_dq(*dq)
        barriers = P.H.build_barriers(gen, gn, lb, ub)
        if barriers is None:
            np.savez(f"{args.resdir}/res_{idx}.npz", t=0.0, r=999.0, v=9999.0, i=ipk)
            continue
        design.add_rotor()
        for b in barriers:
            design.add_rotor_barrier(b)
        try:
            res = design.compute(*[float(x) for x in dq], NUM_CORES=args.num_cores)
            m = res["means"]
            T, _, rip = analyze_results(np.asarray(res["Tor"], float))
            vpk = P.combined_voltage_peak(m["V_d1"], m["V_q1"], m["V_d3"], m["V_q3"])
            if not np.isfinite(T) or not np.isfinite(rip) or not np.isfinite(vpk):
                T, rip, vpk = 0.0, 999.0, 9999.0
        except Exception as e:  # noqa: BLE001
            print(f"  [w{args.worker_id}] FEA exc: {e}", flush=True)
            T, rip, vpk = 0.0, 999.0, 9999.0
        design.delete_rotor()
        np.savez(f"{args.resdir}/res_{idx}.npz", t=T, r=rip, v=vpk, i=ipk)
        print(f"  [w{args.worker_id}] {idx}: T={T:.2f} rip={rip:.2f} Vpk={vpk:.0f} Ipk={ipk:.2f}", flush=True)
    design.close_project()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--worker", action="store_true")
    p.add_argument("--worker-id", type=int, default=0)
    p.add_argument("--n-workers", type=int, default=12)
    p.add_argument("--shard"); p.add_argument("--resdir")
    p.add_argument("--num-cores", type=int, default=1)
    p.add_argument("--aedt-version", default="2024.2")
    p.add_argument("--step", type=float, default=0.08)
    p.add_argument("--fhz", type=float, default=0.0, help="electrical freq [Hz]; 0=default 50, 63.1=voltage regime")
    args = p.parse_args()
    if args.worker:
        worker(args)
        return

    os.makedirs(OUT, exist_ok=True)
    s2 = json.load(open("results_volt/stage2_best.json"))
    geom0 = np.array(s2["geom_norm"])
    cur0_norm = np.clip(P.cur_to_norm(np.array(s2["dq"])), 0.0, 1.0)
    pts = build_points(geom0, cur0_norm, args.step)
    labels = [x[0] for x in pts]
    G = np.array([x[1] for x in pts]); DQ = np.array([x[2] for x in pts])
    shard = f"{OUT}/shard.npz"; np.savez(shard, G=G, DQ=DQ)
    resdir = f"{OUT}/res"; os.makedirs(resdir, exist_ok=True)
    for f in os.listdir(resdir):
        os.remove(os.path.join(resdir, f))
    k = min(args.n_workers, len(G))
    procs = [subprocess.Popen([sys.executable, os.path.abspath(__file__), "--worker",
             "--worker-id", str(w), "--n-workers", str(k), "--shard", shard, "--resdir", resdir,
             "--num-cores", str(args.num_cores), "--aedt-version", args.aedt_version,
             "--fhz", str(args.fhz)]) for w in range(k)]
    for pr in procs:
        pr.wait()

    N = len(G)
    T = np.full(N, np.nan); R = np.full(N, np.nan); V = np.full(N, np.nan); I = np.full(N, np.nan)
    for idx in range(N):
        f = f"{resdir}/res_{idx}.npz"
        if os.path.exists(f):
            d = np.load(f); T[idx], R[idx], V[idx], I[idx] = d["t"], d["r"], d["v"], d["i"]
    feas = (T > 0) & (R <= 5) & (V <= 800) & (I <= 10.001)
    np.savez(f"{OUT}/grid_results.npz", labels=labels, T=T, R=R, V=V, I=I, feas=feas, G=G, DQ=DQ)
    T0 = T[0]
    print(f"=== GRID DONE  P0: T={T0:.2f} rip={R[0]:.2f} Vpk={V[0]:.0f} Ipk={I[0]:.2f} feas={bool(feas[0])} ===", flush=True)
    for idx in range(1, N):
        print(f"  {labels[idx]:>4}: dT={T[idx]-T0:+6.2f} rip={R[idx]:5.2f} Vpk={V[idx]:4.0f} Ipk={I[idx]:5.2f} feas={bool(feas[idx])}", flush=True)
    gimp = [(labels[i], round(float(T[i] - T0), 2)) for i in range(1, N)
            if feas[i] and labels[i].startswith("g") and T[i] > T0]
    print(f"FEASIBLE geometry dirs improving torque: {gimp if gimp else 'NONE'}", flush=True)
    print(f"feasible total: {int(feas.sum())}/{N}", flush=True)


if __name__ == "__main__":
    main()
