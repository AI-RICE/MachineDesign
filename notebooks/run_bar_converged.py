"""Step 1 — SET THE BAR: re-evaluate the three Hackl families' union Pareto front
at the CONVERGED mesh (rotor 0.5 + airgap 0.5), on the ORIGINAL Hackl geometry
(the honest "existing method" front the unified Bézier BO must beat).

The library torques are at the old 3 mm mesh; the mesh reorders the front, so we
take the 3 mm non-dominated set PLUS a near-front buffer, re-solve each at the
converged mesh, and recompute the front from the converged torques.

Parallelism (see memory: 1-core solve = 0 HPC, ~25 seats): DRIVER spawns
--n-workers worker processes; each owns ONE AEDT desktop + project and solves a
round-robin shard at 1 core. Resumable (per-design npz). Collates at the end.

  # driver (bayes):
  export ANSYSEM_ROOT242=/data/Ansys/v242/Linux64/
  ~/Public/PFN/venv_newparam/bin/python notebooks/run_bar_converged.py \
      --results-root ../MachineDesign/results --n-workers 8 --max-designs 200
  # dry run (no ANSYS): add --mock
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from machine_design.fea_emulator import load_fea_designs  # noqa: E402
from machine_design.generators import (  # noqa: E402
    HacklGenerator_3BrokenLines,
    HacklGenerator_OneLambda,
    HacklGenerator_SixLambdas,
)
from machine_design.geometry import analyze_results  # noqa: E402
from machine_design.lumped.geometry import REFERENCE_MACHINE  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
GENS = {"OneLambda": HacklGenerator_OneLambda, "SixLambdas": HacklGenerator_SixLambdas,
        "ThreeBrokenLines": HacklGenerator_3BrokenLines}
SHORTS = ["OneLambda", "SixLambdas", "ThreeBrokenLines"]
ROTOR, AIRGAP = 0.5, 0.5
RUN_DIR = Path("results_bar")


def pareto_mask(Tm, Tr):
    """Non-dominated for (T_mean ↑, T_ripple ↓)."""
    f = np.column_stack([Tm, -Tr])  # both maximised
    nd = np.ones(len(f), bool)
    for i in range(len(f)):
        if nd[i]:
            # drop points strictly DOMINATED by i (i is >= in all, > in some)
            dominated = np.all(f <= f[i], axis=1) & np.any(f < f[i], axis=1)
            nd[dominated] = False
    return nd


def build_bar_list(results_root, max_designs, buffer_frac=0.5):
    """Pool the 3 families; pick the 3 mm non-dominated front + a near-front
    buffer (within buffer_frac of the front in normalised objective space)."""
    Xs, shorts, Tm, Tr = [], [], [], []
    for short in SHORTS:
        d = load_fea_designs(short, results_root=results_root, constrained=None)
        for x, tm, tr in zip(d.X, d.T_mean, d.T_ripple):
            Xs.append(np.asarray(x, float)); shorts.append(short)
            Tm.append(float(tm)); Tr.append(float(tr))
    Tm, Tr = np.array(Tm), np.array(Tr)
    nd = pareto_mask(Tm, Tr)
    # near-front buffer: distance to front in normalised (T_mean, T_ripple) space
    fm = (Tm - Tm.min()) / (np.ptp(Tm) + 1e-9)
    fr = (Tr - Tr.min()) / (np.ptp(Tr) + 1e-9)
    P = np.column_stack([fm[nd], fr[nd]])
    keep = nd.copy()
    if buffer_frac > 0:
        Q = np.column_stack([fm, fr])
        dmin = np.array([np.min(np.linalg.norm(P - q, axis=1)) for q in Q])
        keep |= dmin < buffer_frac * 0.1  # within 10%*buffer_frac of the front
    idx = np.where(keep)[0]
    # cap: keep the front, subsample the buffer evenly along T_mean
    if len(idx) > max_designs:
        front_idx = np.where(nd)[0]
        buf_idx = np.setdiff1d(idx, front_idx)
        n_buf = max(0, max_designs - len(front_idx))
        if n_buf and len(buf_idx) > n_buf:
            order = np.argsort(Tm[buf_idx])
            buf_idx = buf_idx[order[np.linspace(0, len(buf_idx) - 1, n_buf).astype(int)]]
        idx = np.sort(np.concatenate([front_idx, buf_idx]))[:max_designs]
    return ([Xs[i] for i in idx], [shorts[i] for i in idx],
            Tm[idx], Tr[idx], int(nd.sum()))


def solve_one(design, short, X, cores, mock=False):
    gen = GENS[short](REFERENCE_MACHINE, r_stator_end=0.7, offset=0.35)
    gen.set_parameters(gen.X_to_params(np.asarray(X, float)))
    bars = gen.split_barriers(gen.generate_barriers())
    if mock:
        return float("nan"), float("nan"), "mock"
    design.add_rotor()
    for b in bars:
        design.add_rotor_barrier(b)
    try:
        Tor = design.compute(cores, mesh_length=ROTOR, airgap_mesh=AIRGAP)
        Tm, _, Tr = analyze_results(np.asarray(Tor, float))
        status = "ok"
    except Exception as e:
        Tm, Tr, status = float("nan"), float("nan"), f"exception:{type(e).__name__}"
    finally:
        try:
            design.delete_rotor()
        except Exception:
            pass
    return float(Tm), float(Tr), status


def worker(args):
    d = np.load(RUN_DIR / "bar_list.npz", allow_pickle=True)
    Xs, shorts = d["Xs"], d["shorts"]
    Tm3, Tr3 = d["Tm3"], d["Tr3"]
    my = list(range(args.worker_id, len(Xs), args.n_workers))
    todo = [i for i in my if not (RUN_DIR / f"d{i:04d}.npz").exists()]
    print(f"[w{args.worker_id}] {len(todo)}/{len(my)} designs to solve", flush=True)
    if not todo:
        return
    design = None
    if not args.mock:
        from machine_design import load_design
        proj = f"data/bar_w{args.worker_id}.aedt"
        design = load_design(proj, f"bar_w{args.worker_id}", "Design01",
                             args.aedt_version, new_desktop=True)
    for i in todo:
        t0 = time.time()
        Tm, Tr, status = solve_one(design, str(shorts[i]), Xs[i], args.cores, args.mock)
        np.savez(RUN_DIR / f"d{i:04d}.npz", idx=i, short=str(shorts[i]), X=Xs[i],
                 T_mean_3mm=float(Tm3[i]), T_ripple_3mm=float(Tr3[i]),
                 T_mean=Tm, T_ripple=Tr, status=status)
        print(f"[w{args.worker_id}] d{i:04d} {shorts[i]:16s} 3mm({Tm3[i]:.3f},{Tr3[i]:.2f}) "
              f"-> conv({Tm:.3f},{Tr:.2f}) {status} {time.time()-t0:.0f}s", flush=True)
    if design is not None:
        design.close_project()
    print(f"[w{args.worker_id}] done", flush=True)


def driver(args):
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    if not (RUN_DIR / "bar_list.npz").exists():
        Xs, shorts, Tm3, Tr3, nfront = build_bar_list(args.results_root, args.max_designs)
        np.savez(RUN_DIR / "bar_list.npz", Xs=np.array(Xs, dtype=object),
                 shorts=np.array(shorts), Tm3=Tm3, Tr3=Tr3)
        print(f"bar list: {len(Xs)} designs ({nfront} on 3mm front + buffer); "
              f"per-family {[ (s, int((np.array(shorts)==s).sum())) for s in SHORTS ]}", flush=True)
    else:
        d = np.load(RUN_DIR / "bar_list.npz", allow_pickle=True)
        print(f"resuming with existing bar list: {len(d['Xs'])} designs", flush=True)

    py = sys.executable
    procs = []
    for k in range(args.n_workers):
        cmd = [py, os.path.abspath(__file__), "--worker-id", str(k),
               "--n-workers", str(args.n_workers), "--cores", str(args.cores),
               "--aedt-version", args.aedt_version]
        if args.mock:
            cmd.append("--mock")
        lf = open(RUN_DIR / f"worker_{k}.log", "a")
        procs.append(subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT))
        time.sleep(2)  # stagger AEDT startup
    print(f"launched {args.n_workers} workers", flush=True)
    for p in procs:
        p.wait()

    # collate
    rows = sorted(RUN_DIR.glob("d*.npz"))
    Tm, Tr, sh, ok = [], [], [], 0
    for fp in rows:
        e = np.load(fp, allow_pickle=True)
        if str(e["status"]) == "ok":
            Tm.append(float(e["T_mean"])); Tr.append(float(e["T_ripple"])); sh.append(str(e["short"])); ok += 1
    Tm, Tr = np.array(Tm), np.array(Tr)
    csv = RUN_DIR / "converged_bar.csv"
    with open(csv, "w") as f:
        f.write("idx,short,T_mean_3mm,T_ripple_3mm,T_mean_conv,T_ripple_conv,on_conv_front,status\n")
        nd = pareto_mask(Tm, Tr) if len(Tm) else np.array([], bool)
        j = 0
        for fp in rows:
            e = np.load(fp, allow_pickle=True)
            isok = str(e["status"]) == "ok"
            front = bool(nd[j]) if isok else False
            f.write(f"{int(e['idx'])},{e['short']},{float(e['T_mean_3mm']):.4f},"
                    f"{float(e['T_ripple_3mm']):.4f},{float(e['T_mean']):.4f},"
                    f"{float(e['T_ripple']):.4f},{int(front)},{e['status']}\n")
            if isok:
                j += 1
    nfront = int(pareto_mask(Tm, Tr).sum()) if len(Tm) else 0
    print(f"=== bar done: {ok}/{len(rows)} solved, {nfront} on converged front -> {csv} ===", flush=True)
    if len(Tm):
        m = pareto_mask(Tm, Tr)
        print(f"converged front: T_mean {Tm[m].min():.3f}-{Tm[m].max():.3f}, "
              f"ripple {Tr[m].min():.2f}-{Tr[m].max():.2f}%", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="../MachineDesign/results")
    ap.add_argument("--n-workers", type=int, default=8)
    ap.add_argument("--cores", type=int, default=1)
    ap.add_argument("--max-designs", type=int, default=200)
    ap.add_argument("--aedt-version", default="2024.2")
    ap.add_argument("--worker-id", type=int, default=-1)
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()
    if args.worker_id >= 0:
        worker(args)
    else:
        driver(args)


if __name__ == "__main__":
    main()
