"""High-fidelity re-eval of a Step-5 run's Pareto front: strip the ~0.9% n_per=160
decode bias so the BO front is comparable to the Hackl bar (which is on original
geometry). Re-decodes each front design at a high n_per and re-solves at the
converged mesh, then prints the bias-corrected front vs the bar.

  ~/Public/PFN/venv_newparam/bin/python notebooks/reeval_front.py \
      --run-name pilot2 --warmstart-npz notebooks/Bezier_warmstart_converged.npz \
      --n-per 320 --num-cores 4
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from run_bezier_live_mo import evaluate_fea_converged  # noqa: E402

from machine_design.bezier_bo import warmstart_box  # noqa: E402
from machine_design.bezier_generator import BezierSupersetGenerator  # noqa: E402
from machine_design.lumped.geometry import REFERENCE_MACHINE  # noqa: E402

MESH, AIRGAP = 0.5, 0.5


def pareto_mask(Tm, Tr):
    f = np.column_stack([Tm, -Tr])
    nd = np.ones(len(f), bool)
    for i in range(len(f)):
        if nd[i]:
            nd[np.all(f <= f[i], axis=1) & np.any(f < f[i], axis=1)] = False
    return nd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="pilot2")
    ap.add_argument("--run-root", default="results_bezier_live")
    ap.add_argument("--warmstart-npz", default=os.path.join(HERE, "Bezier_warmstart_converged.npz"))
    ap.add_argument("--n-per", type=int, default=320)
    ap.add_argument("--num-cores", type=int, default=4)
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run_root) / args.run_name
    import csv as _csv
    phase_by_idx = {int(r["idx"]): r["phase"] for r in _csv.DictReader(open(run_dir / "evals.csv"))}
    evals = sorted(run_dir.glob("eval_*.npz"))
    U, Tm, Tr, phase = [], [], [], []
    for fp in evals:
        e = np.load(fp, allow_pickle=True)
        U.append(e["X_norm"].astype(float)); Tm.append(float(e["T_mean"]))
        Tr.append(float(e["T_ripple"])); phase.append(phase_by_idx.get(int(e["idx"]), "?"))
    U, Tm, Tr = np.array(U), np.array(Tm), np.array(Tr)
    nd = pareto_mask(Tm, Tr)
    front = np.where(nd)[0]
    print(f"{args.run_name}: {len(U)} evals, {len(front)} on the (decode) front -> re-eval at n_per={args.n_per}",
          flush=True)

    d = np.load(args.warmstart_npz, allow_pickle=True)
    gen0 = BezierSupersetGenerator(REFERENCE_MACHINE, M=int(d["M"]))
    lo, span = warmstart_box(d["X_bz"], gen0)
    gen_hi = BezierSupersetGenerator(REFERENCE_MACHINE, M=int(d["M"]), n_per=args.n_per)

    design = None
    if not args.mock:
        from machine_design import load_design
        design = load_design("data/reeval.aedt", "reeval", "Design01", "2024.2", new_desktop=True)

    print(f"{'idx':>4} {'phase':>9} | {'decode T/rip':>16} | {'hi-fid T/rip':>16} | dT%", flush=True)
    hi_Tm, hi_Tr = [], []
    for i in front:
        if args.mock:
            tm, tr, st = Tm[i], Tr[i], "mock"
        else:
            rec = evaluate_fea_converged(U[i], gen_hi, design, lo, span, args.num_cores, MESH, AIRGAP)
            tm, tr, st = rec["T_mean"], rec["T_ripple"], rec["status"]
        hi_Tm.append(tm); hi_Tr.append(tr)
        dT = 100 * (tm - Tm[i]) / Tm[i] if Tm[i] else float("nan")
        print(f"{i:>4} {phase[i]:>9} | {Tm[i]:7.3f}/{Tr[i]:6.2f} | {tm:7.3f}/{tr:6.2f} | {dT:+5.2f} [{st}]",
              flush=True)
    if design is not None:
        design.close_project()

    hi_Tm, hi_Tr = np.array(hi_Tm), np.array(hi_Tr)
    ok = ~np.isnan(hi_Tm)
    m2 = pareto_mask(hi_Tm[ok], hi_Tr[ok])
    print(f"\nbias-corrected front ({m2.sum()} pts): "
          f"T_mean {hi_Tm[ok][m2].min():.3f}-{hi_Tm[ok][m2].max():.3f}, "
          f"ripple {hi_Tr[ok][m2].min():.2f}-{hi_Tr[ok][m2].max():.2f}%", flush=True)
    print("BAR (Hackl converged): T_mean 4.410-4.492, ripple 2.80-14.87%", flush=True)


if __name__ == "__main__":
    main()
