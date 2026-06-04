"""Build a CONVERGED-labelled Bézier warm-start pool from the Step-1 bar.

The 3 mm-labelled pooled warm-start (Bezier_reencoded_pooled.npz) selects designs
by unreliable 3 mm torques, so the BO warm-start missed the true converged front
(e.g. the 4.410@2.80% low-ripple design looked like 4.26% at 3 mm). This rebuilds
the bar's 200 designs (same build_bar_list selection), re-encodes each to Bézier,
and labels them with the CONVERGED-mesh torques from the bar run — so
select_warmstart picks the actual converged front.

  .venv/bin/python notebooks/bar_to_warmstart.py \
      --results-root ../MachineDesign/results --bar-csv docs/tables/bar_converged.csv

Output: notebooks/Bezier_warmstart_converged.npz  (same schema as the pooled npz:
  X_bz, T_mean, T_ripple, gen_id, gen_names, M, D)
"""

import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from machine_design.bezier_generator import BezierSupersetGenerator  # noqa: E402
from machine_design.lumped.geometry import REFERENCE_MACHINE  # noqa: E402
from notebooks.run_bar_converged import GENS, SHORTS, build_bar_list  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="../MachineDesign/results")
    ap.add_argument("--bar-csv", default="docs/tables/bar_converged.csv")
    ap.add_argument("--M", type=int, default=6)
    ap.add_argument("--max-designs", type=int, default=200)
    args = ap.parse_args()

    # rebuild the SAME design list/order the bar run used (idx = position)
    Xs, shorts, Tm3, Tr3, _ = build_bar_list(args.results_root, args.max_designs)
    # converged torques + status, keyed by idx
    conv = {}
    for r in csv.DictReader(open(args.bar_csv)):
        conv[int(r["idx"])] = (float(r["T_mean_conv"]), float(r["T_ripple_conv"]), r["status"])

    bg = BezierSupersetGenerator(REFERENCE_MACHINE, M=args.M)
    D = bg.block * bg.N
    Xb, Tm, Tr, gid = [], [], [], []
    nskip = 0
    for i, (x, short) in enumerate(zip(Xs, shorts)):
        if i not in conv or conv[i][2] != "ok":
            nskip += 1
            continue
        tm, tr, _ = conv[i]
        hk = GENS[short](REFERENCE_MACHINE, r_stator_end=0.7, offset=0.35)
        hk.set_parameters(hk.X_to_params(np.asarray(x, float)))
        try:
            X = bg.fit_barriers(hk.generate_barriers())
            bg.set_parameters(X)
            if not bg.feasible_barriers(bg.generate_barriers()):
                nskip += 1
                continue
        except Exception:
            nskip += 1
            continue
        Xb.append(X); Tm.append(tm); Tr.append(tr); gid.append(SHORTS.index(short))

    out = os.path.join(HERE, "Bezier_warmstart_converged.npz")
    np.savez(out, X_bz=np.array(Xb), T_mean=np.array(Tm), T_ripple=np.array(Tr),
             gen_id=np.array(gid), gen_names=np.array(SHORTS), M=args.M, D=D)
    Tm, Tr = np.array(Tm), np.array(Tr)
    print(f"converged warm-start: {len(Xb)} designs (skipped {nskip}) -> {out}")
    print(f"  T_mean {Tm.min():.3f}-{Tm.max():.3f}, ripple {Tr.min():.2f}-{Tr.max():.2f}%  "
          f"per-family {[(s, int((np.array(gid)==i).sum())) for i,s in enumerate(SHORTS)]}")


if __name__ == "__main__":
    main()
