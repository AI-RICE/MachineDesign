"""Re-encode all three Hackl parameterisations into RadialSpline coordinates,
keeping BOTH objectives (T_mean, T_ripple), pooled into one file for the
multi-objective unified-warm-start Phase-3 run (E6).

(E2's cache had OneLambda only and T_mean only; this redoes all three with both
objectives.)  Geometry-only re-encode (no FEA): for each FEA design, build the
Hackl barriers and least-squares-fit RadialSpline coords.

  .venv/bin/python notebooks/radialspline_reencode_mo.py --results-root ../MachineDesign/results

Output: notebooks/RadialSpline_reencoded_pooled.npz
  X_rs (N,114), T_mean (N,), T_ripple (N,), gen_id (N,) in {0:OneLambda,
  1:SixLambdas, 2:ThreeBrokenLines}, gen_names.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from machine_design.fea_emulator import load_fea_designs  # noqa: E402
from machine_design.generators import (  # noqa: E402
    HacklGenerator_3BrokenLines,
    HacklGenerator_OneLambda,
    HacklGenerator_SixLambdas,
    RadialSplineGenerator,
)
from machine_design.lumped.geometry import REFERENCE_MACHINE  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
GENS = [("OneLambda", HacklGenerator_OneLambda), ("SixLambdas", HacklGenerator_SixLambdas),
        ("ThreeBrokenLines", HacklGenerator_3BrokenLines)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="../MachineDesign/results")
    args = ap.parse_args()

    rs = RadialSplineGenerator(REFERENCE_MACHINE)
    Xs, Tm, Tr, gid = [], [], [], []
    for g, (short, cls) in enumerate(GENS):
        loaded = load_fea_designs(short, results_root=args.results_root, constrained=None)
        hk = cls(REFERENCE_MACHINE, r_stator_end=0.7, offset=0.35)
        nkeep = 0
        for x, tm, tr in zip(loaded.X, loaded.T_mean, loaded.T_ripple):
            hk.set_parameters(hk.X_to_params(np.asarray(x, float)))
            bars = hk.generate_barriers()
            if not hk.feasible_barriers(bars):
                continue
            Xs.append(rs.fit_barriers(bars))
            Tm.append(float(tm)); Tr.append(float(tr)); gid.append(g); nkeep += 1
        print(f"{short:16s}: {nkeep}/{len(loaded.X)} feasible re-encoded "
              f"(T_mean {min(loaded.T_mean):.2f}-{max(loaded.T_mean):.2f}, "
              f"ripple {min(loaded.T_ripple):.1f}-{max(loaded.T_ripple):.1f}%)", flush=True)

    out = os.path.join(HERE, "RadialSpline_reencoded_pooled.npz")
    np.savez(out, X_rs=np.array(Xs), T_mean=np.array(Tm), T_ripple=np.array(Tr),
             gen_id=np.array(gid), gen_names=np.array([s for s, _ in GENS]))
    print(f"\npooled N={len(Xs)} -> {out}")


if __name__ == "__main__":
    main()
