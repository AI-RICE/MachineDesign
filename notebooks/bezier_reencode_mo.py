"""Re-encode all three Hackl parameterisations into the unified Bézier-superset
coordinates (D=6*M*N, default M=6 -> 108), keeping BOTH objectives
(T_mean, T_ripple), pooled into one file for the multi-objective unified
warm-start (Step 4 / Step 5).

Geometry-only re-encode (no FEA): for each FEA design, build the Hackl barriers,
encode to the fixed-M Bézier chain, and KEEP IT ONLY IF the Bézier re-encode is
itself feasible (Shapely simple + min-iron/rib/bridge/shaft). The ~1% of designs
whose re-encode self-intersects (extreme thin 3BL) are skipped — logged per the
no-silent-truncation rule.

Labels are the original Hackl FEA torques; the M=6 decode under-torques ~0.9%
(Step 4b) so these are within ~1% of the true Bézier-decode value. Final-front
designs get a high-fidelity reeval before any dominate claim (see docs §Step 4b).

  .venv/bin/python notebooks/bezier_reencode_mo.py --results-root ../MachineDesign/results

Output: notebooks/Bezier_reencoded_pooled.npz
  X_bz (N,D), T_mean (N,), T_ripple (N,), gen_id (N,) in
  {0:OneLambda, 1:SixLambdas, 2:ThreeBrokenLines}, gen_names, M, D.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from machine_design.bezier_generator import BezierSupersetGenerator  # noqa: E402
from machine_design.fea_emulator import load_fea_designs  # noqa: E402
from machine_design.generators import (  # noqa: E402
    HacklGenerator_3BrokenLines,
    HacklGenerator_OneLambda,
    HacklGenerator_SixLambdas,
)
from machine_design.lumped.geometry import REFERENCE_MACHINE  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
GENS = [("OneLambda", HacklGenerator_OneLambda), ("SixLambdas", HacklGenerator_SixLambdas),
        ("ThreeBrokenLines", HacklGenerator_3BrokenLines)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="../MachineDesign/results")
    ap.add_argument("--M", type=int, default=6)
    args = ap.parse_args()

    bg = BezierSupersetGenerator(REFERENCE_MACHINE, M=args.M)
    D = bg.block * bg.N
    Xs, Tm, Tr, gid = [], [], [], []
    skipped = {}
    for g, (short, cls) in enumerate(GENS):
        loaded = load_fea_designs(short, results_root=args.results_root, constrained=None)
        hk = cls(REFERENCE_MACHINE, r_stator_end=0.7, offset=0.35)
        nkeep, nskip = 0, 0
        for x, tm, tr in zip(loaded.X, loaded.T_mean, loaded.T_ripple):
            hk.set_parameters(hk.X_to_params(np.asarray(x, float)))
            bars = hk.generate_barriers()
            try:
                X = bg.fit_barriers(bars)
                bg.set_parameters(X)
                if not bg.feasible_barriers(bg.generate_barriers()):
                    nskip += 1
                    continue
            except Exception:
                nskip += 1
                continue
            Xs.append(X)
            Tm.append(float(tm)); Tr.append(float(tr)); gid.append(g); nkeep += 1
        skipped[short] = nskip
        print(f"{short:16s}: {nkeep}/{len(loaded.X)} feasible re-encoded, {nskip} skipped "
              f"(T_mean {min(loaded.T_mean):.2f}-{max(loaded.T_mean):.2f}, "
              f"ripple {min(loaded.T_ripple):.1f}-{max(loaded.T_ripple):.1f}%)", flush=True)

    out = os.path.join(HERE, "Bezier_reencoded_pooled.npz")
    np.savez(out, X_bz=np.array(Xs), T_mean=np.array(Tm), T_ripple=np.array(Tr),
             gen_id=np.array(gid), gen_names=np.array([s for s, _ in GENS]),
             M=args.M, D=D)
    print(f"\npooled N={len(Xs)} (D={D}) -> {out}  | skipped {skipped}")


if __name__ == "__main__":
    main()
