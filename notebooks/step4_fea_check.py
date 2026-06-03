"""Step 4b: does the BezierSupersetGenerator re-encode at the BO-vector resolution
(fixed M, D≈100) preserve torque at the converged mesh? This is the warm-start
fidelity that matters for BO (Step 3 was the lossless ~0.04mm encoder).

Per design, FEA at (rotor 0.5, airgap 0.5): H = Hackl, B_M = generator re-encode
at given M. Compare T_mean/ripple.

  ~/Public/PFN/venv_newparam/bin/python notebooks/step4_fea_check.py --M 6 --num-cores 4
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from machine_design import load_design  # noqa: E402
from machine_design.bezier_generator import BezierSupersetGenerator  # noqa: E402
from machine_design.generators import (  # noqa: E402
    HacklGenerator_3BrokenLines,
    HacklGenerator_OneLambda,
    HacklGenerator_SixLambdas,
)
from machine_design.geometry import analyze_results  # noqa: E402
from machine_design.lumped.geometry import REFERENCE_MACHINE  # noqa: E402

GENS = {"OneLambda": HacklGenerator_OneLambda, "SixLambdas": HacklGenerator_SixLambdas,
        "ThreeBrokenLines": HacklGenerator_3BrokenLines}
ROTOR, AIRGAP = 0.5, 0.5


def fea(design, splitter, barriers, cores):
    bars = splitter.split_barriers(barriers)
    design.add_rotor()
    for b in bars:
        design.add_rotor_barrier(b)
    try:
        Tor = design.compute(cores, mesh_length=ROTOR, airgap_mesh=AIRGAP)
        Tm, _, Tr = analyze_results(np.asarray(Tor, float))
    except Exception as e:
        print("  FEA exception:", repr(e), flush=True)
        Tm, Tr = float("nan"), float("nan")
    design.delete_rotor()
    return float(Tm), float(Tr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aedt-project", default="data/SynRM_test.aedt")
    ap.add_argument("--aedt-version", default="2024.2")
    ap.add_argument("--num-cores", type=int, default=4)
    ap.add_argument("--M", type=int, default=6)
    ap.add_argument("--designs-npz", default="notebooks/step3_designs.npz")
    args = ap.parse_args()

    dd = np.load(args.designs_npz, allow_pickle=True)
    tests = list(zip(dd["shorts"], dd["Xs"], dd["tags"]))
    bg = BezierSupersetGenerator(REFERENCE_MACHINE, M=args.M)
    design = load_design(args.aedt_project, "SynRM_test", "Design01", args.aedt_version)
    print(f"M={args.M} (D={bg.block*bg.N}), mesh (rotor {ROTOR}, airgap {AIRGAP})", flush=True)
    print(f"{'design':22s} | {'Hackl':>16} | {'Bézier M':>16} | {'ΔT%':>6} {'Δrip':>6} feas", flush=True)
    worst = 0.0
    for short, X, tag in tests:
        hk = GENS[short](REFERENCE_MACHINE, r_stator_end=0.7, offset=0.35)
        hk.set_parameters(hk.X_to_params(np.asarray(X, float)))
        hb = hk.generate_barriers()
        Th, Rh = fea(design, hk, hb, args.num_cores)
        bg.set_parameters(bg.fit_barriers(hb))
        bb = bg.generate_barriers()
        feas = bg.feasible_barriers(bb)
        Tb, Rb = fea(design, bg, bb, args.num_cores)
        dT = 100 * (Tb - Th) / Th if Th else float("nan")
        worst = max(worst, abs(dT))
        print(f"{short+'/'+tag:22s} | T{Th:.3f} r{Rh:5.2f} | T{Tb:.3f} r{Rb:5.2f} | {dT:+6.2f} {Rb-Rh:+6.2f} {feas}", flush=True)
    print(f"\nworst |ΔT| = {worst:.2f}%  ({'PASS <1%' if worst < 1 else 'PASS<2%' if worst < 2 else 'CHECK'})", flush=True)
    design.close_project()


if __name__ == "__main__":
    main()
