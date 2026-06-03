"""Step 3 (v2): converged-FEA fidelity gate. Does the corner-aware 2-D Bézier
re-encode (~0.04 mm geometry) preserve torque at the converged mesh — the check
the r(θ) family failed (11 mm → ~9-12% loss)?

Per design, FEA at the converged setting (rotor 0.5 mm + airgap 0.5 mm):
  H = original Hackl geometry, B = Bézier-chain re-encode. Compare T_mean/ripple.
PASS if |B - H| < 1% for all.

  export ANSYSEM_ROOT242=/data/Ansys/v242/Linux64/
  ~/Public/PFN/venv_newparam/bin/python notebooks/step3_fidelity_gate.py --num-cores 4
"""

import argparse
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from machine_design import load_design  # noqa: E402
from machine_design.generators import (  # noqa: E402
    HacklGenerator_3BrokenLines,
    HacklGenerator_OneLambda,
    HacklGenerator_SixLambdas,
)
from machine_design.geometry import analyze_results  # noqa: E402
from machine_design.lumped.geometry import REFERENCE_MACHINE  # noqa: E402

_spec = importlib.util.spec_from_file_location("bz", os.path.join(HERE, "bezier_superset_proto.py"))
bz = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bz)
GENS = {"OneLambda": HacklGenerator_OneLambda, "SixLambdas": HacklGenerator_SixLambdas,
        "ThreeBrokenLines": HacklGenerator_3BrokenLines}

ROTOR_MESH, AIRGAP_MESH = 0.5, 0.5  # converged setting (Step 0)


def bezier_reencode(barrier, target_seg_len=4.0, n_per=40):
    chain = bz.fit_chain(barrier, thresh_deg=12.0, target_seg_len=target_seg_len)
    poly = bz.eval_chain(chain, n_per)
    return np.vstack([poly, poly[:1]])  # close for check_barrier


def fea(design, gen, barriers, cores):
    bars = gen.split_barriers(barriers)
    design.add_rotor()
    for b in bars:
        design.add_rotor_barrier(b)
    try:
        Tor = design.compute(cores, mesh_length=ROTOR_MESH, airgap_mesh=AIRGAP_MESH)
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
    ap.add_argument("--designs-npz", default="notebooks/step3_designs.npz")
    args = ap.parse_args()

    dd = np.load(args.designs_npz, allow_pickle=True)
    tests = list(zip(dd["shorts"], dd["Xs"], dd["tags"]))
    design = load_design(args.aedt_project, "SynRM_test", "Design01", args.aedt_version)
    print(f"converged mesh: rotor={ROTOR_MESH} airgap={AIRGAP_MESH}", flush=True)
    print(f"{'design':22s} | {'Hackl (FEA)':>17} | {'Bézier re-enc':>17} | {'ΔT%':>6} {'Δrip%':>6}", flush=True)
    worst = 0.0
    for short, X, tag in tests:
        gen = GENS[short](REFERENCE_MACHINE, r_stator_end=0.7, offset=0.35)
        gen.set_parameters(gen.X_to_params(np.asarray(X, float)))
        hb = gen.generate_barriers()
        Th, Rh = fea(design, gen, hb, args.num_cores)
        Tb, Rb = fea(design, gen, [bezier_reencode(b) for b in hb], args.num_cores)
        dT = 100 * (Tb - Th) / Th if Th else float("nan")
        dR = Rb - Rh
        worst = max(worst, abs(dT))
        print(f"{short+'/'+tag:22s} | T{Th:.3f} r{Rh:5.2f} | T{Tb:.3f} r{Rb:5.2f} | {dT:+6.2f} {dR:+6.2f}", flush=True)
    print(f"\nworst |ΔT| = {worst:.2f}%  -> {'PASS (<1%)' if worst < 1.0 else 'CHECK'}", flush=True)
    design.close_project()


if __name__ == "__main__":
    main()
