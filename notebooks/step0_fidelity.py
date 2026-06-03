"""Step 0 (v2): lock FEA fidelity. The E9 study converged the ROTOR mesh (~0.5 mm)
but left the airgap/band mesh at its project default. Here we (a) check airgap-mesh
sensitivity at rotor=0.5 mm, and (b) measure the residual noise floor (native vs
arc-length resample of the same geometry) at the converged setting.

  export ANSYSEM_ROOT242=/data/Ansys/v242/Linux64/
  ~/Public/PFN/venv_newparam/bin/python notebooks/step0_fidelity.py --num-cores 4
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from machine_design import load_design  # noqa: E402
from machine_design.generators import HacklGenerator_OneLambda, HacklGenerator_SixLambdas  # noqa: E402
from machine_design.geometry import analyze_results  # noqa: E402
from machine_design.lumped.geometry import REFERENCE_MACHINE  # noqa: E402

GENS = {"OneLambda": HacklGenerator_OneLambda, "SixLambdas": HacklGenerator_SixLambdas}


def resample(barrier, n=1500):
    x, y = barrier[:, 0], barrier[:, 1]
    u = np.r_[0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    u /= u[-1]
    un = np.linspace(0, 1, n)
    return np.column_stack([np.interp(un, u, x), np.interp(un, u, y)])


def fea(design, gen, barriers, cores, mesh, airgap):
    bars = gen.split_barriers(barriers)
    design.add_rotor()
    for b in bars:
        design.add_rotor_barrier(b)
    try:
        Tor = design.compute(cores, mesh_length=mesh, airgap_mesh=airgap)
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
    ap.add_argument("--rotor-mesh", type=float, default=0.5)
    ap.add_argument("--airgaps", nargs="+", default=["none", "1.0", "0.5", "0.25"])
    ap.add_argument("--designs-npz", default="notebooks/step0_designs.npz")
    args = ap.parse_args()

    dd = np.load(args.designs_npz, allow_pickle=True)
    tests = list(zip(dd["shorts"], dd["Xs"]))
    design = load_design(args.aedt_project, "SynRM_test", "Design01", args.aedt_version)

    print(f"=== A. airgap-mesh sensitivity at rotor={args.rotor_mesh}mm ===", flush=True)
    print(f"{'design':14s} | " + " | ".join(f"ag={a:>5}" for a in args.airgaps), flush=True)
    for short, X in tests:
        gen = GENS[short](REFERENCE_MACHINE, r_stator_end=0.7, offset=0.35)
        gen.set_parameters(gen.X_to_params(np.asarray(X, float)))
        hb = gen.generate_barriers()
        cells = []
        for a in args.airgaps:
            ag = None if a == "none" else float(a)
            T, R = fea(design, gen, hb, args.num_cores, args.rotor_mesh, ag)
            cells.append(f"T{T:.3f}")
        print(f"{short:14s} | " + " | ".join(f"{c:>8}" for c in cells), flush=True)

    print(f"\n=== B. noise floor at rotor={args.rotor_mesh}mm (native vs resample, same shape) ===", flush=True)
    for short, X in tests:
        gen = GENS[short](REFERENCE_MACHINE, r_stator_end=0.7, offset=0.35)
        gen.set_parameters(gen.X_to_params(np.asarray(X, float)))
        hb = gen.generate_barriers()
        Tn, _ = fea(design, gen, hb, args.num_cores, args.rotor_mesh, None)
        Tr, _ = fea(design, gen, [resample(b) for b in hb], args.num_cores, args.rotor_mesh, None)
        print(f"  {short:14s}: native T={Tn:.4f}  resample T={Tr:.4f}  |diff|={abs(Tn-Tr):.4f}", flush=True)
    design.close_project()
    print("done", flush=True)


if __name__ == "__main__":
    main()
