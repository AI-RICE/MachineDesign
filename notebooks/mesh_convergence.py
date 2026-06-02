"""Mesh-convergence control: is the ~0.13 N·m 're-encoding loss' a meshing
artifact? Run the SAME geometry (Hackl native vs arc-length resample of the same
curve) across rotor mesh sizes. If native and resample CONVERGE as the mesh
refines, the loss is mesh sensitivity (fix = finer mesh). If they stay split,
it's the rib-by-index / boundary discretisation.

  export ANSYSEM_ROOT242=/data/Ansys/v242/Linux64/
  ~/Public/PFN/venv_newparam/bin/python notebooks/mesh_convergence.py --num-cores 4
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


def fea(design, splitter, barriers, cores, mesh):
    bars = splitter.split_barriers(barriers)
    design.add_rotor()
    for b in bars:
        design.add_rotor_barrier(b)
    try:
        Tor = design.compute(cores, mesh_length=mesh)
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
    ap.add_argument("--designs-npz", default="notebooks/test_designs_2d.npz")
    ap.add_argument("--meshes", nargs="+", type=float, default=[3.0, 1.5, 0.75])
    args = ap.parse_args()

    dd = np.load(args.designs_npz, allow_pickle=True)
    tests = [(s, X) for s, X, t in zip(dd["shorts"], dd["Xs"], dd["tags"]) if t == "maxT" and s in GENS]

    design = load_design(args.aedt_project, "SynRM_test", "Design01", args.aedt_version)
    print(f"{'design':14s} | {'mesh':>5} | {'Hackl native':>13} | {'resample~exact':>14} | {'|diff|':>7}", flush=True)
    for short, X in tests:
        hk = GENS[short](REFERENCE_MACHINE, r_stator_end=0.7, offset=0.35)
        hk.set_parameters(hk.X_to_params(np.asarray(X, float)))
        hb = hk.generate_barriers()
        rs = [resample(b) for b in hb]
        for mesh in args.meshes:
            Tn, _ = fea(design, hk, hb, args.num_cores, mesh)
            Tr, _ = fea(design, hk, rs, args.num_cores, mesh)
            print(f"{short:14s} | {mesh:>5.2f} | T={Tn:.4f}   | T={Tr:.4f}    | {abs(Tn-Tr):>7.4f}", flush=True)
    design.close_project()


if __name__ == "__main__":
    main()
