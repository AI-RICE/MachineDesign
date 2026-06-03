"""Re-evaluate the three Hackl families' Pareto-front designs at the CONVERGED
mesh (~0.5 mm) to settle the OneLambda/SixLambdas/ThreeBrokenLines comparison
that is buried in coarse-mesh (3 mm) artifact. Records T_mean, T_ripple at the
converged mesh alongside the 3 mm library values.

  export ANSYSEM_ROOT242=/data/Ansys/v242/Linux64/
  ~/Public/PFN/venv_newparam/bin/python notebooks/reeval_converged.py --mesh 0.5 --num-cores 4
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from machine_design import load_design  # noqa: E402
from machine_design.generators import (  # noqa: E402
    HacklGenerator_3BrokenLines,
    HacklGenerator_OneLambda,
    HacklGenerator_SixLambdas,
)
from machine_design.geometry import analyze_results  # noqa: E402
from machine_design.lumped.geometry import REFERENCE_MACHINE  # noqa: E402

GENS = {"OneLambda": HacklGenerator_OneLambda, "SixLambdas": HacklGenerator_SixLambdas,
        "ThreeBrokenLines": HacklGenerator_3BrokenLines}


def fea(design, gen, barriers, cores, mesh):
    bars = gen.split_barriers(barriers)
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
    ap.add_argument("--mesh", type=float, default=0.5)
    ap.add_argument("--designs-npz", default="notebooks/pareto_designs.npz")
    ap.add_argument("--out", default="results_radialspline_live/reeval_converged.csv")
    args = ap.parse_args()

    dd = np.load(args.designs_npz, allow_pickle=True)
    shorts, Xs, T3, R3 = dd["shorts"], dd["Xs"], dd["T3"], dd["R3"]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    done = set()
    if os.path.exists(args.out):
        for ln in open(args.out).read().splitlines()[1:]:
            done.add(int(ln.split(",")[0]))
    else:
        open(args.out, "w").write("idx,family,T3mm,R3mm,Tconv,Rconv,mesh\n")

    design = load_design(args.aedt_project, "SynRM_test", "Design01", args.aedt_version)
    print(f"re-eval {len(shorts)} Pareto designs at mesh={args.mesh}mm", flush=True)
    for i, (s, X, t3, r3) in enumerate(zip(shorts, Xs, T3, R3)):
        if i in done:
            continue
        gen = GENS[s](REFERENCE_MACHINE, r_stator_end=0.7, offset=0.35)
        gen.set_parameters(gen.X_to_params(np.asarray(X, float)))
        Tc, Rc = fea(design, gen, gen.generate_barriers(), args.num_cores, args.mesh)
        with open(args.out, "a") as f:
            f.write(f"{i},{s},{t3:.4f},{r3:.4f},{Tc:.4f},{Rc:.4f},{args.mesh}\n")
        print(f"  [{i:2d}] {s:16s} 3mm:T{t3:.3f}/r{r3:4.1f}  conv:T{Tc:.3f}/r{Rc:4.1f}", flush=True)
    design.close_project()
    print("done", flush=True)


if __name__ == "__main__":
    main()
