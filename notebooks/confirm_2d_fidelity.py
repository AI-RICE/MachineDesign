"""End-to-end FEA confirmation: does the 2-D parametric (spline) re-encode
preserve torque, where the r(theta) re-encode lost 9-12% (E7/E8)?

For each test Hackl design, evaluate THREE geometries in the same ANSYS session:
  H  = original Hackl geometry,
  2D = periodic 2-D cubic-spline re-encode of the Hackl barriers,
  R  = r(theta) RadialSpline re-encode (K=48),
each with the same central rib (split_barriers, offset=0.35). Compare T_mean,
T_ripple. If 2D ~= H and R < H, the 2-D family closes the fidelity loss.

Run (bayes, isolated venv, v242):
  export ANSYSEM_ROOT242=/data/Ansys/v242/Linux64/
  ~/Public/PFN/venv_newparam/bin/python notebooks/confirm_2d_fidelity.py --num-cores 4
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from scipy.interpolate import splev, splprep

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from machine_design import load_design  # noqa: E402
from machine_design.fea_emulator import load_fea_designs  # noqa: E402
from machine_design.generators import (  # noqa: E402
    HacklGenerator_3BrokenLines,
    HacklGenerator_OneLambda,
    HacklGenerator_SixLambdas,
    RadialSplineGenerator,
)
from machine_design.geometry import analyze_results  # noqa: E402
from machine_design.lumped.geometry import REFERENCE_MACHINE  # noqa: E402

GENS = {"OneLambda": HacklGenerator_OneLambda, "SixLambdas": HacklGenerator_SixLambdas,
        "ThreeBrokenLines": HacklGenerator_3BrokenLines}


def refit_2d(barrier, s=0.2, n=600):
    p = barrier[:-1] if np.allclose(barrier[0], barrier[-1]) else barrier
    x, y = p[:, 0], p[:, 1]
    u = np.r_[0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    u /= u[-1]
    tck, _ = splprep([x, y], u=u, k=3, per=1, s=s)
    xf, yf = splev(np.linspace(0, 1, n), tck)
    poly = np.column_stack([xf, yf])
    return np.vstack([poly, poly[:1]])  # close (check_barrier)


def resample(barrier, n=1500):
    """Near-exact geometry: re-sample the original boundary by arc length (no
    smoothing, corners preserved) -> isolates the rib/discretisation pipeline."""
    p = barrier
    x, y = p[:, 0], p[:, 1]
    u = np.r_[0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    u /= u[-1]
    un = np.linspace(0, 1, n)
    return np.column_stack([np.interp(un, u, x), np.interp(un, u, y)])


def fea(design, splitter, barriers, cores):
    bars = splitter.split_barriers(barriers)
    design.add_rotor()
    for b in bars:
        design.add_rotor_barrier(b)
    try:
        Tor = design.compute(cores)
        Tm, _, Tr = analyze_results(np.asarray(Tor, float))
    except Exception as e:
        print("   FEA exception:", repr(e), flush=True)
        Tm, Tr = float("nan"), float("nan")
    design.delete_rotor()
    return float(Tm), float(Tr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aedt-project", default="data/SynRM_test.aedt")
    ap.add_argument("--aedt-version", default="2024.2")
    ap.add_argument("--num-cores", type=int, default=4)
    ap.add_argument("--designs-npz", default="notebooks/test_designs_2d.npz",
                    help="pre-extracted test Hackl designs (shorts, Xs, Tlib, Rlib, tags)")
    args = ap.parse_args()

    dd = np.load(args.designs_npz, allow_pickle=True)
    tests = list(zip(dd["shorts"], dd["Xs"], dd["Tlib"], dd["Rlib"], dd["tags"]))

    design = load_design(args.aedt_project, "SynRM_test", "Design01", args.aedt_version)
    rg = RadialSplineGenerator(REFERENCE_MACHINE, K=48)
    hdr = f"{'design':22s} | {'H (FEA)':>13} | {'resample~exact':>13} | {'2D s=0.2':>13} | {'r(theta)':>13}"
    print(hdr, flush=True)
    for short, X, Tlib, Rlib, tag in tests:
        hk = GENS[short](REFERENCE_MACHINE, r_stator_end=0.7, offset=0.35)
        hk.set_parameters(hk.X_to_params(np.asarray(X, float)))
        hb = hk.generate_barriers()
        Th, Rh = fea(design, hk, hb, args.num_cores)
        Te, Re = fea(design, hk, [resample(b) for b in hb], args.num_cores)        # near-exact geom
        T2, R2 = fea(design, hk, [refit_2d(b, 0.2, 600) for b in hb], args.num_cores)  # smoothed
        rg.set_parameters(rg.fit_barriers(hb))
        T_r, R_r = fea(design, rg, rg.generate_barriers(), args.num_cores)
        print(f"{short+'/'+tag:22s} | T{Th:.3f} r{Rh:4.1f} | T{Te:.3f} r{Re:4.1f} | "
              f"T{T2:.3f} r{R2:4.1f} | T{T_r:.3f} r{R_r:4.1f}", flush=True)
    design.close_project()


if __name__ == "__main__":
    main()
