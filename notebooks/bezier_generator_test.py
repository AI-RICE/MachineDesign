"""Step 4a checks (no FEA): BezierSupersetGenerator containment round-trip at the
D≈100 setting, feasibility of re-encoded Hackl designs, and a perturbation
feasibility rate (informs the constraint-as-filter prior).

  .venv/bin/python notebooks/bezier_generator_test.py
"""

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

GENS = [("OneLambda", HacklGenerator_OneLambda), ("SixLambdas", HacklGenerator_SixLambdas),
        ("ThreeBrokenLines", HacklGenerator_3BrokenLines)]


def maxerr(orig, dec):
    o = orig[:-1] if np.allclose(orig[0], orig[-1]) else orig
    d = np.sqrt(((o[:, None, :] - dec[None, :, :]) ** 2).sum(-1)).min(1)
    return d.max(), np.sqrt((d ** 2).mean())


def good(short, cls, k=6):
    d = load_fea_designs(short, "../MachineDesign/results", None)
    idx = np.argsort(d.T_mean)[-k:]
    hk = cls(REFERENCE_MACHINE, r_stator_end=0.7, offset=0.35)
    out = []
    for j in idx:
        hk.set_parameters(hk.X_to_params(np.asarray(d.X[j], float)))
        out.append(hk.generate_barriers())
    return out


def main():
    for M in (6, 8):
        gen = BezierSupersetGenerator(REFERENCE_MACHINE, M=M)
        D = gen.block * gen.N
        print(f"\n=== M={M}  (D={D})  containment round-trip + feasibility ===")
        print(f"{'family':16s} | {'max err':>8} | {'rms':>7} | {'re-enc feasible':>15}")
        for short, cls in GENS:
            designs = good(short, cls)
            mx, rms, feas = [], [], 0
            for bars in designs:
                X = gen.fit_barriers(bars)
                gen.set_parameters(X)
                dec = gen.generate_barriers()
                feas += gen.feasible_barriers(dec)
                ds = sorted(dec, key=lambda p: np.linalg.norm(p, axis=1).min())
                hs = sorted(bars, key=lambda p: np.linalg.norm(p, axis=1).min())
                for o, d in zip(hs, ds):
                    m, r = maxerr(o, d); mx.append(m); rms.append(r)
            print(f"{short:16s} | {np.mean(mx):>8.3f} | {np.mean(rms):>7.4f} | {feas:>13d}/{len(designs)}")

    # perturbation feasibility rate (informs constraint-as-filter prior)
    gen = BezierSupersetGenerator(REFERENCE_MACHINE, M=6)
    bars = good("SixLambdas", HacklGenerator_SixLambdas, k=1)[0]
    X0 = gen.fit_barriers(bars)
    rng = np.random.default_rng(0)
    print("\n=== feasibility rate of Gaussian perturbations around a warm-start design ===")
    for sigma in (0.0, 0.1, 0.25, 0.5, 1.0):
        ok = 0
        for _ in range(200):
            gen.set_parameters(X0 + rng.normal(0, sigma, X0.shape))
            try:
                ok += gen.feasible_barriers(gen.generate_barriers())
            except Exception:
                pass
        print(f"  sigma={sigma:4.2f} mm : {ok/200*100:5.1f}% feasible")


if __name__ == "__main__":
    main()
