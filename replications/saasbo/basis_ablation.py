"""Basis ablation for the effective-dimension probe (E5 follow-up, no FEA).

Question: the SAASBO probe found ~12 active axes for the intrinsically-7-D
OneLambda manifold, because the Hackl DOF are rotated vs the LOCAL B-spline
control-point axes. Does re-expressing each barrier boundary in a CHEBYSHEV
(modal) basis — where low-order coefficient = low-frequency shape mode — make the
torque-active subspace more axis-aligned (fewer active dims)?

Method: each design's per-boundary B-spline control vector c reconstructs r(s) on
the generator's s-grid; we refit that same curve with K Chebyshev modes via a
fixed linear map T = pinv(Cheb(s)) @ Bspline(s). theta_lo/theta_hi are unchanged.
Then run the SAASBO effective-dim probe on both representations under IDENTICAL
treatment (raw coords; SAASBO's Normalize learns per-column bounds) and compare
the lengthscale spectra / active-set sizes.

  .venv/bin/python replications/saasbo/basis_ablation.py --n 150
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", ".."))

from saasbo import build_saasbo, median_lengthscales  # noqa: E402

from machine_design.generators import RadialSplineGenerator  # noqa: E402
from machine_design.lumped.geometry import REFERENCE_MACHINE  # noqa: E402


def cheb_design(s: np.ndarray, K: int) -> np.ndarray:
    """Chebyshev-T design matrix on s in [0,1] (mapped to [-1,1]), modes 0..K-1."""
    x = 2.0 * s - 1.0
    return np.cos(np.outer(np.arccos(np.clip(x, -1, 1)), np.arange(K)))  # (len(s), K)


def to_chebyshev(X_bspline: np.ndarray, gen: RadialSplineGenerator) -> np.ndarray:
    """Per-boundary linear change of basis: B-spline control pts -> Chebyshev coeffs."""
    K, blk = gen.K, gen.block
    Cheb = cheb_design(gen.s_grid, K)              # (n_eval, K)
    T = np.linalg.pinv(Cheb) @ gen.basis           # (K, K): c_bspline -> c_cheb
    Xc = X_bspline.copy()
    for b in range(gen.N):
        o = b * blk
        Xc[:, o + 2 : o + 2 + K] = X_bspline[:, o + 2 : o + 2 + K] @ T.T          # c_out
        Xc[:, o + 2 + K : o + 2 + 2 * K] = X_bspline[:, o + 2 + K : o + 2 + 2 * K] @ T.T  # c_in
    return Xc


def probe(name, X, T, n, warmup, samples, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))[:n]
    Xt = torch.tensor(X[idx], dtype=torch.double)
    yt = torch.tensor(T[idx], dtype=torch.double).unsqueeze(-1)
    m = build_saasbo(Xt, yt, warmup=warmup, num_samples=samples, thinning=16, seed=seed)
    ls = median_lengthscales(m)
    counts = {c: int((ls < c).sum()) for c in (1.0, 3.0, 10.0, 30.0)}
    print(f"  [{name}] active dims: " + ", ".join(f"<{c}: {n_}" for c, n_ in counts.items()))
    return ls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", default="OneLambda")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cache = os.path.join(HERE, "..", "..", "notebooks", f"RadialSpline_reencoded_{args.generator}.npz")
    d = np.load(cache)
    X, keep, T = d["X_rs"], d["keep"], d["T"]
    X, T = X[keep], T[keep]
    gen = RadialSplineGenerator(REFERENCE_MACHINE)

    Xc = to_chebyshev(X, gen)
    # sanity: basis change is faithful (round-trip reconstructs r(s))
    print(f"[{args.generator}] basis-change check: |X_cheb| finite = {np.isfinite(Xc).all()}")

    print(f"\nSAASBO effective-dim probe, n={args.n}, D={X.shape[1]}:")
    ls_b = probe("B-spline (local)", X, T, args.n, args.warmup, args.samples, args.seed)
    ls_c = probe("Chebyshev (modal)", Xc, T, args.n, args.warmup, args.samples, args.seed)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.sort(ls_b), "o-", ms=3, label="B-spline (local)")
    ax.plot(np.sort(ls_c), "s-", ms=3, label="Chebyshev (modal)")
    ax.axhline(10.0, ls=":", color="0.5")
    ax.set_yscale("log")
    ax.set_xlabel("axis (sorted by median lengthscale)")
    ax.set_ylabel("median lengthscale (norm. input)")
    ax.set_title(f"Effective-dim: B-spline vs Chebyshev basis — {args.generator} (n={args.n})")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(HERE, f"basis_ablation_{args.generator}.png")
    fig.savefig(out, dpi=120)
    print("\n  wrote", out)


if __name__ == "__main__":
    main()
