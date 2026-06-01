"""Effective-dimension probe (the 'run on simulator' SAASBO outcome).

Fit a SAASBO GP to the re-encoded RadialSpline (X 114-D -> T_mean) data and read
off how many of the 114 axes the torque response actually uses (short median
lengthscale = active). This estimates the EFFECTIVE dimension of the on-manifold
torque response in RadialSpline coordinates, which sets the realistic FEA budget
for Phase-3 high-D BO (budget ~ 10-20 x effective dim, not 20 x 114).

Caveat: trained on the Hackl manifold (the only torque data) — this is the
effective dim of the KNOWN-good region, not of the full off-manifold space.

  .venv/bin/python replications/saasbo/effective_dim_probe.py --n 200
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
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "MachineDesign-newparam"))
sys.path.insert(0, os.path.join(HERE, "..", ".."))

from saasbo import build_saasbo, median_lengthscales  # noqa: E402

from machine_design.generators import RadialSplineGenerator  # noqa: E402
from machine_design.lumped.geometry import REFERENCE_MACHINE  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", default="OneLambda")
    ap.add_argument("--n", type=int, default=200, help="subsample size for the NUTS fit")
    ap.add_argument("--warmup", type=int, default=256)
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cache = os.path.join(HERE, "..", "..", "notebooks", f"RadialSpline_reencoded_{args.generator}.npz")
    d = np.load(cache)
    X, keep, T = d["X_rs"], d["keep"], d["T"]
    X, T = X[keep], T[keep]

    gen = RadialSplineGenerator(REFERENCE_MACHINE)
    lo, hi = gen.bounds
    span = (hi - lo)
    U = np.clip((X - lo) / span, 0.0, 1.0)  # to [0,1]^114

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(U))[: args.n]
    Ut = torch.tensor(U[idx], dtype=torch.double)
    yt = torch.tensor(T[idx], dtype=torch.double).unsqueeze(-1)
    print(f"[{args.generator}] SAASBO effective-dim probe on n={len(idx)} re-encoded designs, D={U.shape[1]}",
          flush=True)

    m = build_saasbo(Ut, yt, warmup=args.warmup, num_samples=args.samples, thinning=16, seed=args.seed)
    ls = median_lengthscales(m)
    order = np.argsort(ls)

    print("\n  shortest 15 median lengthscales (most active axes):")
    for r, j in enumerate(order[:15]):
        # decode which barrier/coordinate this axis is
        blk = gen.block
        b, off = j // blk, j % blk
        if off == 0: what = f"barrier{b}.theta_lo"
        elif off == 1: what = f"barrier{b}.theta_hi"
        elif off < 2 + gen.K: what = f"barrier{b}.c_out[{off-2}]"
        else: what = f"barrier{b}.c_in[{off-2-gen.K}]"
        print(f"    dim {j:3d}  ls={ls[j]:7.2f}   {what}")

    for cut in (1.0, 3.0, 10.0, 30.0):
        print(f"  active dims (ls < {cut:5.1f}): {(ls < cut).sum():3d} / {len(ls)}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.sort(ls), "o-", ms=3)
    ax.axhline(10.0, ls=":", color="C3", label="cutoff=10")
    ax.set_yscale("log")
    ax.set_xlabel("axis (sorted by median lengthscale)")
    ax.set_ylabel("median lengthscale (norm. input)")
    ax.set_title(f"SAASBO lengthscale spectrum — RadialSpline {args.generator} (n={len(idx)})")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(HERE, f"effdim_{args.generator}.png")
    fig.savefig(out, dpi=120)
    print("\n  wrote", out)


if __name__ == "__main__":
    main()
