"""Generate the lumped library for one parameterisation (M2).

Library entries store v3 saturated `T_proxy` evaluated at all three
granularities, so the runtime sampler can amortise across granularity
without ever calling the solver online.

Usage:
    python notebooks/m2_build_library.py OneLambda --n 1000 --workers 8
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from machine_design.pfn.library import build_library, save_library


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("generator", choices=["OneLambda", "SixLambdas", "ThreeBrokenLines"])
    ap.add_argument("--n", type=int, default=200, help="number of samples to attempt")
    ap.add_argument("--workers", type=int, default=None,
                    help="ProcessPool workers (default: all cores)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=Path("libraries"))
    args = ap.parse_args()

    print(f"Generator: {args.generator}")
    print(f"N requested: {args.n}")
    print(f"Workers: {args.workers or 'auto'}")
    print()

    t0 = time.time()
    lib = build_library(
        args.generator, n_samples=args.n, base_seed=args.seed,
        n_workers=args.workers, verbose=True,
    )
    dt = time.time() - t0
    print(f"\nBuilt library: {len(lib)} entries, D={lib.params.shape[1]}, in {dt:.1f}s")

    out_path = args.out_dir / f"{args.generator}_n{len(lib)}_s{args.seed}.npz"
    save_library(lib, out_path)
    print(f"Wrote {out_path}")

    for g, arr in lib.T_proxy.items():
        print(f"  {g:<7}  T_proxy mean={arr.mean():.3e}  std={arr.std():.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
