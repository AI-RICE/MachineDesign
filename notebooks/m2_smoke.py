"""M2 smoke test: verify the prior sampler at ≥10⁴ tasks/s on CPU.

Assumes a smoke library exists at `libraries/<gen>_n*.npz`. Run
`m2_build_library.py` first if not.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from machine_design.pfn import PriorSampler, load_library


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("library", type=Path, help="path to .npz library")
    ap.add_argument("--n-tasks", type=int, default=20_000)
    ap.add_argument("--n-context", type=int, default=32)
    ap.add_argument("--n-target", type=int, default=1)
    ap.add_argument("--granularity", default="random",
                    choices=["random", "COARSE", "MEDIUM", "FINE"])
    args = ap.parse_args()

    print(f"Loading library: {args.library}")
    lib = load_library(args.library)
    print(f"  generator: {lib.generator_name}")
    print(f"  N entries: {len(lib)}, D = {lib.params.shape[1]}")

    sampler = PriorSampler(lib, granularity=args.granularity)
    print(f"  granularity mode: {args.granularity}")
    print(f"  input_dim: {sampler.input_dim}")
    print()

    # Warm-up draw so JIT-like effects don't taint the timing.
    rng = np.random.default_rng(0)
    sampler.sample(rng, n_context=args.n_context, n_target=args.n_target)

    t0 = time.time()
    counts_per_g: dict[str, int] = {}
    for _ in range(args.n_tasks):
        task = sampler.sample(
            rng, n_context=args.n_context, n_target=args.n_target,
        )
        counts_per_g[task.granularity] = counts_per_g.get(task.granularity, 0) + 1
    dt = time.time() - t0

    rate = args.n_tasks / dt
    print(f"Drew {args.n_tasks} tasks in {dt:.2f}s  → {rate:,.0f} tasks/s")
    print(f"Granularity counts: {counts_per_g}")

    threshold = 10_000.0
    if rate >= threshold:
        print(f"\nPASS: {rate:,.0f} ≥ {threshold:,.0f} tasks/s")
        return 0
    print(f"\nFAIL: {rate:,.0f} < {threshold:,.0f} tasks/s")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
