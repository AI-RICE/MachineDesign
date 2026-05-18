"""Build one FEA emulator per parameterisation (M4.5).

Loads all FEA-evaluated designs from `results/results*/`, runs 5-fold CV
with RMSE reported separately on the uniform-initials and BO-trace
subsets, then trains the production emulator on all data and saves it
to `emulators/<gen>_fea_emulator.joblib`.

Per CLAUDE.md §6.5 we report two CV RMSEs per output so callers can see
that BO-trace points are predicted more accurately than uniform initials
(densely-sampled BO regions are easier — that's by design).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from machine_design.fea_emulator import (
    FEAEmulator,
    cv_evaluate,
    load_fea_designs,
)


GENERATORS = ("OneLambda", "SixLambdas", "ThreeBrokenLines")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--out-dir", type=Path, default=Path("emulators"))
    ap.add_argument("--only", choices=GENERATORS, default=None)
    ap.add_argument("--constrained-only", action="store_true",
                    help="train only on the constrained sweep (True suffix)")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    targets = (args.only,) if args.only else GENERATORS
    args.out_dir.mkdir(parents=True, exist_ok=True)

    constrained_flag = True if args.constrained_only else None

    for gen in targets:
        print(f"\n{'='*60}\n{gen}\n{'='*60}")
        t0 = time.time()
        loaded = load_fea_designs(gen, results_root=args.results_root,
                                  constrained=constrained_flag)
        print(f"  loaded {len(loaded.X)} designs, D={loaded.X.shape[1]}")
        print(f"    uniform initials: {int(loaded.is_uniform_init.sum())}")
        print(f"    BO trace:         {int((~loaded.is_uniform_init).sum())}")
        print(f"    constrained set:  {int(loaded.constrained.sum())}")
        print(f"    unconstrained:    {int((~loaded.constrained).sum())}")
        print(f"    T_mean   range:   [{loaded.T_mean.min():.3f}, {loaded.T_mean.max():.3f}] mean={loaded.T_mean.mean():.3f}")
        print(f"    T_ripple range:   [{loaded.T_ripple.min():.3f}, {loaded.T_ripple.max():.3f}] mean={loaded.T_ripple.mean():.3f}")

        print(f"\n  {args.k}-fold CV (honest RMSE on held-out folds):")
        cv = cv_evaluate(
            loaded.X, loaded.T_mean, loaded.T_ripple, loaded.is_uniform_init,
            k=args.k, seed=0,
        )
        print(f"    T_mean   RMSE  uniform = {cv['rmse_T_uniform_mean']:.3f} ± {cv['rmse_T_uniform_std']:.3f}")
        print(f"    T_mean   RMSE  BO      = {cv['rmse_T_bo_mean']:.3f} ± {cv['rmse_T_bo_std']:.3f}")
        print(f"    T_ripple RMSE  uniform = {cv['rmse_R_uniform_mean']:.2f}% ± {cv['rmse_R_uniform_std']:.2f}%")
        print(f"    T_ripple RMSE  BO      = {cv['rmse_R_bo_mean']:.2f}% ± {cv['rmse_R_bo_std']:.2f}%")

        # Train production emulator on all data.
        em = FEAEmulator(generator_short=gen)
        em.fit(loaded.X, loaded.T_mean, loaded.T_ripple)

        out_path = args.out_dir / f"{gen}_fea_emulator.joblib"
        em.save(out_path)
        print(f"\n  saved {out_path}  ({time.time() - t0:.1f}s)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
