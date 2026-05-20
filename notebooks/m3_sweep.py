"""M3 overnight sweep: build libraries + train PFNs for all three parameterisations.

Sequential because the bottleneck per parameterisation is library generation
(~3 h with 8 workers for n=100k via v3 saturated) and PFN training
(~1-5 h on GPU). One per parameterisation; three total.

Defaults are sized for an overnight run. Override with CLI args if you
want a shorter / smaller / GPU-less sweep first.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


GENERATORS = ("OneLambda", "SixLambdas", "ThreeBrokenLines")


def _run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}")
    t0 = time.time()
    rc = subprocess.call(cmd)
    dt = time.time() - t0
    print(f"  exit={rc}  elapsed={dt/60:.1f} min")
    if rc != 0:
        sys.exit(rc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-library", type=int, default=100_000,
                    help="library size per parameterisation (default: 100k)")
    ap.add_argument("--workers", type=int, default=os.cpu_count(),
                    help="ProcessPool workers for library generation")
    ap.add_argument("--steps", type=int, default=200_000,
                    help="PFN training steps per parameterisation")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--nlayers", type=int, default=6)
    ap.add_argument("--ninp", type=int, default=512)
    ap.add_argument("--nhid", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--granularity", default="random",
                    choices=["random", "COARSE", "MEDIUM", "FINE"])
    ap.add_argument("--lib-dir", type=Path, default=Path("libraries"))
    ap.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints"))
    ap.add_argument("--skip-library", action="store_true",
                    help="skip library generation (assume libraries already exist)")
    ap.add_argument("--only", choices=GENERATORS,
                    help="run only one parameterisation")
    args = ap.parse_args()

    targets = (args.only,) if args.only else GENERATORS
    args.lib_dir.mkdir(parents=True, exist_ok=True)
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)

    overall_t0 = time.time()
    for gen in targets:
        print(f"\n{'='*60}\n{gen}\n{'='*60}")
        lib_path = args.lib_dir / f"{gen}_n{args.n_library}_s{args.seed}.npz"
        ckpt_path = args.ckpt_dir / f"{gen}_pfn.pt"

        if not args.skip_library:
            _run([
                sys.executable, "-u", "notebooks/m2_build_library.py", gen,
                "--n", str(args.n_library),
                "--workers", str(args.workers),
                "--seed", str(args.seed),
                "--out-dir", str(args.lib_dir),
            ])
        else:
            print(f"--skip-library: assuming {lib_path} exists")

        _run([
            sys.executable, "-u", "-m", "machine_design.pfn.train",
            str(lib_path),
            "--out", str(ckpt_path),
            "--granularity", args.granularity,
            "--steps", str(args.steps),
            "--batch-size", str(args.batch_size),
            "--lr", str(args.lr),
            "--nlayers", str(args.nlayers),
            "--ninp", str(args.ninp),
            "--nhid", str(args.nhid),
            "--seed", str(args.seed),
        ])

    total_min = (time.time() - overall_t0) / 60.0
    print(f"\nSweep complete in {total_min:.1f} min ({total_min/60:.1f} h)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
