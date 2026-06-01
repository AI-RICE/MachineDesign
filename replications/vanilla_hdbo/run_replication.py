"""Replication runner for Hvarfner et al. 2024 (DSP) on the paper's simplest
embedded synthetic benchmarks (Fig. 5).

Compares three surrogates inside an identical LogEI BO loop:
  * dsp      -- sqrt(D)-scaled LogNormal lengthscale prior, sigma_f^2 = 1 (the method),
  * default  -- classic Gamma(3,6) lengthscale prior + ScaleKernel (the control),
  * random   -- Sobol random search (sanity floor).

Metric: log10 regret = log10(best_so_far - f_opt) vs BO iteration, mean over seeds.
Paper claim to recover (Fig. 5, Hartmann-6 / Levy-4 embedded in D>=25): the DSP
prior reaches markedly lower regret than the default-Gamma prior in high D.

Resumable: each (func,dim,method,seed) trace is cached to a .npz and skipped if
present. Run e.g.:

  .venv/bin/python replications/vanilla_hdbo/run_replication.py \
      --func hartmann6 --dim 25 --seeds 5 --init 20 --iters 60
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.mlls import ExactMarginalLogLikelihood

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmarks import EmbeddedTestFunction  # noqa: E402
from dsp_prior import build_default_gp, build_dsp_gp  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
BUILDERS = {"dsp": build_dsp_gp, "default": build_default_gp}


def _sobol(bounds, n, seed):
    return draw_sobol_samples(bounds=bounds, n=n, q=1, seed=seed).squeeze(1)


def run_one(func, dim, method, seed, n_init, n_iters, raw_samples, restarts):
    """One BO trace. Returns best_so_far per evaluation (length n_init+n_iters)."""
    torch.manual_seed(seed)
    f = EmbeddedTestFunction(func, dim=dim, seed=seed)
    bounds = f.bounds  # (2, D), float64

    X = _sobol(bounds, n_init, seed)
    Y = f(X).unsqueeze(-1)  # minimise -> store raw; we track running min
    best = [float(Y[: i + 1].min()) for i in range(len(Y))]

    if method == "random":
        extra = _sobol(bounds, n_iters, seed + 10_000)
        Yx = f(extra).unsqueeze(-1)
        Y = torch.cat([Y, Yx])
        best = [float(Y[: i + 1].min()) for i in range(len(Y))]
        return np.array(best)

    build = BUILDERS[method]
    for _ in range(n_iters):
        # GP models maximisation of -f (BO maximises); train on negated objective.
        gp = build(X, -Y)
        fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
        acqf = LogExpectedImprovement(gp, best_f=(-Y).max())
        cand, _ = optimize_acqf(
            acqf, bounds=bounds, q=1, num_restarts=restarts, raw_samples=raw_samples
        )
        y = f(cand).unsqueeze(-1)
        X = torch.cat([X, cand])
        Y = torch.cat([Y, y])
        best.append(float(Y.min()))
    return np.array(best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--func", default="hartmann6", choices=["hartmann6", "levy4"])
    ap.add_argument("--dim", type=int, default=25)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--init", type=int, default=20)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--methods", nargs="+", default=["dsp", "default", "random"])
    ap.add_argument("--raw-samples", type=int, default=256)
    ap.add_argument("--restarts", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    f_opt = EmbeddedTestFunction(args.func, dim=args.dim, seed=0).f_opt
    for method in args.methods:
        for seed in range(args.seeds):
            tag = f"{args.func}_d{args.dim}_{method}_s{seed}"
            path = os.path.join(OUT, tag + ".npz")
            if os.path.exists(path):
                print(f"skip {tag} (cached)", flush=True)
                continue
            t0 = time.time()
            best = run_one(args.func, args.dim, method, seed, args.init, args.iters,
                           args.raw_samples, args.restarts)
            np.savez(path, best=best, f_opt=f_opt, n_init=args.init,
                     func=args.func, dim=args.dim, method=method, seed=seed)
            print(f"done {tag}: final best={best[-1]:.4f} "
                  f"logregret={np.log10(max(best[-1]-f_opt,1e-10)):.3f} ({time.time()-t0:.0f}s)",
                  flush=True)

    # aggregate
    print("\n=== summary: log10 regret at final iter (mean +/- std over seeds) ===")
    for method in args.methods:
        finals = []
        for seed in range(args.seeds):
            p = os.path.join(OUT, f"{args.func}_d{args.dim}_{method}_s{seed}.npz")
            if os.path.exists(p):
                b = np.load(p)["best"]
                finals.append(np.log10(max(b[-1] - f_opt, 1e-10)))
        if finals:
            print(f"  {method:8s}: {np.mean(finals):+.3f} +/- {np.std(finals):.3f}  (n={len(finals)})")


if __name__ == "__main__":
    main()
