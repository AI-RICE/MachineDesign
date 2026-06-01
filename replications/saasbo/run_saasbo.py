"""Elementary BO comparison (H-REPL): SAASBO vs DSP vs random on embedded
Hartmann-6, the sparse-subspace regime SAASBO is built for. Same benchmark and
LogEI loop as the DSP replication (../vanilla_hdbo), surrogate swapped.

  .venv/bin/python replications/saasbo/run_saasbo.py --dim 50 --seeds 2 --init 10 --iters 25
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.mlls import ExactMarginalLogLikelihood

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "vanilla_hdbo"))

from benchmarks import EmbeddedTestFunction  # noqa: E402
from dsp_prior import build_dsp_gp  # noqa: E402
from saasbo import active_dimensions, build_saasbo  # noqa: E402

OUT = os.path.join(HERE, "results")


def _sobol(b, n, s):
    return draw_sobol_samples(bounds=b, n=n, q=1, seed=s).squeeze(1)


def run_one(method, seed, f, D, n_init, n_iters):
    torch.manual_seed(seed)
    bounds = f.bounds
    X = _sobol(bounds, n_init, seed)
    Y = (-f(X)).unsqueeze(-1)  # maximise -Hartmann
    best = [float(f(X[: i + 1]).min()) for i in range(len(X))]  # min f (regret) so far
    recovered = None
    if method == "random":
        extra = _sobol(bounds, n_iters, seed + 9999)
        allX = torch.cat([X, extra])
        best = [float(f(allX[: i + 1]).min()) for i in range(len(allX))]
        return np.array(best), None
    for _ in range(n_iters):
        if method == "saasbo":
            gp = build_saasbo(X, Y, warmup=128, num_samples=128, thinning=16)
            acqf = qLogExpectedImprovement(gp, best_f=Y.max())
        else:  # dsp
            gp = build_dsp_gp(X, Y)
            fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
            acqf = LogExpectedImprovement(gp, best_f=Y.max())
        cand, _ = optimize_acqf(acqf, bounds=bounds, q=1, num_restarts=4, raw_samples=256)
        X = torch.cat([X, cand])
        Y = torch.cat([Y, (-f(cand)).unsqueeze(-1)])
        best.append(float(f(X).min()))
    if method == "saasbo":
        act, _ = active_dimensions(gp, cutoff=10.0)
        recovered = len(set(act.tolist()) & set(f.active.tolist()))
    return np.array(best), recovered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=50)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--init", type=int, default=10)
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--methods", nargs="+", default=["saasbo", "dsp", "random"])
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    f_opt = -3.32237

    for method in args.methods:
        for seed in range(args.seeds):
            f = EmbeddedTestFunction("hartmann6", dim=args.dim, seed=seed)
            tag = f"hartmann6_d{args.dim}_{method}_s{seed}"
            path = os.path.join(OUT, tag + ".npz")
            if os.path.exists(path):
                print(f"skip {tag}", flush=True)
                continue
            t0 = time.time()
            best, rec = run_one(method, seed, f, args.dim, args.init, args.iters)
            np.savez(path, best=best, n_init=args.init, recovered=(-1 if rec is None else rec))
            extra = f" | recovered {rec}/6 active dims" if rec is not None else ""
            print(f"done {tag}: logregret={np.log10(max(best[-1]-f_opt,1e-10)):.3f} "
                  f"({time.time()-t0:.0f}s){extra}", flush=True)

    print("\n=== final log10 regret (mean ± std) ===")
    for method in args.methods:
        v = [np.log10(max(float(np.load(os.path.join(OUT, f"hartmann6_d{args.dim}_{method}_s{s}.npz"))["best"][-1]) - f_opt, 1e-10))
             for s in range(args.seeds)
             if os.path.exists(os.path.join(OUT, f"hartmann6_d{args.dim}_{method}_s{s}.npz"))]
        if v:
            print(f"  {method:7s}: {np.mean(v):+.3f} ± {np.std(v):.3f}")


if __name__ == "__main__":
    main()
