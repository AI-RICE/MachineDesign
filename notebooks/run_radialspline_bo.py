"""Phase-2 application: DSP-GP Bayesian optimization over the 114-D RadialSpline
SynRM rotor parameterisation, maximising T_mean.

Strategy (validated in replications/vanilla_hdbo, E3): vanilla global GP-BO with
the dimensionality-scaled LogNormal lengthscale prior (Hvarfner et al. 2024),
via BoTorch's `get_covar_module_with_dim_scaled_prior` + analytic LogEI.

Two RadialSpline-specific simplifications/notes:
  * the repair decoder makes EVERY box point feasible -> no rejection loop;
  * each raw candidate is decoded to geometry and RE-ENCODED (project onto the
    actual-geometry coordinates the emulator understands) before scoring.

ORACLE & SCOPE CAVEAT (PARAMETERISATION.md §13, CLAUDE.md §6.5):
  The only torque data touching RadialSpline is the re-encoded Hackl designs
  (E2). The emulator (GBM on those) is valid ONLY on the Hackl manifold. BO over
  the rich 114-D space roams OFF-manifold, where the emulator EXTRAPOLATES.
  => This is a Phase-2 machinery shakedown + on-manifold-recovery check, NOT a
     scientific optimum. On-manifold best (reference) = max re-encoded T_mean.
     Any BO design must be FEA-verified (Phase 3) before it is believed.

Run:
  .venv/bin/python notebooks/run_radialspline_bo.py --generator OneLambda \
      --seeds 3 --init 20 --iters 60
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
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.mlls import ExactMarginalLogLikelihood
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from machine_design.generators import RadialSplineGenerator  # noqa: E402
from machine_design.lumped.geometry import REFERENCE_MACHINE  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "RadialSpline_bo_results")


def build_emulator(generator_short):
    """GBM emulator: re-encoded RadialSpline coords -> T_mean (on-manifold only)."""
    cache = os.path.join(HERE, f"RadialSpline_reencoded_{generator_short}.npz")
    d = np.load(cache)
    X, keep, T = d["X_rs"], d["keep"], d["T"]
    X, T = X[keep], T[keep]
    gbm = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, min_samples_leaf=10,
                                        l2_regularization=1.0, random_state=0).fit(X, T)
    return gbm, float(T.max()), X.shape[1]


def make_objective(gen, gbm, lo, span):
    """u in [0,1]^D -> decode (repair) -> re-encode -> emulator T_mean."""
    def f(u_row: np.ndarray) -> float:
        X_raw = lo + u_row * span
        gen.set_parameters(X_raw)
        X_canon = gen.fit_barriers(gen.generate_barriers())  # project onto geometry coords
        return float(gbm.predict(X_canon.reshape(1, -1))[0])
    return f


def dsp_gp(u: torch.Tensor, y: torch.Tensor) -> SingleTaskGP:
    d = u.shape[-1]
    return SingleTaskGP(
        u, y,
        covar_module=get_covar_module_with_dim_scaled_prior(ard_num_dims=d),  # DSP, validated in E3
        outcome_transform=Standardize(m=1),
    )  # inputs already in [0,1]^D -> no input transform needed


def run_one(method, seed, gen, objective, D, n_init, n_iters):
    torch.manual_seed(seed)
    unit = torch.stack([torch.zeros(D, dtype=torch.double), torch.ones(D, dtype=torch.double)])
    U = draw_sobol_samples(bounds=unit, n=n_init, q=1, seed=seed).squeeze(1)
    Y = torch.tensor([objective(u.numpy()) for u in U], dtype=torch.double).unsqueeze(-1)
    best = [float(Y[: i + 1].max()) for i in range(len(Y))]

    if method == "random":
        extra = draw_sobol_samples(bounds=unit, n=n_iters, q=1, seed=seed + 9999).squeeze(1)
        for u in extra:
            Y = torch.cat([Y, torch.tensor([[objective(u.numpy())]], dtype=torch.double)])
            best.append(float(Y.max()))
        return np.array(best)

    for _ in range(n_iters):
        gp = dsp_gp(U, Y)
        fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
        acqf = LogExpectedImprovement(gp, best_f=Y.max())
        cand, _ = optimize_acqf(acqf, bounds=unit, q=1, num_restarts=4, raw_samples=256)
        y = objective(cand.squeeze(0).numpy())
        U = torch.cat([U, cand])
        Y = torch.cat([Y, torch.tensor([[y]], dtype=torch.double)])
        best.append(float(Y.max()))
    return np.array(best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", default="OneLambda")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--init", type=int, default=20)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--methods", nargs="+", default=["dsp", "random"])
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    gen = RadialSplineGenerator(REFERENCE_MACHINE)
    lo, hi = gen.bounds
    lo, span = lo.astype(float), (hi - lo).astype(float)
    D = lo.shape[0]
    gbm, on_manifold_best, n_feat = build_emulator(args.generator)
    assert n_feat == D, (n_feat, D)
    objective = make_objective(gen, gbm, lo, span)
    print(f"RadialSpline D={D} | emulator on-manifold best T_mean = {on_manifold_best:.4f} N·m "
          f"(reference; off-manifold = extrapolation)", flush=True)

    for method in args.methods:
        for seed in range(args.seeds):
            tag = f"{args.generator}_{method}_it{args.iters}_s{seed}"
            path = os.path.join(OUT, tag + ".npz")
            if os.path.exists(path):
                print(f"skip {tag} (cached)", flush=True)
                continue
            t0 = time.time()
            best = run_one(method, seed, gen, objective, D, args.init, args.iters)
            np.savez(path, best=best, n_init=args.init, on_manifold_best=on_manifold_best,
                     method=method, seed=seed)
            print(f"done {tag}: best T_mean={best[-1]:.4f} "
                  f"({100*best[-1]/on_manifold_best:.1f}% of on-manifold best) ({time.time()-t0:.0f}s)",
                  flush=True)

    print("\n=== best T_mean reached (mean ± std over seeds) ===")
    for method in args.methods:
        finals = [float(np.load(os.path.join(OUT, f"{args.generator}_{method}_it{args.iters}_s{s}.npz"))["best"][-1])
                  for s in range(args.seeds)
                  if os.path.exists(os.path.join(OUT, f"{args.generator}_{method}_it{args.iters}_s{s}.npz"))]
        if finals:
            print(f"  {method:7s}: {np.mean(finals):.4f} ± {np.std(finals):.4f}  "
                  f"({100*np.mean(finals)/on_manifold_best:.1f}% of on-manifold best={on_manifold_best:.3f})")


if __name__ == "__main__":
    main()
