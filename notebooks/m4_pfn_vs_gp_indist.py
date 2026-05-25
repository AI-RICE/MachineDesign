"""In-distribution sanity check: PFN vs Type-II-ML GP on samples from the PFN's own training prior.

If the GP-prior PFN is correctly Bayesian under its training prior (Nagler 2023),
then on functions sampled from THAT prior, the PFN's predictions should be
close to a Type-II-ML GP's — both are reasonable estimators in-distribution.

A persistent PFN vs GP gap on in-distribution data is the smoking gun for
PFN training under-fitting or architecture-bound — independent of the OOD
(FEA-vs-prior) question.

Protocol per task:
  - Sample (ls, outputscale, noise, nu) from GPPriorConfig (the very same
    distribution used during training).
  - Sample N = n_ctx + n_test inputs in [0, 1]^D, generate y via Cholesky.
  - Split into context (n_ctx) and test (n_test).
  - PFN predicts on test points with that context (PFNSurrogate from_loaded_with_real_Y).
  - Type-II ML GP (SingleTaskGP, Matern+ARD, BoTorch) trained on context, predicts test.
  - Score both: RMSE, Spearman, R^2.

Repeat for many random tasks, aggregate. If PFN ~ GP on these tasks, the
FEA gap (m4_parametric_bottleneck.py) is purely OOD — PFN training is fine.
If PFN ≪ GP here too, PFN itself is under-trained / architecture-bound.

Note: we use the GENERATOR's bounds to size X_raw for the surrogate, but the
underlying GP samples are in [0, 1]^D unit space — exactly as during training.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score

from machine_design.generators import HacklGenerator_OneLambda
from machine_design.lumped import REFERENCE_MACHINE
from machine_design.pfn import PFNSurrogate, load_checkpoint
from machine_design.pfn.gp_prior_sampler import GPPriorConfig, GPPriorSampler


def _metrics(y_true, y_pred) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    rho, _ = spearmanr(y_true, y_pred)
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "spearman_rho": float(rho), "r2": r2}


def _pfn_predict(loaded, X_ctx_np, y_ctx_np, X_test_np) -> np.ndarray:
    surr = PFNSurrogate.from_loaded_with_real_Y(
        loaded,
        torch.from_numpy(X_ctx_np.astype(np.float32)),
        torch.from_numpy(y_ctx_np.astype(np.float32)).unsqueeze(-1),
    )
    with torch.no_grad():
        post = surr.posterior(torch.from_numpy(X_test_np.astype(np.float32)).unsqueeze(0))
        mean_norm = post.mean.squeeze().cpu().numpy()
    return mean_norm * surr.y_std + surr.y_mean


def _gp_predict(X_ctx, y_ctx, X_test, bounds_t):
    X_tr = torch.from_numpy(X_ctx.astype(np.float64))
    Y_tr = torch.from_numpy(y_ctx.astype(np.float64)).unsqueeze(-1)
    gp = SingleTaskGP(X_tr, Y_tr, input_transform=Normalize(d=X_tr.shape[1], bounds=bounds_t))
    fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
    gp.eval()
    with torch.no_grad():
        X_te = torch.from_numpy(X_test.astype(np.float64))
        return gp.posterior(X_te).mean.squeeze(-1).cpu().numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n-tasks", type=int, default=50)
    ap.add_argument("--n-ctx", type=int, default=64)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=Path("sweeps"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_checkpoint(args.checkpoint)
    D = loaded.input_dim

    # Use the OneLambda generator's bounds so X_raw is on the same scale the
    # PFN saw during training (the per-dim x-norm uses these bounds).
    gen = HacklGenerator_OneLambda(REFERENCE_MACHINE, r_stator_end=0.7, offset=0.35)
    lo, hi = gen.bounds
    bounds_np = np.stack([np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)])
    bounds_t = torch.from_numpy(bounds_np)
    assert bounds_t.shape[1] == D, f"generator bounds D={bounds_t.shape[1]} != ckpt D={D}"

    cfg = GPPriorConfig()  # same defaults the PFN was trained on (wide)
    sampler = GPPriorSampler(input_dim=D, bounds=bounds_np, cfg=cfg)

    rng = np.random.default_rng(args.seed)
    rows = []
    print(f"In-distribution PFN-vs-GP on {args.n_tasks} GP-prior tasks (n_ctx={args.n_ctx}, n_test={args.n_test}, D={D})")
    print(f"GPPriorConfig: log_ls_std={cfg.log_ls_std}, log_noise=[{cfg.log_noise_min}, {cfg.log_noise_max}], nu={cfg.nu_choices}")
    print()

    for k in range(args.n_tasks):
        # IMPORTANT: normalise=False — otherwise the sampler z-scores y using
        # stats from ALL n_ctx+n_test points, but PFNSurrogate at inference
        # z-scores using only the n_ctx context. That scale mismatch would
        # spuriously inflate the PFN's RMSE. With normalise=False, both PFN
        # and GP work in raw GP-sample units; the per-context z-score lives
        # only inside the surrogate's forward path (matching training).
        task = sampler.sample(rng, n_context=args.n_ctx, n_target=args.n_test, normalise=False)
        X_ctx = task.x_context
        y_ctx = task.y_context
        X_test = task.x_target
        y_test = task.y_target

        pfn_pred = _pfn_predict(loaded, X_ctx, y_ctx, X_test)
        try:
            gp_pred = _gp_predict(X_ctx, y_ctx, X_test, bounds_t)
        except Exception as e:
            print(f"  task {k:>3}: GP fit failed ({type(e).__name__})")
            continue

        m_pfn = _metrics(y_test, pfn_pred)
        m_gp = _metrics(y_test, gp_pred)
        rows.append({"task": k, **{f"pfn_{k2}": v for k2, v in m_pfn.items()},
                     **{f"gp_{k2}": v for k2, v in m_gp.items()}})

        if k < 10 or k % 10 == 0:
            print(f"  task {k:>3}: PFN RMSE={m_pfn['rmse']:.3f}  ρ={m_pfn['spearman_rho']:+.2f}  | "
                  f"GP RMSE={m_gp['rmse']:.3f}  ρ={m_gp['spearman_rho']:+.2f}  | "
                  f"PFN/GP RMSE ratio = {m_pfn['rmse']/m_gp['rmse']:.2f}")

    df = pd.DataFrame(rows)
    csv_path = args.out_dir / "OneLambda_pfn_vs_gp_indist.csv"
    df.to_csv(csv_path, index=False)

    print("\n=== aggregate (in-distribution, n_tasks={}) ===".format(len(df)))
    print(f"PFN RMSE: median={df['pfn_rmse'].median():.3f}  mean={df['pfn_rmse'].mean():.3f}")
    print(f"GP  RMSE: median={df['gp_rmse'].median():.3f}  mean={df['gp_rmse'].mean():.3f}")
    print(f"PFN/GP RMSE ratio: median={(df['pfn_rmse']/df['gp_rmse']).median():.2f}  mean={(df['pfn_rmse']/df['gp_rmse']).mean():.2f}")
    print(f"PFN ρ: median={df['pfn_spearman_rho'].median():+.3f}")
    print(f"GP  ρ: median={df['gp_spearman_rho'].median():+.3f}")
    print(f"\nWrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
