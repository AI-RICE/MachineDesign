"""BO-quality benchmark: PFN-EI vs GP-EI vs Random.

Two target families:
  --target gp   : functions sampled from the same wide GP prior the PFN was
                  trained on. In-distribution test (Nagler-theorem-relevant).
  --target fea  : FEA-emulator T_mean values for OneLambda designs. Out-of-
                  distribution test (FEA != GP samples).

Protocol (discrete BO over a candidate pool):
  - Build a pool of N candidates and corresponding true y values.
  - Initialise with k_init random points; run T BO iterations.
  - At each iteration, surrogate fitted to (X_obs, y_obs); compute analytic
    EI on remaining pool; pick argmax; evaluate true y; add to context.
  - Track best-so-far per surrogate.

Discrete BO over a pool sidesteps acquisition-optimisation differences
between PFN and GP (we don't want to confound surrogate-quality differences
with optimisation-of-EI differences) and matches the paired-comparison
protocols used in the BO literature.

Output: CSV (per seed × iter × surrogate) + PDF (simple-regret curves).
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.stats import norm

from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from gpytorch.mlls import ExactMarginalLogLikelihood

from machine_design.pfn import PFNSurrogate, load_checkpoint
from machine_design.pfn.gp_prior_sampler import GPPriorConfig, GPPriorSampler
from machine_design.generators import HacklGenerator_OneLambda
from machine_design.lumped import REFERENCE_MACHINE


def _analytic_ei(mu: np.ndarray, sigma: np.ndarray, best: float) -> np.ndarray:
    sigma = np.maximum(sigma, 1e-12)
    z = (mu - best) / sigma
    return (mu - best) * norm.cdf(z) + sigma * norm.pdf(z)


def _pfn_predict(loaded, X_ctx, y_ctx, X_query):
    """Return (mean, std) for X_query in RAW y units."""
    surr = PFNSurrogate.from_loaded_with_real_Y(
        loaded,
        torch.from_numpy(X_ctx.astype(np.float32)),
        torch.from_numpy(y_ctx.astype(np.float32)).unsqueeze(-1),
    )
    with torch.no_grad():
        post = surr.posterior(torch.from_numpy(X_query.astype(np.float32)).unsqueeze(0))
        m_norm = post.mean.squeeze().cpu().numpy()
        v_norm = post.variance.squeeze().cpu().numpy()
    mu = m_norm * surr.y_std + surr.y_mean
    sigma = np.sqrt(np.maximum(v_norm, 1e-20)) * surr.y_std
    return mu, sigma


def _gp_predict(X_ctx, y_ctx, X_query, bounds_t):
    Xt = torch.from_numpy(X_ctx.astype(np.float64))
    yt = torch.from_numpy(y_ctx.astype(np.float64)).unsqueeze(-1)
    gp = SingleTaskGP(Xt, yt, input_transform=Normalize(d=Xt.shape[1], bounds=bounds_t))
    fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
    gp.eval()
    with torch.no_grad():
        Xq = torch.from_numpy(X_query.astype(np.float64))
        post = gp.posterior(Xq)
        mu = post.mean.squeeze(-1).cpu().numpy()
        sigma = post.variance.squeeze(-1).clamp_min(1e-20).sqrt().cpu().numpy()
    return mu, sigma


def _run_one_bo(X_pool, y_pool, k_init, T, predict_fn, rng):
    """Generic BO loop. predict_fn(X_obs, y_obs, X_query) -> (mu, sigma)."""
    init_idx = rng.choice(len(X_pool), size=k_init, replace=False)
    obs_mask = np.zeros(len(X_pool), bool); obs_mask[init_idx] = True
    X_obs = X_pool[init_idx].copy()
    y_obs = y_pool[init_idx].copy()
    best_history = [float(y_obs.max())]

    for _ in range(T):
        rem_idx = np.where(~obs_mask)[0]
        if predict_fn is None:           # Random baseline
            pick = rng.choice(rem_idx)
        else:
            mu, sigma = predict_fn(X_obs, y_obs, X_pool[rem_idx])
            ei = _analytic_ei(mu, sigma, float(y_obs.max()))
            pick = rem_idx[int(ei.argmax())]
        X_obs = np.vstack([X_obs, X_pool[pick][None]])
        y_obs = np.append(y_obs, y_pool[pick])
        obs_mask[pick] = True
        best_history.append(float(y_obs.max()))
    return np.array(best_history)


def _build_gp_target(rng, n_pool, D, bounds_np):
    """Sample one GP function on n_pool random inputs from the wide prior."""
    sampler = GPPriorSampler(input_dim=D, bounds=bounds_np, cfg=GPPriorConfig())
    # n_target=0 in sampler doesn't exist; trick by treating all as context with
    # normalise=False, then peel apart.
    task = sampler.sample(rng, n_context=n_pool, n_target=1, normalise=False)
    X = np.vstack([task.x_context, task.x_target])  # (n_pool+1, D)
    y = np.concatenate([task.y_context, task.y_target])
    return X[:n_pool], y[:n_pool]


def _build_fea_target(rng, n_pool):
    """Random sample of FEA-evaluated OneLambda designs (without replacement)."""
    from machine_design.fea_emulator import load_fea_designs
    fea = load_fea_designs("OneLambda")
    idx = rng.choice(len(fea.X), size=min(n_pool, len(fea.X)), replace=False)
    return fea.X[idx], fea.T_mean[idx]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=["gp", "fea"])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n-pool", type=int, default=500)
    ap.add_argument("--k-init", type=int, default=8)
    ap.add_argument("--T", type=int, default=40)
    ap.add_argument("--n-seeds", type=int, default=20)
    ap.add_argument("--out-dir", type=Path, default=Path("sweeps/bo_bench"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_checkpoint(args.checkpoint)
    gen = HacklGenerator_OneLambda(REFERENCE_MACHINE, r_stator_end=0.7, offset=0.35)
    lo, hi = gen.bounds
    bounds_np = np.stack([np.asarray(lo, float), np.asarray(hi, float)])
    bounds_t = torch.from_numpy(bounds_np)
    D = bounds_np.shape[1]

    rows = []
    for seed in range(args.n_seeds):
        rng = np.random.default_rng(seed)
        if args.target == "gp":
            X_pool, y_pool = _build_gp_target(rng, args.n_pool, D, bounds_np)
        else:
            X_pool, y_pool = _build_fea_target(rng, args.n_pool)
        y_max_true = float(y_pool.max())

        # Same initial points (RNG re-seed per surrogate so init is identical)
        for label, predict in [
            ("PFN",    lambda Xo, yo, Xq, _l=loaded: _pfn_predict(_l, Xo, yo, Xq)),
            ("GP",     lambda Xo, yo, Xq, _b=bounds_t: _gp_predict(Xo, yo, Xq, _b)),
            ("Random", None),
        ]:
            r = np.random.default_rng(seed)
            hist = _run_one_bo(X_pool, y_pool, args.k_init, args.T, predict, r)
            regret = y_max_true - hist
            for t, (b, sr) in enumerate(zip(hist, regret)):
                rows.append({"seed": seed, "iter": t, "surrogate": label,
                             "best_so_far": float(b), "simple_regret": float(sr),
                             "y_max_true": y_max_true})

        s_summary = pd.DataFrame(rows).query(f"seed=={seed} & iter=={args.T}")
        s_summary = s_summary.set_index("surrogate")["simple_regret"]
        print(f"seed {seed:>2}: PFN regret={s_summary.get('PFN', float('nan')):.4f}  "
              f"GP regret={s_summary.get('GP', float('nan')):.4f}  "
              f"Random regret={s_summary.get('Random', float('nan')):.4f}", flush=True)

    df = pd.DataFrame(rows)
    csv_path = args.out_dir / f"bo_{args.target}.csv"
    df.to_csv(csv_path, index=False)

    # Aggregate: median + IQR per (surrogate, iter)
    summary = df.groupby(["surrogate", "iter"])["simple_regret"].agg(
        median="median", p25=lambda x: x.quantile(0.25), p75=lambda x: x.quantile(0.75)
    ).reset_index()

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    palette = {"PFN": "#e31a1c", "GP": "#1f78b4", "Random": "#888888"}
    for s in ("PFN", "GP", "Random"):
        sub = summary[summary.surrogate == s]
        ax.plot(sub["iter"], sub["median"], "-", color=palette[s], label=s, lw=2)
        ax.fill_between(sub["iter"], sub["p25"], sub["p75"], color=palette[s], alpha=0.18)
    ax.set_yscale("log")
    ax.set_xlabel("BO iteration")
    ax.set_ylabel("Simple regret (log)")
    ax.set_title(f"BO benchmark — target={args.target}  n_seeds={args.n_seeds}  k_init={args.k_init}")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    pdf = args.out_dir / f"bo_{args.target}.pdf"
    png = args.out_dir / f"bo_{args.target}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=180, bbox_inches="tight")
    print(f"\nWrote {csv_path}\nWrote {pdf} and {png}")

    # Final-iter summary
    final = df[df["iter"] == args.T].groupby("surrogate")["simple_regret"].agg(
        median="median", mean="mean", std="std"
    )
    print("\n=== final-iteration simple regret (median / mean +/- std) ===")
    print(final.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
