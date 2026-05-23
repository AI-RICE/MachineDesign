"""Is the lumped-prior PFN parametric-bottlenecked?

Three diagnostics in one pass, on the 1,250 FEA-evaluated OneLambda designs:

  1. Predictive RMSE / Spearman of:
        - PFN(context=64 FEA points) on a held-out FEA test set
        - GBM(n_train=k FEA points) for k in {16, 32, 64, 128, 256, 500, all-but-test}
     Both surrogates measured on identical test sets.

     If GBM at k≈64 already meets or beats the PFN's number, the matched-prior
     advantage at low n is mostly noise; the PFN brings no extra information
     beyond what 64 FEA points can express directly.
     If GBM keeps improving past the PFN as k grows toward 1000, the lumped
     prior has a capacity ceiling (parametric bottleneck): no amount of
     in-context FEA evidence can lift the PFN past what the lumped solver
     can express, because the PFN was meta-trained against that solver.

  2. Residual structure: PFN_pred vs FEA_true scatter + best-fit line.
     Pattern in residuals (curvature / clusters) → systematic miss → bottleneck.
     Random scatter → noise-limited, would improve with more / better data.

  3. PFN saturation with context: PFN predictive RMSE as n_ctx grows
     {16, 32, 50, 64}. Capped at 64 because the model was meta-trained with
     n_context_max=64. A truly capacity-rich prior keeps improving; a
     bottlenecked one plateaus.

Saves a CSV summary + PDF figure to `sweeps/`. Pure analysis — no FEA, no ANSYS.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor

from machine_design.fea_emulator import load_fea_designs
from machine_design.pfn import PFNSurrogate, load_checkpoint


def _pfn_predict(loaded, X_ctx_np, Y_ctx_np, X_test_np) -> np.ndarray:
    """Mean prediction from PFN with given context, on test points.

    `PFNSurrogate.posterior.mean` is in per-context z-score space; we must
    de-normalise with the same context (mean, std) to recover raw T_mean
    units. Without this the predictions look ~0 against FEA truth ~4.
    """
    surr = PFNSurrogate.from_loaded_with_real_Y(
        loaded,
        torch.from_numpy(X_ctx_np.astype(np.float32)),
        torch.from_numpy(Y_ctx_np.astype(np.float32)).unsqueeze(-1),
    )
    with torch.no_grad():
        post = surr.posterior(torch.from_numpy(X_test_np.astype(np.float32)).unsqueeze(0))
        mean_norm = post.mean.squeeze().cpu().numpy()
    return mean_norm * surr.y_std + surr.y_mean


def _metrics(y_true, y_pred) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    rho, p = spearmanr(y_true, y_pred)
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "spearman_rho": float(rho), "spearman_p": float(p), "r2": r2}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", default="OneLambda")
    ap.add_argument("--checkpoint", default="checkpoints/OneLambda_pfn.pt")
    ap.add_argument("--emulator", default="emulators/OneLambda_fea_emulator.joblib")
    ap.add_argument("--n-test", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=Path("sweeps"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # === Load data + PFN ===
    loaded = load_checkpoint(args.checkpoint)
    fea = load_fea_designs(args.generator)
    print(f"FEA OneLambda: n={len(fea.X)}, D={fea.X.shape[1]}, T_mean range=[{fea.T_mean.min():.3f}, {fea.T_mean.max():.3f}]")

    # === Fixed test split ===
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(fea.X))
    test_idx = idx[: args.n_test]
    train_idx_pool = idx[args.n_test :]

    X_test = fea.X[test_idx]
    T_test = fea.T_mean[test_idx]
    X_pool = fea.X[train_idx_pool]
    T_pool = fea.T_mean[train_idx_pool]

    print(f"\nsplit: train pool n={len(X_pool)}, held-out test n={len(X_test)}")

    # === Diagnostic 1: PFN(ctx=64) vs GBM at varying n_train ===
    rows = []

    # 1a. PFN at the highest n_ctx the model was trained with (64).
    pfn_ctx_n = 64
    ctx_idx = rng.choice(len(X_pool), pfn_ctx_n, replace=False)
    pfn_pred = _pfn_predict(loaded, X_pool[ctx_idx], T_pool[ctx_idx], X_test)
    pfn_metrics = _metrics(T_test, pfn_pred)
    rows.append({"model": "PFN (matched lumped prior)", "n_train": pfn_ctx_n, **pfn_metrics})
    print(f"\nPFN(ctx={pfn_ctx_n}): RMSE={pfn_metrics['rmse']:.4f}  ρ={pfn_metrics['spearman_rho']:+.3f}  R²={pfn_metrics['r2']:+.3f}")

    # 1b. GBM at increasing n_train, predicting on the same X_test.
    n_trains = [16, 32, 64, 128, 256, 500, len(X_pool)]
    for n in n_trains:
        n = min(n, len(X_pool))
        gbm_idx = rng.choice(len(X_pool), n, replace=False)
        gbm = GradientBoostingRegressor(random_state=args.seed, n_estimators=300, max_depth=4)
        gbm.fit(X_pool[gbm_idx], T_pool[gbm_idx])
        gbm_pred = gbm.predict(X_test)
        m = _metrics(T_test, gbm_pred)
        rows.append({"model": "GBM (FEA-trained, non-parametric)", "n_train": n, **m})
        print(f"GBM(n={n:>4}): RMSE={m['rmse']:.4f}  ρ={m['spearman_rho']:+.3f}  R²={m['r2']:+.3f}")

    # === Diagnostic 3: PFN saturation with context (16, 32, 50, 64) ===
    print("\n--- PFN saturation with context ---")
    for n_ctx in [16, 32, 50, 64]:
        ctx_idx = rng.choice(len(X_pool), n_ctx, replace=False)
        pfn_pred_n = _pfn_predict(loaded, X_pool[ctx_idx], T_pool[ctx_idx], X_test)
        m = _metrics(T_test, pfn_pred_n)
        rows.append({"model": f"PFN(ctx={n_ctx})", "n_train": n_ctx, **m})
        print(f"PFN(ctx={n_ctx:>3}): RMSE={m['rmse']:.4f}  ρ={m['spearman_rho']:+.3f}  R²={m['r2']:+.3f}")

    df = pd.DataFrame(rows)
    csv_path = args.out_dir / f"{args.generator}_parametric_bottleneck.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")

    # === Figure: RMSE/ρ curves + residual scatter ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    gbm_rows = df[df["model"].str.startswith("GBM")]
    pfn_main = df[df["model"] == "PFN (matched lumped prior)"].iloc[0]
    pfn_sat = df[df["model"].str.startswith("PFN(ctx=")]

    # Panel 1: RMSE vs n_train (GBM curve + PFN horizontal)
    axes[0].loglog(gbm_rows["n_train"], gbm_rows["rmse"], "o-", label="GBM (FEA-trained)", color="#1f78b4")
    axes[0].axhline(pfn_main["rmse"], color="#e31a1c", ls="--",
                    label=f"PFN(ctx={pfn_ctx_n}) RMSE={pfn_main['rmse']:.3f}")
    axes[0].set_xlabel("n training FEA designs (log)")
    axes[0].set_ylabel("RMSE on held-out FEA (log)")
    axes[0].set_title("(1) Capacity ceiling")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.2)

    # Panel 2: PFN ctx saturation
    axes[1].plot(pfn_sat["n_train"], pfn_sat["rmse"], "s-", color="#e31a1c", label="PFN RMSE")
    axes[1].set_xlabel("PFN context size")
    axes[1].set_ylabel("RMSE on held-out FEA")
    axes[1].set_title("(3) PFN saturation with context")
    axes[1].grid(True, alpha=0.2)

    # Panel 3: residual scatter at n_ctx=64
    axes[2].scatter(T_test, pfn_pred, s=14, alpha=0.5, color="#1f78b4")
    lo, hi = float(min(T_test.min(), pfn_pred.min())), float(max(T_test.max(), pfn_pred.max()))
    axes[2].plot([lo, hi], [lo, hi], "k--", lw=0.7, label="y = x")
    axes[2].set_xlabel("FEA T_mean (truth)")
    axes[2].set_ylabel("PFN mean prediction")
    axes[2].set_title(f"(2) PFN residuals  (ρ={pfn_main['spearman_rho']:+.3f}, R²={pfn_main['r2']:+.3f})")
    axes[2].grid(True, alpha=0.2)
    axes[2].legend(fontsize=8)

    fig.suptitle(f"{args.generator}: parametric-bottleneck diagnostics", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    pdf_path = args.out_dir / f"{args.generator}_parametric_bottleneck.pdf"
    png_path = args.out_dir / f"{args.generator}_parametric_bottleneck.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    print(f"Wrote {pdf_path} and {png_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
