"""Detailed FEA-emulator quality report (M4.5).

Two cross-validation protocols per parameterisation:

1. **5-fold random CV** (already in the build script): shuffle all 1,250
   designs, hold out 1/5, predict, repeat 5×. Stratify RMSE by whether
   the held-out point was a uniform initial or a BO-trace point.

2. **Leave-one-seed-out (LOSO)**: stricter — train on 4 of the 5
   `results*/` seeds, predict on the 5th. Each seed's BO trace explored
   similar high-T regions; random CV mixes those regions across train/test
   so it can look optimistic. LOSO tests whether the emulator generalises
   to *unseen* BO trajectories.

Outputs:
- console table with RMSE, MAE, R² (test only, no peeking)
- `sweeps/fea_emulator_quality.csv` for the paper appendix
- `sweeps/fea_emulator_quality.pdf` — 2 × 3 scatter of predicted vs
  actual for (T_mean, T_ripple) × three parameterisations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

from machine_design.fea_emulator import FEAEmulator, load_fea_designs


GENERATORS = ("OneLambda", "SixLambdas", "ThreeBrokenLines")
DIMS = {"OneLambda": 7, "SixLambdas": 12, "ThreeBrokenLines": 13}


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    rel_rmse = rmse / float(np.std(y_true) + 1e-12)
    return {"rmse": rmse, "mae": mae, "r2": r2, "rel_rmse": rel_rmse}


def kfold_predictions(loaded, k: int = 5, seed: int = 0):
    """Return arrays of held-out predictions for T_mean and T_ripple."""
    X, T, R, uni = loaded.X, loaded.T_mean, loaded.T_ripple, loaded.is_uniform_init
    T_pred = np.full_like(T, np.nan)
    R_pred = np.full_like(R, np.nan)
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    for tr, te in kf.split(X):
        m = FEAEmulator().fit(X[tr], T[tr], R[tr])
        Tp, Rp = m.predict(X[te])
        T_pred[te] = Tp
        R_pred[te] = Rp
    return T_pred, R_pred, uni


def loso_predictions(loaded):
    """Leave-one-seed-out predictions across the 5 `results*/` seeds."""
    X, T, R, sid, uni = (
        loaded.X, loaded.T_mean, loaded.T_ripple, loaded.seed_id, loaded.is_uniform_init,
    )
    T_pred = np.full_like(T, np.nan)
    R_pred = np.full_like(R, np.nan)
    for s in np.unique(sid):
        train_mask = sid != s
        test_mask = sid == s
        if not test_mask.any():
            continue
        m = FEAEmulator().fit(X[train_mask], T[train_mask], R[train_mask])
        Tp, Rp = m.predict(X[test_mask])
        T_pred[test_mask] = Tp
        R_pred[test_mask] = Rp
    return T_pred, R_pred, uni, sid


def _scatter(ax, y_true, y_pred, mask_uniform, title: str, unit: str):
    ax.scatter(y_true[~mask_uniform], y_pred[~mask_uniform],
               s=8, alpha=0.5, label="BO trace", color="#1f78b4")
    ax.scatter(y_true[mask_uniform], y_pred[mask_uniform],
               s=14, alpha=0.7, label="uniform initials", color="#e31a1c")
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], color="black", lw=0.7, ls="--")
    ax.set_xlabel(f"FEA {unit}")
    ax.set_ylabel(f"emulator {unit}")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--out-dir", type=Path, default=Path("sweeps"))
    ap.add_argument("--constrained-only", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    constrained_flag = True if args.constrained_only else None
    rows: list[dict] = []

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for col, gen in enumerate(GENERATORS):
        print(f"\n=== {gen} (D={DIMS[gen]}) ===")
        loaded = load_fea_designs(gen, results_root=args.results_root,
                                  constrained=constrained_flag)
        N = len(loaded.X)
        print(f"  n_total={N}  T_mean range=[{loaded.T_mean.min():.3f}, {loaded.T_mean.max():.3f}]"
              f"  T_ripple range=[{loaded.T_ripple.min():.2f}, {loaded.T_ripple.max():.2f}]")

        for protocol, predict_fn in (("5-fold random CV", "kfold"), ("leave-one-seed-out", "loso")):
            if predict_fn == "kfold":
                Tp, Rp, uni = kfold_predictions(loaded, k=5, seed=0)
            else:
                Tp, Rp, uni, _ = loso_predictions(loaded)

            for name, y_true, y_pred, unit in (
                ("T_mean", loaded.T_mean, Tp, "N·m"),
                ("T_ripple", loaded.T_ripple, Rp, "%"),
            ):
                full = _metrics(y_true, y_pred)
                u = _metrics(y_true[uni], y_pred[uni]) if uni.any() else None
                b = _metrics(y_true[~uni], y_pred[~uni]) if (~uni).any() else None
                row = {
                    "generator": gen, "protocol": protocol, "output": name,
                    "rmse_all": full["rmse"], "mae_all": full["mae"], "r2_all": full["r2"],
                    "rel_rmse_all": full["rel_rmse"],
                    "rmse_uniform": u["rmse"] if u else None,
                    "rmse_bo": b["rmse"] if b else None,
                    "r2_uniform": u["r2"] if u else None,
                    "r2_bo": b["r2"] if b else None,
                }
                rows.append(row)
                tag = "(N·m)" if unit == "N·m" else "(%)"
                print(f"  [{protocol:<22}] {name:<9}: "
                      f"RMSE={full['rmse']:.3f}{tag:<6} "
                      f"R²={full['r2']:.3f}  "
                      f"rel_RMSE={full['rel_rmse']:.3f}  "
                      f"uniform/BO RMSE = {u['rmse']:.3f}/{b['rmse']:.3f}")

        # Scatter for LOSO predictions (the stricter protocol).
        Tp, Rp, uni, _ = loso_predictions(loaded)
        _scatter(axes[0, col], loaded.T_mean, Tp, uni,
                 f"{gen} (D={DIMS[gen]})  —  T_mean", "T_mean (N·m)")
        _scatter(axes[1, col], loaded.T_ripple, Rp, uni,
                 f"{gen} (D={DIMS[gen]})  —  T_ripple", "T_ripple (%)")

    df = pd.DataFrame(rows)
    csv_path = args.out_dir / "fea_emulator_quality.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")

    fig.suptitle("FEA emulator quality (leave-one-seed-out predictions)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    pdf_path = args.out_dir / "fea_emulator_quality.pdf"
    png_path = args.out_dir / "fea_emulator_quality.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    print(f"Wrote {pdf_path}\nWrote {png_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
