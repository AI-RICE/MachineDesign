"""GBM-based FEA emulator: one regressor per output (T_mean, T_ripple).

Uses scikit-learn's `HistGradientBoostingRegressor` so there's no extra
dependency to install. Each parameterisation gets its own emulator
(different input dim D ∈ {7, 12, 13}).

K-fold CV diagnostics report RMSE separately on the uniform-initials
subset and the BO-trace subset of held-out folds, per CLAUDE.md §6.5
("BO-trace RMSE is expected to be lower; document it openly").
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold


def _make_regressor(n_iter: int = 500) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=n_iter,
        learning_rate=0.05,
        max_depth=None,            # let it grow shallow trees adaptively
        min_samples_leaf=10,
        l2_regularization=1.0,
        random_state=0,
    )


@dataclass
class FEAEmulator:
    model_T: Any = None         # HistGradientBoostingRegressor for T_mean
    model_R: Any = None         # HistGradientBoostingRegressor for T_ripple
    n_features: int = 0
    generator_short: str = ""

    def fit(self, X: np.ndarray, T_mean: np.ndarray, T_ripple: np.ndarray) -> "FEAEmulator":
        self.model_T = _make_regressor()
        self.model_T.fit(X, T_mean)
        self.model_R = _make_regressor()
        self.model_R.fit(X, T_ripple)
        self.n_features = X.shape[1]
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (T_mean, T_ripple) predictions. No NaN-mimicry here;
        callers should use the BarrierGenerator's `feasible_barriers` test
        before calling this for an honest infeasible→NaN path.
        """
        return self.model_T.predict(X), self.model_R.predict(X)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model_T": self.model_T,
                "model_R": self.model_R,
                "n_features": self.n_features,
                "generator_short": self.generator_short,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "FEAEmulator":
        d = joblib.load(path)
        return cls(
            model_T=d["model_T"],
            model_R=d["model_R"],
            n_features=int(d["n_features"]),
            generator_short=str(d.get("generator_short", "")),
        )


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def cv_evaluate(
    X: np.ndarray,
    T_mean: np.ndarray,
    T_ripple: np.ndarray,
    is_uniform: np.ndarray,
    k: int = 5,
    seed: int = 0,
) -> dict[str, float]:
    """K-fold CV with separate RMSEs on uniform-initials vs BO-trace test points.

    Each fold trains one `FEAEmulator` on the (k-1)/k training subset (which
    mixes uniform and BO points) and predicts on the remaining 1/k. The
    held-out predictions are split by `is_uniform` and RMSE aggregated.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    uni_T, bo_T, uni_R, bo_R = [], [], [], []
    for tr, te in kf.split(X):
        m = FEAEmulator().fit(X[tr], T_mean[tr], T_ripple[tr])
        T_pred, R_pred = m.predict(X[te])
        u_mask = is_uniform[te]
        if u_mask.any():
            uni_T.append(_rmse(T_mean[te][u_mask], T_pred[u_mask]))
            uni_R.append(_rmse(T_ripple[te][u_mask], R_pred[u_mask]))
        if (~u_mask).any():
            bo_T.append(_rmse(T_mean[te][~u_mask], T_pred[~u_mask]))
            bo_R.append(_rmse(T_ripple[te][~u_mask], R_pred[~u_mask]))

    return {
        "n_uniform_total": int(is_uniform.sum()),
        "n_bo_total": int((~is_uniform).sum()),
        "rmse_T_uniform_mean": float(np.mean(uni_T)),
        "rmse_T_bo_mean": float(np.mean(bo_T)),
        "rmse_R_uniform_mean": float(np.mean(uni_R)),
        "rmse_R_bo_mean": float(np.mean(bo_R)),
        "rmse_T_uniform_std": float(np.std(uni_T)),
        "rmse_T_bo_std": float(np.std(bo_T)),
        "rmse_R_uniform_std": float(np.std(uni_R)),
        "rmse_R_bo_std": float(np.std(bo_R)),
    }
