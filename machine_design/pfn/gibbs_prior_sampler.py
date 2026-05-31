"""Gibbs (anisotropic Paciorek--Schervish) non-stationary kernel prior sampler.

Drop-in replacement for `GPPriorSampler` exposing the same `sample` /
`sample_batch` interface. Per CLAUDE.md §13.4 + `PLAN_gibbs_prior.md`:
tests whether a non-stationary kernel prior (length scale varies with x)
beats the stationary wide-Matern-ARD prior on FEA-OOD BO.

Per training task:
  1. Per-dim length-scale field   log ell_d(x_d) = a_d + b_d * x_tilde_d,
     x_tilde_d = 2 (x_d - mid_d) / range_d   in   [-1, 1].
     b_d = 0 for all d recovers the stationary case (exactly the wide GP
     prior C2 was trained on).
  2. outputscale, noise  --  same draws as `GPPriorConfig`.
  3. Kernel: anisotropic Paciorek--Schervish, Matern-5/2 flavour:
        k(x, x') = sigma_f^2 * N(x, x') * m_{5/2}(d_eff(x, x')),
     where
        N(x, x') = prod_d sqrt(2 ell_d(x) ell_d(x') / (ell_d^2(x) + ell_d^2(x')))
        d_eff^2(x, x') = sum_d 2 (x_d - x'_d)^2 / (ell_d^2(x) + ell_d^2(x'))
        m_{5/2}(r)  = (1 + sqrt(5) r + 5 r^2 / 3) exp(-sqrt(5) r).
  4. y = Cholesky(K + sigma_n^2 I) @ z,  z ~ N(0, I).
  5. CONTEXT-ONLY per-task y normalisation (the leak-fix invariant; see
     `gp_prior_sampler.py` for the explanation of why context+target is wrong).

Numerical care:
  - Normalising factor N(x, x') is in (0, 1] by AM--GM on ell_d^2(x), ell_d^2(x').
    Computed via 0.5 * sum_d log(2 ell ell' / (ell^2 + ell'^2)) and exponentiated
    once; clipped at 0 to defeat float noise.
  - d_eff is sqrt of a clamped >= 0 sum.
  - Same adaptive-jitter Cholesky loop as `gp_prior_sampler.py`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .prior_sampler import PFNTask


_SQRT5 = math.sqrt(5.0)


@dataclass(frozen=True)
class GibbsPriorConfig:
    """Hyperparameter ranges for the Gibbs (non-stationary Matern-5/2) PFN training prior.

    log_a_std mirrors `GPPriorConfig.log_ls_std` (overall length-scale spread).
    log_b_std controls how strongly the per-dim length scale varies across the
    unit input range (b_d = 0 ⇒ stationary). Increasing log_b_std widens the
    non-stationarity; we sweep this knob in the experiments.
    """
    log_a_std: float = 1.4              # a_d ~ Normal(0, log_a_std^2); matches GPPriorConfig.log_ls_std
    log_b_std: float = 1.0              # b_d ~ Normal(0, log_b_std^2); 0 ⇒ pure stationary recovery
    log_outputscale_mean: float = 0.0
    log_outputscale_std: float = 0.7
    log_noise_min: float = -10.0
    log_noise_max: float = -2.0
    nu: float = 2.5                     # Matern-5/2 only


# ---------- Pure-NumPy kernel pieces (also used by tests) ----------

def log_ell_field(a: np.ndarray, b: np.ndarray, x_unit: np.ndarray) -> np.ndarray:
    """log length-scale field at each point.

    a, b : (D,) per-dim coefficients.
    x_unit : (N, D) inputs in [-1, 1].
    returns log_ell : (N, D).
    """
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape:
        raise ValueError(f"a, b must be 1-D and same shape; got {a.shape}, {b.shape}")
    if x_unit.ndim != 2 or x_unit.shape[1] != a.shape[0]:
        raise ValueError(f"x_unit must be (N, D={a.shape[0]}); got {x_unit.shape}")
    return a[None, :] + b[None, :] * x_unit


def paciorek_schervish_matern52(
    X: np.ndarray,
    X_prime: np.ndarray,
    log_ell_X: np.ndarray,
    log_ell_Xprime: np.ndarray,
    outputscale: float,
) -> np.ndarray:
    """Anisotropic Paciorek--Schervish kernel, Matern-5/2 flavour.

    X, X_prime           : (N, D), (N', D)
    log_ell_X, log_ell_Xprime : (N, D), (N', D)
    outputscale          : scalar sigma_f^2
    returns K : (N, N')
    """
    if X.ndim != 2 or X_prime.ndim != 2 or X.shape[1] != X_prime.shape[1]:
        raise ValueError(f"X, X' must be 2-D and same D; got {X.shape}, {X_prime.shape}")
    if log_ell_X.shape != X.shape or log_ell_Xprime.shape != X_prime.shape:
        raise ValueError("log_ell shapes must match X / X' shapes")

    # ell^2 broadcasted to (N, N', D).
    ell_sq = np.exp(2.0 * log_ell_X)[:, None, :]              # (N, 1, D)
    ell_sq_p = np.exp(2.0 * log_ell_Xprime)[None, :, :]       # (1, N', D)
    sum_ell_sq = ell_sq + ell_sq_p                            # (N, N', D)

    # Normalising factor N(x, x') = prod_d sqrt(2 ell ell' / (ell^2 + ell'^2)).
    # Compute in log-space for stability with extreme length-scale ratios.
    # log(2 ell ell') = log 2 + log_ell + log_ell'
    log_num = math.log(2.0) + log_ell_X[:, None, :] + log_ell_Xprime[None, :, :]   # (N, N', D)
    log_den = np.log(sum_ell_sq)                                                    # (N, N', D)
    log_norm = 0.5 * (log_num - log_den).sum(axis=-1)                              # (N, N')
    norm = np.exp(log_norm)

    # Effective-distance squared: sum_d 2 (x - x')^2 / (ell^2 + ell'^2).
    diff = X[:, None, :] - X_prime[None, :, :]                # (N, N', D)
    d_eff_sq_terms = 2.0 * diff * diff / sum_ell_sq           # (N, N', D)
    d_eff_sq = np.maximum(d_eff_sq_terms.sum(axis=-1), 0.0)   # (N, N')
    d_eff = np.sqrt(d_eff_sq)

    matern = (1.0 + _SQRT5 * d_eff + (5.0 / 3.0) * d_eff_sq) * np.exp(-_SQRT5 * d_eff)
    return outputscale * norm * matern


def _stable_chol(K: np.ndarray) -> np.ndarray:
    """Cholesky with adaptive jitter (same recipe as gp_prior_sampler)."""
    N = K.shape[0]
    jitter = 1e-6 * float(np.trace(K)) / N + 1e-8
    for _ in range(8):
        try:
            return np.linalg.cholesky(K + jitter * np.eye(N))
        except np.linalg.LinAlgError:
            jitter *= 10.0
    return np.linalg.cholesky(K + 1e-2 * np.eye(N))


# ---------- The sampler class ----------

class GibbsPriorSampler:
    """Sample one independent Gibbs-kernel function per task.

    Matches `GPPriorSampler` API: `sample(rng, n_context, n_target, normalise)`
    returns a `PFNTask` in raw input units. Internally the kernel is built in
    unit-input space x_tilde in [-1, 1]^D for numerical convenience.
    """

    def __init__(
        self,
        input_dim: int,
        bounds: np.ndarray,
        cfg: GibbsPriorConfig | None = None,
    ) -> None:
        if bounds.shape != (2, input_dim):
            raise ValueError(f"bounds must be (2, D={input_dim}), got {bounds.shape}")
        self.input_dim = int(input_dim)
        self.bounds = bounds.astype(np.float64)
        self.cfg = cfg or GibbsPriorConfig()
        self.granularity_mode = "Gibbs"

    @property
    def library(self):
        raise AttributeError("GibbsPriorSampler is library-free; no .library attribute")

    def sample(
        self,
        rng: np.random.Generator,
        n_context: int = 32,
        n_target: int = 1,
        normalise: bool = True,
    ) -> PFNTask:
        N = n_context + n_target
        D = self.input_dim
        cfg = self.cfg

        # 1. Hyperparameter draws.
        a = rng.normal(0.0, cfg.log_a_std, size=D)
        b = rng.normal(0.0, cfg.log_b_std, size=D)
        outputscale = float(np.exp(rng.normal(cfg.log_outputscale_mean, cfg.log_outputscale_std)))
        noise = float(np.exp(rng.uniform(cfg.log_noise_min, cfg.log_noise_max)))

        # 2. Sample N inputs uniformly in [0, 1]^D and map to [-1, 1]^D for kernel.
        X_unit01 = rng.uniform(0.0, 1.0, size=(N, D))
        X_unit = 2.0 * X_unit01 - 1.0

        # 3. Length-scale field, kernel, draw y.
        log_ell = log_ell_field(a, b, X_unit)
        K = paciorek_schervish_matern52(X_unit, X_unit, log_ell, log_ell, outputscale)
        K = K + noise * np.eye(N)
        L = _stable_chol(K)
        z = rng.standard_normal(N)
        y = L @ z

        # 4. Map X to raw input units (the rest of the pipeline expects raw X).
        lo, hi = self.bounds
        X_raw = lo + X_unit01 * (hi - lo)

        # 5. Per-task y normalisation using CONTEXT-ONLY statistics (leak-fix invariant).
        if normalise:
            yc = y[:n_context]
            ymean = float(np.mean(yc))
            ystd = float(np.std(yc) + 1e-12)
            y = (y - ymean) / ystd

        return PFNTask(
            x_context=X_raw[:n_context],
            y_context=y[:n_context],
            x_target=X_raw[n_context:],
            y_target=y[n_context:],
            granularity=self.granularity_mode,
        )

    def sample_batch(
        self,
        rng: np.random.Generator,
        batch_size: int = 64,
        n_context: int = 32,
        n_target: int = 1,
        normalise: bool = True,
    ) -> Iterable[PFNTask]:
        return [
            self.sample(rng, n_context=n_context, n_target=n_target, normalise=normalise)
            for _ in range(batch_size)
        ]
