"""On-the-fly GP-prior task sampler — drop-in replacement for `PriorSampler`.

Used as the negative control / sanity-check for the matched-prior PFN
hypothesis: per CLAUDE.md §11 and §12.5.P4, training a PFN on samples
from a Gaussian-process prior (no physics at all) tests whether the
PFN approximates the underlying Bayesian-optimal predictor well enough
that we can trust the meta-training. By the PFNs4BO theorem (Nagler 2023)
a fully-trained GP-prior PFN should reproduce a GP's posterior predictive;
the sanity check is exactly to confirm this in our pipeline.

PFNs4BO-flavoured recipe per task:
  - random Matern kernel (ν ∈ {1.5, 2.5})
  - ARD: independent length scale per dimension, drawn log-normal
  - output scale drawn log-normal
  - observation noise drawn log-uniform
  - x ~ Uniform([lo, hi]^D)   where (lo, hi) are the target generator's bounds
  - y = Cholesky(K) @ z   then z-scored per task

The interface mirrors `PriorSampler.sample / sample_batch`, so
`train.py` can swap one for the other without touching the training loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .prior_sampler import PFNTask


@dataclass(frozen=True)
class GPPriorConfig:
    """Hyperparameter ranges for the GP-prior PFN training distribution.

    Defaults revised 2026-05-24 from PFNs4BO's TabPFN-flavoured original to
    cover what the FEA OneLambda response surface actually looks like, as
    measured by the §6.7 diagnostic on the 200k PFN:

      - FEA-direct GP picks length scales up to ~14 (very smooth in some
        dims). Original log_ls_std=0.7 capped max ls at ~4; revised
        log_ls_std=1.4 covers up to ~16.
      - FEA-direct GP picks noise ~4e-3 (essentially deterministic).
        Original log_noise_min=-4 had a floor of ~0.018; revised
        log_noise_min=-10 covers down to ~4e-5.
      - FEA torque is smooth, ν=2.5 matches better than mixing in ν=1.5.

    Crucially, the revision only WIDENS the prior — no FEA hyperparameters
    are used to set centers. The §6.7 FEA-direct GP fit only served as a
    measurement target, not as a training-prior input (preserves §11
    hygiene per CLAUDE.md).
    """
    log_ls_mean: float = 0.0
    log_ls_std: float = 1.4              # was 0.7; allows ls ∈ [~0.06, ~16]
    log_outputscale_mean: float = 0.0
    log_outputscale_std: float = 0.7
    log_noise_min: float = -10.0         # was -4.0; allows near-deterministic
    log_noise_max: float = -2.0          # was -1.0
    nu_choices: tuple = (2.5,)           # was (1.5, 2.5); FEA smooth


class GPPriorSampler:
    """Sample one independent GP function per task."""

    def __init__(
        self,
        input_dim: int,
        bounds: np.ndarray,
        cfg: GPPriorConfig | None = None,
    ) -> None:
        if bounds.shape != (2, input_dim):
            raise ValueError(f"bounds must be (2, D={input_dim}), got {bounds.shape}")
        self.input_dim = int(input_dim)
        self.bounds = bounds.astype(np.float64)
        self.cfg = cfg or GPPriorConfig()
        self.granularity_mode = "GP"

    # Mirror PriorSampler property for train.py compatibility.
    @property
    def library(self):
        raise AttributeError("GPPriorSampler is library-free; no .library attribute")

    def _matern(self, X: np.ndarray, ell: np.ndarray, outputscale: float, nu: float) -> np.ndarray:
        """Matern kernel with per-dim length scales `ell` on X (shape (N, D))."""
        X_scaled = X / ell[None, :]
        diff = X_scaled[:, None, :] - X_scaled[None, :, :]
        d = np.sqrt(np.maximum((diff ** 2).sum(-1), 0.0))
        if nu == 1.5:
            s = np.sqrt(3.0) * d
            return outputscale * (1.0 + s) * np.exp(-s)
        if nu == 2.5:
            s = np.sqrt(5.0) * d
            return outputscale * (1.0 + s + (s ** 2) / 3.0) * np.exp(-s)
        raise ValueError(f"nu must be 1.5 or 2.5, got {nu}")

    def _stable_chol(self, K: np.ndarray) -> np.ndarray:
        """Cholesky with adaptive jitter (covariance matrices get nearly-singular at
        short length scales / closely-spaced sample points)."""
        N = K.shape[0]
        jitter = 1e-6 * float(np.trace(K)) / N + 1e-8
        for _ in range(8):
            try:
                return np.linalg.cholesky(K + jitter * np.eye(N))
            except np.linalg.LinAlgError:
                jitter *= 10.0
        return np.linalg.cholesky(K + 1e-2 * np.eye(N))

    def sample(
        self,
        rng: np.random.Generator,
        n_context: int = 32,
        n_target: int = 1,
        normalise: bool = True,
    ) -> PFNTask:
        N = n_context + n_target
        # 1. sample hyperparameters
        log_ls = rng.normal(self.cfg.log_ls_mean, self.cfg.log_ls_std, size=self.input_dim)
        ell = np.exp(log_ls)
        outputscale = float(np.exp(rng.normal(self.cfg.log_outputscale_mean, self.cfg.log_outputscale_std)))
        noise = float(np.exp(rng.uniform(self.cfg.log_noise_min, self.cfg.log_noise_max)))
        nu = float(self.cfg.nu_choices[int(rng.integers(0, len(self.cfg.nu_choices)))])

        # 2. sample N inputs uniformly in [0, 1]^D
        X_unit = rng.uniform(0.0, 1.0, size=(N, self.input_dim))

        # 3. build covariance + sample y
        K = self._matern(X_unit, ell=ell, outputscale=outputscale, nu=nu)
        K = K + noise * np.eye(N)
        L = self._stable_chol(K)
        z = rng.standard_normal(N)
        y = L @ z  # (N,)

        # 4. map X_unit back to the target generator's raw bounds — the rest of
        #    the pipeline expects raw-units X (per-dim x normalisation in
        #    train.py rescales it for the encoder).
        lo, hi = self.bounds
        X_raw = lo + X_unit * (hi - lo)

        # 5. per-task y normalisation using CONTEXT-ONLY statistics.
        #    Normalising over context+target leaks the target when n_target is
        #    small: z-scores sum to zero over the normalisation set, so with
        #    n_target=1 the target equals -sum(context z) regardless of its x.
        #    The model then learns that shortcut and predicts a near-constant at
        #    inference (where we normalise by context only). Using context-only
        #    stats here removes the leak and matches the inference convention
        #    (PFNSurrogate.from_loaded_with_real_Y z-scores by context).
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
