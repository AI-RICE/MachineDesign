"""BoTorch-compatible single-output PFN surrogate (M4).

Wraps a trained `PFNBoModel` plus a context buffer of `(X_train, Y_train)`
into a `botorch.models.model.Model`. The PFN's binned (Riemann)
predictive distribution is moment-matched to a Gaussian via the
BarDistribution's `mean` / `variance` methods, then handed to BoTorch
acquisition functions as a `GPyTorchPosterior` over a `MultivariateNormal`.

For α (marginal × marginal) on a single output, this is sufficient.
For the two-output case in the future, wrap two `PFNSurrogate` instances
in a `botorch.models.model_list_gp_regression.ModelListGP`-style list
(or use a custom list class that supports non-GPyTorch posteriors).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from botorch.models.model import Model
from botorch.posteriors import Posterior
from botorch.sampling import IIDNormalSampler
from botorch.sampling.get_sampler import GetSampler

from .checkpoint import LoadedPFN
from .model import PFNBoModel


class PFNGaussianPosterior(Posterior):
    """Minimal single-output Gaussian posterior backed by PFN moments.

    Shapes follow BoTorch's multi-output convention `(..., q, 1)` so the
    standard single-output acquisition functions (`qLogExpectedImprovement`,
    `qExpectedImprovement`, …) work without modification.
    """

    def __init__(self, mean: torch.Tensor, variance: torch.Tensor) -> None:
        if mean.shape != variance.shape:
            raise ValueError(f"mean / variance shape mismatch: {mean.shape} vs {variance.shape}")
        if mean.shape[-1] != 1:
            raise ValueError(f"single-output PFN posterior expects trailing dim 1, got {mean.shape}")
        self._mean = mean
        self._variance = variance

    @property
    def device(self) -> torch.device:
        return self._mean.device

    @property
    def dtype(self) -> torch.dtype:
        return self._mean.dtype

    @property
    def mean(self) -> torch.Tensor:
        return self._mean

    @property
    def variance(self) -> torch.Tensor:
        return self._variance

    @property
    def batch_shape(self) -> torch.Size:
        return self._mean.shape[:-2]

    @property
    def base_sample_shape(self) -> torch.Size:
        return self._mean.shape

    @property
    def event_shape(self) -> torch.Size:
        return self._mean.shape[-2:]

    @property
    def batch_range(self) -> tuple[int, int]:
        # All dims except the last two (q, m) are batch dims.
        return 0, -2

    @property
    def _extended_shape(self) -> torch.Size:
        return self._mean.shape

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        std = self._variance.clamp_min(1e-12).sqrt()
        eps = torch.randn(
            sample_shape + self._mean.shape, device=self._mean.device, dtype=self._mean.dtype
        )
        return self._mean + std * eps

    def rsample_from_base_samples(
        self, sample_shape: torch.Size, base_samples: torch.Tensor
    ) -> torch.Tensor:
        std = self._variance.clamp_min(1e-12).sqrt()
        return self._mean + std * base_samples


# Tell BoTorch how to construct an MCSampler for our Gaussian posterior.
# IID normal sampler suffices because the per-point variances are independent
# (diagonal covariance) — the same assumption holds for SingleTaskGP's posterior
# at acquisition-optim time and BoTorch handles that with IIDNormalSampler too.
@GetSampler.register(PFNGaussianPosterior)
def _get_pfn_sampler(posterior, sample_shape, seed=None):  # noqa: ARG001
    return IIDNormalSampler(sample_shape=sample_shape, seed=seed)


class PFNSurrogate(Model):
    """Single-output PFN surrogate, BoTorch-compatible.

    The model is *stateless* in the GPyTorch sense — no learnable
    parameters from BoTorch's perspective — but it carries a context
    buffer `(train_X, train_Y)` that the underlying transformer reads
    in-context on every forward pass.

    Parameters
    ----------
    pfn : PFNBoModel
        A trained PFN.
    train_X : Tensor
        `(n_ctx, D)`. Inputs in PFN-space (raw library params).
    train_Y : Tensor
        `(n_ctx, 1)`. Targets in **normalised** PFN-space (z-scored). The
        caller is responsible for normalising real T values before passing
        them in (use `y_mean` / `y_std` properties to recover the
        scaling).
    y_mean, y_std : float
        Training-time z-score statistics. Stored as properties so callers
        can de-normalise posteriors and BO loops can normalise `best_f`
        consistently. **Note**: the surrogate's posterior returns mean and
        variance in normalised PFN-space, NOT real units. This keeps the
        numerical scale O(1) and avoids float32 gradient overflow in
        acquisition optimisation (real `y_std` can be O(1e12)).
    device : torch.device
    """

    _num_outputs = 1

    def __init__(
        self,
        pfn: PFNBoModel,
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
        y_mean: float,
        y_std: float,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        if train_Y.ndim != 2 or train_Y.shape[1] != 1:
            raise ValueError(f"train_Y must be (n, 1), got {tuple(train_Y.shape)}")
        if train_X.shape[0] != train_Y.shape[0]:
            raise ValueError(f"train_X / train_Y row count mismatch: "
                             f"{train_X.shape[0]} vs {train_Y.shape[0]}")
        self.pfn = pfn
        self.device = device or next(pfn.parameters()).device
        self.train_X = train_X.to(self.device, dtype=torch.float32)
        # train_Y is expected pre-normalised (z-scored). Stored as-is.
        self.train_Y = train_Y.to(self.device, dtype=torch.float32)
        self.y_mean = float(y_mean)
        self.y_std = float(y_std)

    @classmethod
    def from_loaded_with_real_Y(
        cls,
        loaded: LoadedPFN,
        train_X: torch.Tensor,
        train_Y_real: torch.Tensor,
    ) -> "PFNSurrogate":
        """Construct a surrogate from real-unit Y values; normalises internally."""
        train_Y_norm = (train_Y_real - loaded.y_mean) / loaded.y_std
        return cls(
            pfn=loaded.model,
            train_X=train_X,
            train_Y=train_Y_norm,
            y_mean=loaded.y_mean,
            y_std=loaded.y_std,
            device=loaded.device,
        )

    def denormalise_mean(self, mean_norm: torch.Tensor) -> torch.Tensor:
        return mean_norm * self.y_std + self.y_mean

    def denormalise_variance(self, var_norm: torch.Tensor) -> torch.Tensor:
        return var_norm * (self.y_std ** 2)

    @classmethod
    def from_loaded(cls, loaded: LoadedPFN, train_X: torch.Tensor, train_Y_real: torch.Tensor) -> "PFNSurrogate":
        """Construct from real-unit Y values (most common case)."""
        return cls.from_loaded_with_real_Y(loaded, train_X, train_Y_real)

    @property
    def num_outputs(self) -> int:
        return self._num_outputs

    def condition_on_observations(self, X: torch.Tensor, Y: torch.Tensor, **kwargs) -> "PFNSurrogate":
        """Return a new surrogate with `(X, Y)` appended to the context."""
        new_X = torch.cat([self.train_X, X.to(self.train_X)], dim=0)
        new_Y = torch.cat([self.train_Y, Y.to(self.train_Y)], dim=0)
        return PFNSurrogate(
            pfn=self.pfn,
            train_X=new_X,
            train_Y=new_Y,
            y_mean=self.y_mean,
            y_std=self.y_std,
            device=self.device,
        )

    # ------------------------------------------------------------------
    # BoTorch Model API
    # ------------------------------------------------------------------
    def posterior(
        self,
        X: torch.Tensor,
        output_indices: Optional[list[int]] = None,
        observation_noise: bool = False,
        posterior_transform=None,
    ) -> GPyTorchPosterior:
        """Return a Gaussian posterior at test points X.

        `X` has shape `(..., q, D)`; we treat the last two dims as
        `(q, D)` and broadcast across any leading batch dims.
        """
        if X.ndim < 2:
            raise ValueError(f"X must have ≥ 2 dims, got {X.ndim}")
        # Flatten leading batch dims so we can run the PFN once.
        orig_shape = X.shape
        D = orig_shape[-1]
        q = orig_shape[-2]
        batch_shape = orig_shape[:-2]
        X_flat = X.reshape(-1, q, D).to(self.device, dtype=torch.float32)  # (B, q, D)
        B = X_flat.shape[0]

        # The PFN's TableTransformer expects `(n_seq, batch, D)` for x and
        # `(n_ctx, batch)` for y. We need to build, per batch element, a
        # context = self.train_{X,Y} (shared across the B query batches)
        # plus the query points X_flat[b].
        n_ctx = self.train_X.shape[0]
        n_seq = n_ctx + q
        x_seq = torch.empty((n_seq, B, D), device=self.device, dtype=torch.float32)
        y_ctx = torch.empty((n_ctx, B), device=self.device, dtype=torch.float32)
        x_seq[:n_ctx] = self.train_X.unsqueeze(1).expand(n_ctx, B, D)
        x_seq[n_ctx:] = X_flat.transpose(0, 1)        # (q, B, D)
        # train_Y is already normalised; broadcast across query batches.
        y_ctx[:] = self.train_Y.squeeze(-1).unsqueeze(-1).expand(n_ctx, B)

        # Don't wrap in `torch.no_grad`: BoTorch's acquisition-function optimiser
        # needs gradients to flow back from `posterior.mean` to the query X.
        # The PFN parameters won't receive optimiser steps; they're frozen by the
        # BO loop's choice to only optimise X.
        logits = self.pfn((None, x_seq, y_ctx), single_eval_pos=n_ctx)  # (n_seq or q, B, num_bins)
        # Extract the query-segment logits.
        if logits.shape[0] == n_seq:
            logits_q = logits[n_ctx:]
        else:
            # Most TableTransformer configs return only the query logits.
            logits_q = logits[-q:]
        logits_q = logits_q.reshape(B * q, -1)
        crit = self.pfn.criterion.to(self.device)

        mean_norm = crit.mean(logits_q).reshape(B, q)               # (B, q) z-scored
        var_norm = crit.variance(logits_q).reshape(B, q)            # (B, q)

        # Return in normalised PFN-space. Caller de-normalises via
        # `surr.denormalise_{mean,variance}` if real units are needed.
        mean = mean_norm.reshape(*batch_shape, q, 1)
        variance = var_norm.reshape(*batch_shape, q, 1)
        return PFNGaussianPosterior(mean=mean, variance=variance)
