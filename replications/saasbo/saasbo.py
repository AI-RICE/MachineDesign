"""SAASBO surrogate (Eriksson & Jankowiak, "High-Dimensional Bayesian
Optimization with Sparse Axis-Aligned Subspaces", UAI 2021, arXiv:2103.00349).

The Sparse Axis-Aligned Subspace prior puts a hierarchical half-Cauchy shrinkage
prior on the GP's per-dimension inverse squared lengthscales: a global parameter
pulls every dimension toward "inactive" (very long lengthscale) and the data must
fight to switch a dimension on. Fully-Bayesian inference (NUTS/HMC) over those
lengthscales then reveals which ORIGINAL axes the objective depends on.

We use BoTorch's shipped implementation (`SaasFullyBayesianSingleTaskGP` +
`fit_fully_bayesian_model_nuts`) and add:
  * `build_saasbo`  -- construct + NUTS-fit on (X in [0,1]^D, y),
  * `active_dimensions` -- read the sparse active set off the median lengthscales,
which is the diagnostic that makes SAASBO useful as an effective-dimension probe.
"""

from __future__ import annotations

import numpy as np
import torch
from botorch.fit import fit_fully_bayesian_model_nuts
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize


def build_saasbo(
    X: torch.Tensor,
    y: torch.Tensor,
    warmup: int = 256,
    num_samples: int = 256,
    thinning: int = 16,
    seed: int | None = None,
) -> SaasFullyBayesianSingleTaskGP:
    """Inputs X assumed in [0,1]^D; y shape (n,1). NUTS-fit fully-Bayesian SAAS GP."""
    d = X.shape[-1]
    model = SaasFullyBayesianSingleTaskGP(
        X, y, input_transform=Normalize(d=d), outcome_transform=Standardize(m=1)
    )
    fit_fully_bayesian_model_nuts(
        model, warmup_steps=warmup, num_samples=num_samples, thinning=thinning,
        disable_progbar=True,
    )
    return model


def median_lengthscales(model: SaasFullyBayesianSingleTaskGP) -> np.ndarray:
    """Per-dimension median lengthscale over the MCMC samples (normalised input)."""
    return model.median_lengthscale.detach().cpu().numpy().reshape(-1)


def active_dimensions(model: SaasFullyBayesianSingleTaskGP, cutoff: float = 10.0):
    """Active axes = those with median lengthscale below `cutoff` (normalised
    [0,1] input units). Returns (active_idx_sorted_by_lengthscale, lengthscales).

    `cutoff` ~ O(1-10) separates 'active' (short ls, objective varies along it)
    from 'inactive' (ls >> 1, near-flat). Also returns the full vector so callers
    can inspect the spectrum / pick their own threshold.
    """
    ls = median_lengthscales(model)
    order = np.argsort(ls)
    active = order[ls[order] < cutoff]
    return active, ls
