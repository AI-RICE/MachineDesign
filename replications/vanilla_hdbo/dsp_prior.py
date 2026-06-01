"""Dimensionality-Scaled lengthscale Prior (DSP) from Hvarfner, Hellsten & Nardi,
"Vanilla Bayesian Optimization Performs Great in High Dimensions", ICML 2024
(arXiv:2402.02229).

Their prescription (paper Eq. 4, §6):
    lengthscale prior   l_i ~ LogNormal( mu0 + log(D)/2 , sigma0 ),
    with mu0 = sqrt(2), sigma0 = sqrt(3)   ->  mode(l) scales as sqrt(D),
    and mode(l) ~= 0.50 at D=6.
    Signal variance is FIXED to sigma_f^2 = 1 (no ScaleKernel).
    Matern-5/2 (or RBF) ARD kernel, constant mean, outputs standardized.

We implement the prior explicitly (so the formula is transparent and unit-
testable) and build two GP factories used by the replication:
  * build_dsp_gp        -- the DSP recipe above,
  * build_default_gp    -- the classic pre-2024 BoTorch default
                           (Matern-5/2 ARD, lengthscale ~ Gamma(3,6),
                            ScaleKernel with outputscale ~ Gamma(2,0.15)),
                           i.e. the "MAP: l ~ Gamma(3,6)" baseline in the paper.

The DSP prior is the *only* difference that the paper claims is needed; the
default-Gamma GP is the control that isolates the mechanism.
"""

from __future__ import annotations

import math

import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.priors import GammaPrior, LogNormalPrior

SQRT2 = math.sqrt(2.0)
SQRT3 = math.sqrt(3.0)
LENGTHSCALE_FLOOR = 2.5e-2  # numerical-stability floor used by BoTorch's port


def dim_scaled_lengthscale_prior(d: int) -> LogNormalPrior:
    """l_i ~ LogNormal(sqrt(2) + log(d)/2, sqrt(3)) -- mode scales as sqrt(d)."""
    return LogNormalPrior(loc=SQRT2 + math.log(d) * 0.5, scale=SQRT3)


def dsp_lengthscale_mode(d: int) -> float:
    """Mode of the DSP lengthscale prior = exp(loc - scale^2) = exp(sqrt2 - 3)*sqrt(d)."""
    return math.exp((SQRT2 + math.log(d) * 0.5) - SQRT3**2)


def _matern_dsp(d: int) -> MaternKernel:
    prior = dim_scaled_lengthscale_prior(d)
    kernel = MaternKernel(
        nu=2.5,
        ard_num_dims=d,
        lengthscale_prior=prior,
        lengthscale_constraint=GreaterThan(LENGTHSCALE_FLOOR, transform=None, initial_value=prior.mode),
    )
    return kernel


def build_dsp_gp(train_X: torch.Tensor, train_Y: torch.Tensor) -> SingleTaskGP:
    """DSP: Matern-5/2 ARD, sqrt(D)-scaled LogNormal lengthscale prior,
    signal variance fixed to 1 (no ScaleKernel), standardized outputs."""
    d = train_X.shape[-1]
    gp = SingleTaskGP(
        train_X,
        train_Y,
        covar_module=_matern_dsp(d),  # bare Matern => outputscale == 1 (sigma_f^2 = 1)
        input_transform=Normalize(d=d),
        outcome_transform=Standardize(m=1),
    )
    return gp


def build_default_gp(train_X: torch.Tensor, train_Y: torch.Tensor) -> SingleTaskGP:
    """Control baseline: classic BoTorch default priors (the D-independent
    Gamma(3,6) lengthscale prior the paper argues is the culprit)."""
    d = train_X.shape[-1]
    covar = ScaleKernel(
        MaternKernel(nu=2.5, ard_num_dims=d, lengthscale_prior=GammaPrior(3.0, 6.0)),
        outputscale_prior=GammaPrior(2.0, 0.15),
    )
    gp = SingleTaskGP(
        train_X,
        train_Y,
        covar_module=covar,
        input_transform=Normalize(d=d),
        outcome_transform=Standardize(m=1),
    )
    return gp
