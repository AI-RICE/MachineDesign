"""Unit tests for the Hvarfner-2024 DSP replication.

Run:  .venv/bin/python -m pytest replications/vanilla_hdbo/test_dsp.py -q
 or:  .venv/bin/python replications/vanilla_hdbo/test_dsp.py
"""

import math
import os
import sys

import torch
from gpytorch.kernels import ScaleKernel
from gpytorch.priors import LogNormalPrior

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmarks import EmbeddedTestFunction  # noqa: E402
from dsp_prior import (  # noqa: E402
    SQRT2,
    SQRT3,
    build_default_gp,
    build_dsp_gp,
    dim_scaled_lengthscale_prior,
    dsp_lengthscale_mode,
)


def test_prior_mode_scaling():
    # Paper: mode ~= 0.50 at D=6.
    assert abs(dsp_lengthscale_mode(6) - 0.50) < 1e-2, dsp_lengthscale_mode(6)
    # Mode scales as sqrt(D): mode(D)/mode(1) == sqrt(D).
    for d in (1, 4, 25, 100, 300, 1000):
        ratio = dsp_lengthscale_mode(d) / dsp_lengthscale_mode(1)
        assert abs(ratio - math.sqrt(d)) < 1e-6, (d, ratio)
    # loc/scale match Eq. 4 with mu0=sqrt2, sigma0=sqrt3.
    p = dim_scaled_lengthscale_prior(25)
    assert abs(float(p.loc) - (SQRT2 + math.log(25) / 2)) < 1e-5  # prior stored in float32
    assert abs(float(p.scale) - SQRT3) < 1e-5


def test_prior_matches_botorch_port():
    """Cross-validate our explicit prior against BoTorch's shipped port."""
    try:
        from botorch.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior
    except Exception:
        return  # helper absent in this BoTorch — skip cross-check
    for d in (6, 25, 100):
        k = get_covar_module_with_dim_scaled_prior(ard_num_dims=d)
        ln = [pr for _, _, pr, *_ in k.named_priors() if isinstance(pr, LogNormalPrior)]
        assert ln, "no LogNormalPrior found in BoTorch port"
        ours = dim_scaled_lengthscale_prior(d)
        assert abs(float(ln[0].loc) - float(ours.loc)) < 1e-9, d
        assert abs(float(ln[0].scale) - float(ours.scale)) < 1e-9, d


def test_gp_construction():
    torch.manual_seed(0)
    d = 25
    X = torch.rand(20, d, dtype=torch.double)
    Y = torch.rand(20, 1, dtype=torch.double)
    dsp = build_dsp_gp(X, Y)
    default = build_default_gp(X, Y)
    # DSP fixes sigma_f^2 = 1 -> bare Matern, no ScaleKernel; default has ScaleKernel.
    assert not isinstance(dsp.covar_module, ScaleKernel)
    assert isinstance(default.covar_module, ScaleKernel)
    # DSP lengthscale prior is the sqrt(D)-scaled LogNormal.
    lns = [pr for _, _, pr, *_ in dsp.covar_module.named_priors() if isinstance(pr, LogNormalPrior)]
    assert lns and abs(float(lns[0].loc) - (SQRT2 + math.log(d) / 2)) < 1e-5  # prior stored in float32
    # posteriors evaluate.
    Xt = torch.rand(5, d, dtype=torch.double)
    for gp in (dsp, default):
        post = gp.posterior(Xt)
        assert post.mean.shape == (5, 1)


def test_benchmark_optima_and_embedding():
    # Hartmann-6 at its known optimiser attains the optimal value.
    h = EmbeddedTestFunction("hartmann6", dim=6, seed=0)
    x_opt = torch.tensor([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573], dtype=torch.double)
    assert abs(float(h(x_opt)) - h.f_opt) < 1e-3, float(h(x_opt))
    # Levy-4 minimum is 0 at the all-ones point (native domain), i.e. u=0.5 in [0,1].
    lv = EmbeddedTestFunction("levy4", dim=4, seed=0)
    # native Levy min at x=1; domain [-10,10] -> u = (1+10)/20 = 0.55
    u = torch.full((4,), 0.55, dtype=torch.double)
    assert float(lv(u)) < 1e-6, float(lv(u))
    # Embedding: perturbing INACTIVE dims must not change the value.
    emb = EmbeddedTestFunction("hartmann6", dim=25, seed=1)
    x = torch.rand(25, dtype=torch.double)
    v0 = float(emb(x))
    inactive = [i for i in range(25) if i not in emb.active.tolist()]
    x2 = x.clone()
    x2[inactive] = torch.rand(len(inactive), dtype=torch.double)
    assert abs(float(emb(x2)) - v0) < 1e-9
    # Active dims DO change the value.
    x3 = x.clone()
    x3[emb.active] = torch.rand(emb.d_eff, dtype=torch.double)
    assert abs(float(emb(x3)) - v0) > 1e-6


def test_log_regret_monotone():
    h = EmbeddedTestFunction("hartmann6", dim=25, seed=0)
    assert h.log_regret(h.f_opt + 1.0) == 0.0
    assert h.log_regret(h.f_opt + 0.01) < h.log_regret(h.f_opt + 1.0)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
