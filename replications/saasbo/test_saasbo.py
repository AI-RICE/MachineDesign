"""Unit tests for the SAASBO replication.

Core H-REPL check: on an embedded benchmark whose TRUE active dimensions are
known, SAASBO must assign markedly shorter lengthscales to the active axes than
to the inert ones (i.e. it recovers the sparse subspace). This is the behaviour
the method exists to deliver.

Run:  .venv/bin/python replications/saasbo/test_saasbo.py
(slow: NUTS fits ~10-30 s.)
"""

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "vanilla_hdbo"))

from benchmarks import EmbeddedTestFunction  # noqa: E402  (from ../vanilla_hdbo)
from saasbo import active_dimensions, build_saasbo, median_lengthscales  # noqa: E402


def test_build_and_posterior():
    torch.manual_seed(0)
    X = torch.rand(12, 6, dtype=torch.double)
    y = (X[:, :1] * 2 - 1).pow(2).sum(-1, keepdim=True)
    m = build_saasbo(X, y, warmup=64, num_samples=64, thinning=8)
    ls = median_lengthscales(m)
    assert ls.shape == (6,)
    post = m.posterior(torch.rand(4, 6, dtype=torch.double))
    # fully-Bayesian posterior carries an MCMC-sample batch dim
    assert post.mean.shape[-2:] == (4, 1)


def test_active_dimension_recovery():
    """Embedded Hartmann-6 in D=20: SAASBO should rank the 6 true active dims
    among the shortest lengthscales, and active << inactive on average."""
    torch.manual_seed(0)
    D, d_eff, n = 20, 6, 40
    f = EmbeddedTestFunction("hartmann6", dim=D, seed=1)
    true_active = set(f.active.tolist())
    X = torch.rand(n, D, dtype=torch.double)
    y = (-f(X)).unsqueeze(-1)  # maximise -Hartmann (as BO would)
    m = build_saasbo(X, y, warmup=160, num_samples=160, thinning=16)
    active, ls = active_dimensions(m, cutoff=10.0)

    # the 6 shortest-lengthscale dims should overlap the true active set well
    shortest6 = set(np.argsort(ls)[:d_eff].tolist())
    overlap = len(shortest6 & true_active)
    assert overlap >= 3, f"recovered only {overlap}/6 active dims; ls={np.round(ls,2)}"

    # active dims get shorter lengthscales than inactive ones (directional;
    # the separation sharpens with more data per the paper — modest at n=40).
    act = np.array(sorted(true_active))
    inact = np.array([i for i in range(D) if i not in true_active])
    assert ls[act].mean() < ls[inact].mean(), (ls[act].mean(), ls[inact].mean())
    print(f"  recovered {overlap}/6 true active dims in the 6 shortest lengthscales; "
          f"mean ls active={ls[act].mean():.1f} inactive={ls[inact].mean():.1f}")


def test_acquisition_runs():
    from botorch.acquisition.logei import qLogExpectedImprovement
    from botorch.optim import optimize_acqf

    torch.manual_seed(0)
    D = 8
    X = torch.rand(10, D, dtype=torch.double)
    y = (X[:, :1] * 2 - 1).pow(2).sum(-1, keepdim=True)
    m = build_saasbo(X, y, warmup=48, num_samples=48, thinning=8)
    acqf = qLogExpectedImprovement(m, best_f=y.max())
    bounds = torch.stack([torch.zeros(D, dtype=torch.double), torch.ones(D, dtype=torch.double)])
    cand, _ = optimize_acqf(acqf, bounds=bounds, q=1, num_restarts=2, raw_samples=64)
    assert cand.shape == (1, D)


if __name__ == "__main__":
    for name in ("test_build_and_posterior", "test_active_dimension_recovery", "test_acquisition_runs"):
        globals()[name]()
        print(f"PASS {name}")
    print("\nAll SAASBO tests passed.")
