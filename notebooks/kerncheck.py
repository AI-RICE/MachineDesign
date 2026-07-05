"""Offline kernel-CV test for the ripple(ptp) surrogate (no FEA). Does the ripple GP's
failure come from the sqrt(D)-DSP lengthscale prior over-smoothing a weak signal? Re-run
k-fold CV of ptp under several kernels/lengthscale settings; if RMSE/sd drops well below 1
the lengthscale/kernel is the lever, if it stays ~1 it's data-starvation.
"""
import math
import sys

import numpy as np
import torch

import gen2
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import LogNormalPrior

OUT = sys.argv[1] if len(sys.argv) > 1 else "results/gen2_v2"
z = np.load(f"{OUT}/gen2.npz", allow_pickle=True)
Xg = np.array(z["Xg"]); Xi = np.array(z["Xi"]); T = np.array(z["T"]); R = np.array(z["R"])
X = np.hstack([Xg, Xi]); n = len(T); d = X.shape[1]
PTP = gen2.ptp_of(R, T)
SQRT2, SQRT3 = math.sqrt(2.0), math.sqrt(3.0)
print(f"[kern] {n} pts, dim {d}, ptp sd {PTP.std():.2f}", flush=True)


def make_gp(Xt, Yt, nu, mode):
    if mode == "dsp":                       # current: sqrt(D)-scaled LogNormal prior
        pr = LogNormalPrior(loc=SQRT2 + 0.5 * math.log(d), scale=SQRT3)
        k = MaternKernel(nu=nu, ard_num_dims=d, lengthscale_prior=pr,
                         lengthscale_constraint=GreaterThan(2.5e-2, transform=None, initial_value=pr.mode))
    elif mode == "short":                   # LogNormal centred at a SHORTER lengthscale
        pr = LogNormalPrior(loc=0.0, scale=1.0)
        k = MaternKernel(nu=nu, ard_num_dims=d, lengthscale_prior=pr,
                         lengthscale_constraint=GreaterThan(1e-3, transform=None, initial_value=0.3))
    else:                                   # free: bare ML fit of lengthscale
        k = MaternKernel(nu=nu, ard_num_dims=d, lengthscale_constraint=GreaterThan(1e-3))
    return SingleTaskGP(Xt, Yt, covar_module=ScaleKernel(k), outcome_transform=Standardize(1))


def cv(Y, nu, mode):
    Kf = 8; idx = np.arange(n); np.random.RandomState(0).shuffle(idx)
    res, zz = [], []
    for f in np.array_split(idx, Kf):
        tr = np.setdiff1d(idx, f)
        g = make_gp(torch.tensor(X[tr]), torch.tensor(Y[tr]).unsqueeze(-1), nu, mode)
        fit_gpytorch_mll(ExactMarginalLogLikelihood(g.likelihood, g))
        with torch.no_grad():
            p = g.posterior(torch.tensor(X[f])); mu = p.mean.numpy().ravel(); sd = np.sqrt(p.variance.numpy().ravel())
        res += list(mu - Y[f]); zz += list((mu - Y[f]) / sd)
    res, zz = np.array(res), np.array(zz)
    rmse = np.sqrt(np.mean(res**2))
    print("  ptp  nu=%.1f  %-6s RMSE %6.3f  ratio %.2f  cover@2sig %.2f  medLS %s"
          % (nu, mode, rmse, rmse / Y.std(), np.mean(np.abs(zz) < 2), lscale_note(nu, mode)), flush=True)


def lscale_note(nu, mode):
    # fit on all data once, report median ARD lengthscale (how short did it want to go?)
    g = make_gp(torch.tensor(X), torch.tensor(PTP).unsqueeze(-1), nu, mode)
    fit_gpytorch_mll(ExactMarginalLogLikelihood(g.likelihood, g))
    ls = g.covar_module.base_kernel.lengthscale.detach().numpy().ravel()
    return "%.2f" % float(np.median(ls))


print("\n=== ptp CV across kernels (ratio<1 = signal captured; medLS = fitted lengthscale) ===")
print("  reference: T with dsp/nu2.5 ->", end=" ", flush=True)
cv(T, 2.5, "dsp")
for nu in (2.5, 1.5, 0.5):
    cv(PTP, nu, "dsp")
for nu in (2.5, 1.5, 0.5):
    cv(PTP, nu, "free")
cv(PTP, 1.5, "short")
print("\n[kern] DONE", flush=True)
