"""Probe the PFN's implicit kernel hyperparameters via GP fitting on its predictions.

Question: what kernel hyperparameters does our GP-prior PFN's posterior
predictive *look like*, when given a real FEA context?

Method (diagnostic B from the post-200k discussion):
  1. Pick a fixed context of `n_ctx` FEA-evaluated OneLambda designs.
  2. Predict the PFN's mean on a dense `query` grid of feasible designs.
  3. Fit a SingleTaskGP (Matern + ARD, Type-II ML) to (query_X, PFN_means).
     The fitted GP's hyperparameters are the PFN's *implicit choice* for
     that context — the kernel that best reproduces what the PFN is
     telling us about the function.

For comparison we also fit a SingleTaskGP to the same context + the real
FEA T_mean values at those context points. Its fitted hyperparameters
are the FEA-aware reference (what a Bayesian-optimal GP would pick on
the actual data). Comparing the two reveals whether the PFN's training
hyperprior is misaligned with FEA's true characteristics.

Output: per-(context-size) implicit hyperparams + FEA-direct reference
hyperparams, printed and CSV'd.

NO FEA hyperparameters are used to *train* anything — this is purely
diagnostic. The §11 hygiene budget for reading FEA HPs is small: we look
once per checkpoint to decide whether a follow-up training is warranted.
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from gpytorch.mlls import ExactMarginalLogLikelihood

from machine_design.fea_emulator import load_fea_designs
from machine_design.generators import HacklGenerator_OneLambda
from machine_design.lumped import REFERENCE_MACHINE
from machine_design.pfn import PFNSurrogate, load_checkpoint


def _fit_gp(X_np, y_np, bounds_t):
    X_t = torch.from_numpy(X_np.astype(np.float64))
    y_t = torch.from_numpy(y_np.astype(np.float64)).unsqueeze(-1)
    gp = SingleTaskGP(X_t, y_t, input_transform=Normalize(d=X_t.shape[1], bounds=bounds_t))
    fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
    gp.eval()
    return gp


def _gp_hyperparams(gp) -> dict:
    """Extract length scales (in normalised [0,1]^d input space) and noise.

    Recent BoTorch SingleTaskGP uses an unwrapped Matern/RBF kernel (no
    ScaleKernel), so we handle both shapes. `outputscale` is reported as
    NaN when not present — the lengthscales and noise are what matter
    for the implicit-kernel diagnosis.
    """
    cm = gp.covar_module
    if hasattr(cm, "base_kernel"):
        ls = cm.base_kernel.lengthscale.detach().cpu().numpy().reshape(-1)
        os = float(cm.outputscale.detach().cpu().item())
    else:
        ls = cm.lengthscale.detach().cpu().numpy().reshape(-1)
        os = float("nan")
    noise = float(gp.likelihood.noise.detach().cpu().item())
    return {"lengthscales": ls.tolist(), "outputscale": os, "noise": noise,
            "lengthscale_min": float(ls.min()), "lengthscale_max": float(ls.max()),
            "lengthscale_median": float(np.median(ls))}


def _pfn_predict(loaded, X_ctx_np, y_ctx_np, X_q_np) -> np.ndarray:
    surr = PFNSurrogate.from_loaded_with_real_Y(
        loaded,
        torch.from_numpy(X_ctx_np.astype(np.float32)),
        torch.from_numpy(y_ctx_np.astype(np.float32)).unsqueeze(-1),
    )
    with torch.no_grad():
        post = surr.posterior(torch.from_numpy(X_q_np.astype(np.float32)).unsqueeze(0))
        mean_norm = post.mean.squeeze().cpu().numpy()
    return mean_norm * surr.y_std + surr.y_mean


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n-query", type=int, default=500,
                    help="how many random-feasible designs to probe the PFN at")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=Path("sweeps"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_checkpoint(args.checkpoint)
    fea = load_fea_designs("OneLambda")

    # bounds for SingleTaskGP normalisation — pulled from generator, not from FEA data
    gen = HacklGenerator_OneLambda(REFERENCE_MACHINE, r_stator_end=0.7, offset=0.35)
    lo, hi = gen.bounds
    bounds_np = np.stack([np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)])
    bounds_t = torch.from_numpy(bounds_np)
    D = bounds_t.shape[1]

    rng = np.random.default_rng(args.seed)
    # Sample `n_query` random feasible designs (no FEA torque needed for these — we'll just
    # ask the PFN what it predicts at these points).
    X_query = []
    while len(X_query) < args.n_query:
        c = rng.uniform(lo, hi)
        gen.set_parameters(gen.X_to_params(c))
        if gen.feasible_barriers(gen.generate_barriers()):
            X_query.append(c)
    X_query = np.stack(X_query)

    rows = []
    for n_ctx in [16, 32, 64]:
        ctx_idx = rng.choice(len(fea.X), n_ctx, replace=False)
        X_ctx = fea.X[ctx_idx]
        y_ctx = fea.T_mean[ctx_idx]

        # PFN's predicted mean on the query grid (real T_mean units after denorm).
        y_query_pfn = _pfn_predict(loaded, X_ctx, y_ctx, X_query)

        # 1) Implicit GP fit to (X_query, PFN_means) — what kernel does the PFN imply?
        gp_implicit = _fit_gp(X_query, y_query_pfn, bounds_t)
        impl = _gp_hyperparams(gp_implicit)
        impl["model"] = "PFN-implicit (GP fit to PFN predictions on n_query grid)"
        impl["n_ctx"] = n_ctx
        impl["n_query"] = args.n_query
        rows.append(impl)

        # 2) FEA-aware reference: GP fit directly to (X_ctx, y_ctx_FEA).
        #    This is what a Bayesian-optimal GP would pick on the same data.
        gp_truth = _fit_gp(X_ctx, y_ctx, bounds_t)
        truth = _gp_hyperparams(gp_truth)
        truth["model"] = "FEA-direct (Type-II ML GP on same context)"
        truth["n_ctx"] = n_ctx
        truth["n_query"] = n_ctx
        rows.append(truth)

        print(f"\n=== n_ctx = {n_ctx} ===")
        print(f"PFN-implicit  | ls range [{impl['lengthscale_min']:.3f}, {impl['lengthscale_max']:.3f}]  median={impl['lengthscale_median']:.3f}  "
              f"os={impl['outputscale']:.3e}  noise={impl['noise']:.3e}")
        print(f"FEA-direct    | ls range [{truth['lengthscale_min']:.3f}, {truth['lengthscale_max']:.3f}]  median={truth['lengthscale_median']:.3f}  "
              f"os={truth['outputscale']:.3e}  noise={truth['noise']:.3e}")
        # Per-dim ratio
        ratio = np.asarray(impl["lengthscales"]) / np.asarray(truth["lengthscales"])
        print(f"per-dim PFN/FEA ls ratio: {ratio.tolist()}")
        print(f"PFN noise vs FEA noise: ratio = {impl['noise'] / max(truth['noise'], 1e-12):.3e}")

    # Tidy CSV.
    flat = []
    for r in rows:
        ls = r.pop("lengthscales")
        for d, v in enumerate(ls):
            r[f"ls_dim{d}"] = float(v)
        flat.append(r)
    df = pd.DataFrame(flat)
    csv_path = args.out_dir / "OneLambda_pfn_implicit_kernel.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
