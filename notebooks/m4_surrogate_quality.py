"""M4 surrogate-quality evaluation.

Loads a trained PFN checkpoint and reports its predictive quality against
two held-out targets:

1. **Lumped library held-out subset**: in-distribution prediction. RMSE,
   Spearman ρ, mean NLL of the binned Riemann distribution. This is the
   pre-FEA sanity check.
2. **FEA probe set** (results1/, the §11 probe): out-of-distribution
   prediction. Spearman ρ vs. T_FEA — the same metric M1 v3 passed at 0.771.

Run:
    python notebooks/m4_surrogate_quality.py checkpoints/<gen>_pfn.pt
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from machine_design.generators import (
    HacklGenerator_3BrokenLines,
    HacklGenerator_OneLambda,
    HacklGenerator_SixLambdas,
)
from machine_design.lumped import (
    GRANULARITY_FINE,
    REFERENCE_MACHINE,
    build_network,
    lumped_torque_proxy_saturated,
)
from machine_design.pfn import PFNSurrogate, load_checkpoint, load_library


GEN_LOOKUP = {
    "OneLambda": ("HacklGenerator_OneLambda", HacklGenerator_OneLambda),
    "SixLambdas": ("HacklGenerator_SixLambdas", HacklGenerator_SixLambdas),
    "ThreeBrokenLines": ("HacklGenerator_3BrokenLines", HacklGenerator_3BrokenLines),
}


def _flatten_params(params) -> np.ndarray:
    flat: list[float] = []
    for x in params:
        if hasattr(x, "__iter__"):
            for y in x:
                flat.append(float(y))
        else:
            flat.append(float(x))
    return np.array(flat, dtype=float)


def eval_lumped_holdout(loaded, library, n_ctx: int = 64, n_test: int = 200, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    N = len(library)
    if N < n_ctx + n_test:
        n_test = max(8, N - n_ctx)
    idx = rng.permutation(N)
    ctx_idx, tst_idx = idx[:n_ctx], idx[n_ctx : n_ctx + n_test]
    y_g = library.T_proxy["FINE"]
    ctx_X = torch.tensor(library.params[ctx_idx], dtype=torch.float32)
    ctx_Y = torch.tensor(y_g[ctx_idx], dtype=torch.float32).unsqueeze(-1)
    tst_X = torch.tensor(library.params[tst_idx], dtype=torch.float32)
    tst_Y_true = y_g[tst_idx]

    surr = PFNSurrogate.from_loaded(loaded, ctx_X, ctx_Y)
    post = surr.posterior(tst_X.unsqueeze(0))
    pred_mean = surr.denormalise_mean(post.mean).squeeze().detach().cpu().numpy()
    pred_std = np.sqrt(surr.denormalise_variance(post.variance).squeeze().detach().cpu().numpy())
    rmse = float(np.sqrt(np.mean((pred_mean - tst_Y_true) ** 2)))
    rho, _ = spearmanr(pred_mean, tst_Y_true)
    rel_rmse = rmse / float(np.std(tst_Y_true) + 1e-12)
    return {
        "n_ctx": n_ctx,
        "n_test": int(len(tst_idx)),
        "rmse": rmse,
        "rel_rmse": rel_rmse,
        "spearman_pred_vs_true": float(rho),
        "pred_mean_avg_std": float(np.mean(pred_std)),
    }


def eval_fea_probe(loaded, library, n_ctx: int = 64, probe_dir: Path | None = None,
                   seed: int = 0) -> dict | None:
    """Predict on the §11 FEA probe set with `n_ctx` lumped-library rows as context."""
    if probe_dir is None:
        probe_dir = Path("results/results1")
    if not probe_dir.exists():
        print(f"  [skip FEA probe: {probe_dir} not found]")
        return None
    method_name, gen_cls = GEN_LOOKUP[loaded.generator_name]
    meta = pd.read_csv(probe_dir / "metadata.csv")
    meta = meta[(meta["method"] == method_name) & (~meta["T"].isnull())].reset_index(drop=True)
    if len(meta) == 0:
        print(f"  [skip FEA probe: no rows for {method_name} in metadata.csv]")
        return None

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(library), n_ctx, replace=False)
    y_g = library.T_proxy["FINE"]
    ctx_X = torch.tensor(library.params[idx], dtype=torch.float32)
    ctx_Y = torch.tensor(y_g[idx], dtype=torch.float32).unsqueeze(-1)

    test_X: list[np.ndarray] = []
    test_FEA: list[float] = []
    gen = gen_cls(REFERENCE_MACHINE, r_stator_end=0.7, offset=0.35)
    for _, row in meta.iterrows():
        with open(probe_dir / f"design_{method_name}_{int(row['design'])}.pkl", "rb") as f:
            params = pickle.load(f)
        test_X.append(_flatten_params(params))
        test_FEA.append(float(row["T"]))

    if not test_X:
        return None
    tst_X = torch.tensor(np.stack(test_X), dtype=torch.float32)
    surr = PFNSurrogate.from_loaded(loaded, ctx_X, ctx_Y)
    post = surr.posterior(tst_X.unsqueeze(0))
    pred_mean = surr.denormalise_mean(post.mean).squeeze().detach().cpu().numpy()
    rho, p = spearmanr(pred_mean, test_FEA)
    return {
        "n_ctx": n_ctx,
        "n_probe": len(test_FEA),
        "spearman_pred_vs_FEA": float(rho),
        "p_value": float(p),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--n-ctx", type=int, default=64)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--probe-dir", type=Path, default=Path("results/results1"))
    args = ap.parse_args()

    loaded = load_checkpoint(args.checkpoint)
    library = load_library(loaded.library_path)
    print(f"Checkpoint:    {args.checkpoint}")
    print(f"Generator:     {loaded.generator_name}  (D={loaded.input_dim})")
    print(f"Library:       {loaded.library_path}  (N={len(library)}, sha={loaded.library_sha256[:12]}…)")
    print(f"Lumped tag:    {loaded.lumped_tag}")
    print(f"Granularity:   {loaded.granularity_mode}")
    print()

    print("=== In-distribution (lumped library held-out) ===")
    r1 = eval_lumped_holdout(loaded, library, n_ctx=args.n_ctx, n_test=args.n_test)
    print(f"  n_ctx={r1['n_ctx']}, n_test={r1['n_test']}")
    print(f"  RMSE             = {r1['rmse']:.3e}")
    print(f"  rel-RMSE         = {r1['rel_rmse']:.3f}")
    print(f"  Spearman ρ       = {r1['spearman_pred_vs_true']:+.3f}")
    print(f"  avg pred σ       = {r1['pred_mean_avg_std']:.3e}")

    print()
    print("=== Out-of-distribution (FEA probe set from results1/) ===")
    r2 = eval_fea_probe(loaded, library, n_ctx=args.n_ctx, probe_dir=args.probe_dir)
    if r2 is None:
        return 0
    print(f"  n_ctx={r2['n_ctx']}, n_probe={r2['n_probe']}")
    print(f"  Spearman ρ vs T_FEA = {r2['spearman_pred_vs_FEA']:+.3f}  (p={r2['p_value']:.2e})")

    # Honest threshold check (the §11 0.5 target was for the LUMPED solver,
    # not the PFN — but this is the most direct comparable number).
    if abs(r2["spearman_pred_vs_FEA"]) >= 0.5:
        print(f"\nPASS: |ρ_PFN_vs_FEA| = {abs(r2['spearman_pred_vs_FEA']):.3f} ≥ 0.5")
        return 0
    print(f"\nNote: |ρ_PFN_vs_FEA| = {abs(r2['spearman_pred_vs_FEA']):.3f} < 0.5 "
          "— v3 lumped on the same probe gives ρ=0.77; gap is the PFN approximation cost.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
