"""M1 probe-set evaluation per CLAUDE.md §11.

Loads 60 designs from `results/results1/` (the only seed allowed at the
probe stage), evaluates the lumped torque proxy for each, and reports
Spearman correlations against the FEA-measured T_mean / ripple. Pass
condition is declared **before** the run: Spearman ≥ 0.5 on T_mean and
|T_lumped| not degenerate.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
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
    lumped_torque_proxy,
)

GENERATORS = {
    "HacklGenerator_OneLambda": HacklGenerator_OneLambda,
    "HacklGenerator_SixLambdas": HacklGenerator_SixLambdas,
    "HacklGenerator_3BrokenLines": HacklGenerator_3BrokenLines,
}


def run_probe(results_root: Path, n_per_generator: int = 20) -> pd.DataFrame:
    """Return a DataFrame with one row per probe design (method, design, T_FEA,
    ripple_FEA, W_d, W_q, T_proxy).
    """
    meta = pd.read_csv(results_root / "metadata.csv")
    meta = meta[~meta["T"].isnull()].reset_index(drop=True)
    rng = np.random.default_rng(0)
    rows = []
    for method, gen_cls in GENERATORS.items():
        method_meta = meta[meta["method"] == method].reset_index(drop=True)
        if len(method_meta) == 0:
            print(f"  WARNING: no rows for {method}")
            continue
        n = min(n_per_generator, len(method_meta))
        picks = rng.choice(len(method_meta), size=n, replace=False)
        gen = gen_cls(REFERENCE_MACHINE, r_stator_end=0.7, offset=0.35)
        for i in picks:
            row = method_meta.iloc[int(i)]
            design_idx = int(row["design"])
            pkl_path = results_root / f"design_{method}_{design_idx}.pkl"
            with open(pkl_path, "rb") as f:
                params = pickle.load(f)
            gen.set_parameters(params)
            barriers = gen.generate_barriers()
            net = build_network(REFERENCE_MACHINE, GRANULARITY_FINE, barrier_polylines=barriers)
            res = lumped_torque_proxy(net)
            rows.append(
                {
                    "method": method,
                    "design": design_idx,
                    "T_FEA": float(row["T"]),
                    "ripple_FEA": float(row["ripple"]),
                    "W_d": res.W_d,
                    "W_q": res.W_q,
                    "T_proxy": res.T_proxy,
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    here = Path(__file__).resolve().parent
    results_root = here.parent / "results" / "results1"
    if not results_root.exists():
        # The repo's gitignored data lives at the parent project's path.
        results_root = (
            Path("/Users/smidl/zcu/PFN4BOrevisited/applications/ReluctanceDrive/MachineDesign/results/results1")
        )
    if not results_root.exists():
        print(f"ERROR: results1 not found at {results_root}", file=sys.stderr)
        return 1
    print(f"Probe set: {results_root}")

    df = run_probe(results_root, n_per_generator=20)
    out_csv = here / "m1_probe_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}  ({len(df)} designs)")

    # ── Spearman correlations ─────────────────────────────────────────────
    def _spearman(a, b, label):
        rho, p = spearmanr(a, b)
        print(f"  {label:<28} ρ={rho:+.3f}  (p={p:.2e})")
        return rho

    print("\nSpearman (full probe set, n={}):".format(len(df)))
    rho_T_proxy = _spearman(df["T_proxy"], df["T_FEA"], "T_proxy vs T_FEA")
    _spearman(df["W_d"], df["T_FEA"], "W_d vs T_FEA")
    _spearman(df["W_q"], df["T_FEA"], "W_q vs T_FEA")
    _spearman(df["T_proxy"], df["ripple_FEA"], "T_proxy vs ripple_FEA")

    print("\nPer-generator:")
    for method, group in df.groupby("method"):
        rho, _ = spearmanr(group["T_proxy"], group["T_FEA"])
        print(f"  {method:<28}  n={len(group):>3}  ρ(T_proxy, T_FEA)={rho:+.3f}")

    threshold = 0.5
    abs_rho = abs(rho_T_proxy)
    if abs_rho >= threshold:
        print(f"\nPASS: |ρ(T_proxy, T_FEA)| = {abs_rho:.3f} ≥ {threshold}")
        return 0
    print(f"\nFAIL: |ρ(T_proxy, T_FEA)| = {abs_rho:.3f} < {threshold}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
