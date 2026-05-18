"""M1 v3 probe-set evaluation — μ(B) saturation enabled.

Same 60 probe designs from `results1/`, same Spearman threshold (≥ 0.5),
same one-shot discipline. The only change vs v2 is the solver:
`lumped_torque_proxy_saturated` iterates the magnetic permeability against
the M350-50A B-H curve until ψ converges.
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
    lumped_torque_proxy_saturated,
)

GENERATORS = {
    "HacklGenerator_OneLambda": HacklGenerator_OneLambda,
    "HacklGenerator_SixLambdas": HacklGenerator_SixLambdas,
    "HacklGenerator_3BrokenLines": HacklGenerator_3BrokenLines,
}


def run_probe(results_root: Path, n_per_generator: int = 20, mmf_amp: float = 200.0) -> pd.DataFrame:
    meta = pd.read_csv(results_root / "metadata.csv")
    meta = meta[~meta["T"].isnull()].reset_index(drop=True)
    rng = np.random.default_rng(0)   # same seed as v1/v2 → identical probe designs
    rows = []
    for method, gen_cls in GENERATORS.items():
        method_meta = meta[meta["method"] == method].reset_index(drop=True)
        if len(method_meta) == 0:
            continue
        n = min(n_per_generator, len(method_meta))
        picks = rng.choice(len(method_meta), size=n, replace=False)
        gen = gen_cls(REFERENCE_MACHINE, r_stator_end=0.7, offset=0.35)
        for i in picks:
            row = method_meta.iloc[int(i)]
            design_idx = int(row["design"])
            with open(results_root / f"design_{method}_{design_idx}.pkl", "rb") as f:
                params = pickle.load(f)
            gen.set_parameters(params)
            barriers = gen.generate_barriers()
            net = build_network(REFERENCE_MACHINE, GRANULARITY_FINE, barrier_polylines=barriers)
            res = lumped_torque_proxy_saturated(net, mmf_amp=mmf_amp)
            rows.append({
                "method": method,
                "design": design_idx,
                "T_FEA": float(row["T"]),
                "ripple_FEA": float(row["ripple"]),
                "W_d": res.W_d,
                "W_q": res.W_q,
                "T_proxy": res.T_proxy,
            })
    return pd.DataFrame(rows)


def main() -> int:
    here = Path(__file__).resolve().parent
    results_root = here.parent / "results" / "results1"
    if not results_root.exists():
        print(f"ERROR: results1 not found at {results_root}", file=sys.stderr)
        return 1
    print(f"Probe set: {results_root}")
    print("Solver: μ(B) saturation (v3)")

    df = run_probe(results_root, n_per_generator=20)
    out_csv = here / "m1_v3_probe_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}  ({len(df)} designs)")

    def _sp(a, b, label):
        rho, p = spearmanr(a, b)
        print(f"  {label:<28} ρ={rho:+.3f}  (p={p:.2e})")
        return rho

    print(f"\nSpearman (full probe set, n={len(df)}):")
    rho = _sp(df["T_proxy"], df["T_FEA"], "T_proxy vs T_FEA")
    _sp(df["W_d"], df["T_FEA"], "W_d vs T_FEA")
    _sp(df["W_q"], df["T_FEA"], "W_q vs T_FEA")
    _sp(df["T_proxy"], df["ripple_FEA"], "T_proxy vs ripple_FEA")

    print("\nPer-generator:")
    for method, g in df.groupby("method"):
        r, _ = spearmanr(g["T_proxy"], g["T_FEA"])
        print(f"  {method:<28}  n={len(g):>3}  ρ(T_proxy, T_FEA)={r:+.3f}")

    if abs(rho) >= 0.5:
        print(f"\nPASS: |ρ| = {abs(rho):.3f} ≥ 0.5")
        return 0
    print(f"\nFAIL: |ρ| = {abs(rho):.3f} < 0.5")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
