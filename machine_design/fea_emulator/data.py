"""Load FEA-evaluated designs from `results/results*/results_<gen>_<bool>.npz`.

Per CLAUDE.md §3 the per-seed result files store `train_X (250, D)` and
`train_Y (250, 2)`. The first 50 rows of each are uniform random feasible
initials; rows 50..249 are BO-suggested. `train_Y` uses ICEM's transformed
convention: column 0 = `T_mean` (N·m, positive), column 1 = `-T_ripple/100`.
This loader undoes that transform so callers see `T_ripple` in % (positive).

Returns a dataclass with stacked arrays + an `is_uniform_init` mask so the
emulator can report CV RMSE on the two distributions separately (§6.5).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


GENERATOR_FILENAMES = {
    "OneLambda": "HacklGenerator_OneLambda",
    "SixLambdas": "HacklGenerator_SixLambdas",
    "ThreeBrokenLines": "HacklGenerator_3BrokenLines",
}


@dataclass(frozen=True)
class LoadedFEADesigns:
    """All FEA designs for one parameterisation, pooled across seeds."""
    generator_short: str       # "OneLambda" / "SixLambdas" / "ThreeBrokenLines"
    generator_full: str        # "HacklGenerator_<short>"
    X: np.ndarray              # (N, D)
    T_mean: np.ndarray         # (N,)
    T_ripple: np.ndarray       # (N,)  ←   in %, positive
    is_uniform_init: np.ndarray  # (N,) bool — first 50 / per seed
    seed_id: np.ndarray        # (N,) which results<k>/ the row came from
    constrained: np.ndarray    # (N,) bool — True / False (constrained sweep)


def load_fea_designs(
    generator_short: str,
    results_root: str | Path = "results",
    constrained: bool | None = None,
) -> LoadedFEADesigns:
    """Pool FEA data for one parameterisation across all `results*/` seeds.

    `constrained=True`: only `_True.npz` (the constrained sweep — the
    headline metric per CLAUDE.md §1).
    `constrained=False`: only `_False.npz`.
    `constrained=None`: both.
    """
    if generator_short not in GENERATOR_FILENAMES:
        raise ValueError(f"unknown generator: {generator_short!r}")
    full = GENERATOR_FILENAMES[generator_short]
    root = Path(results_root)
    if not root.exists():
        raise FileNotFoundError(f"{root} not found")

    if constrained is None:
        suffixes = ("True", "False")
    else:
        suffixes = ("True" if constrained else "False",)

    Xs, Tms, Trs, units, sids, cons = [], [], [], [], [], []
    seed_dirs = sorted(d for d in root.iterdir() if d.is_dir() and d.name.startswith("results"))
    for sd in seed_dirs:
        # Extract numeric seed id from the dir name (results1 → 1).
        try:
            seed_id = int(sd.name.removeprefix("results"))
        except ValueError:
            continue
        for suf in suffixes:
            p = sd / f"results_{full}_{suf}.npz"
            if not p.exists():
                continue
            data = np.load(p, allow_pickle=False)
            X = data["train_X"]              # (250, D)
            Y = data["train_Y"]              # (250, 2)
            T_mean = Y[:, 0]                  # N·m
            T_ripple = -Y[:, 1] * 100.0       # % (undo ICEM's /100 + sign flip)
            uni = np.zeros(len(X), dtype=bool)
            uni[:50] = True
            Xs.append(X)
            Tms.append(T_mean)
            Trs.append(T_ripple)
            units.append(uni)
            sids.append(np.full(len(X), seed_id, dtype=int))
            cons.append(np.full(len(X), suf == "True", dtype=bool))

    if not Xs:
        raise FileNotFoundError(
            f"No FEA files found under {root} for {full}; suffixes={suffixes}"
        )
    return LoadedFEADesigns(
        generator_short=generator_short,
        generator_full=full,
        X=np.concatenate(Xs, axis=0),
        T_mean=np.concatenate(Tms, axis=0),
        T_ripple=np.concatenate(Trs, axis=0),
        is_uniform_init=np.concatenate(units, axis=0),
        seed_id=np.concatenate(sids, axis=0),
        constrained=np.concatenate(cons, axis=0),
    )
