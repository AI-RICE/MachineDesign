"""Offline-generated library of `(barrier_params, T_lumped)` pairs.

Each library is per-parameterisation (flavour A — D6 in CLAUDE.md). For
each randomly sampled feasible barrier design we store the v3 saturated
torque proxy at every supported `Granularity`, so the runtime sampler
can pick a random granularity per PFN training task without ever calling
the solver online.

Storage format: numpy `.npz` with arrays `params` (N, D), `T_proxy_COARSE`,
`T_proxy_MEDIUM`, `T_proxy_FINE`, `W_d_FINE`, `W_q_FINE` (N,).
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from ..generators import (
    HacklGenerator_3BrokenLines,
    HacklGenerator_OneLambda,
    HacklGenerator_SixLambdas,
)
from ..lumped import (
    GRANULARITY_COARSE,
    GRANULARITY_FINE,
    GRANULARITY_MEDIUM,
    REFERENCE_MACHINE,
    Granularity,
    build_network,
    lumped_torque_proxy_saturated,
)


GENERATOR_REGISTRY = {
    "OneLambda": HacklGenerator_OneLambda,
    "SixLambdas": HacklGenerator_SixLambdas,
    "ThreeBrokenLines": HacklGenerator_3BrokenLines,
}

GRANULARITIES = {
    "COARSE": GRANULARITY_COARSE,
    "MEDIUM": GRANULARITY_MEDIUM,
    "FINE": GRANULARITY_FINE,
}


@dataclass(frozen=True)
class LumpedLibraryEntry:
    """One row of the library."""
    params: np.ndarray                    # (D,) flat barrier-param vector
    T_proxy: dict[str, float]             # granularity name → T_proxy
    W_d_fine: float                       # FINE coenergy (for diagnostics)
    W_q_fine: float


@dataclass
class LumpedLibrary:
    """In-memory library, sliceable for PFN training."""
    generator_name: str
    params: np.ndarray                    # (N, D)
    T_proxy: dict[str, np.ndarray]        # granularity → (N,)
    W_d_fine: np.ndarray                  # (N,)
    W_q_fine: np.ndarray                  # (N,)

    def __len__(self) -> int:
        return self.params.shape[0]


def _sample_feasible_params(
    generator_name: str,
    rng: np.random.Generator,
    max_tries: int = 100,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Sample a barrier-parameter vector whose barriers are feasible.

    Returns `(flat_params, barrier_polylines)`.
    """
    gen_cls = GENERATOR_REGISTRY[generator_name]
    gen = gen_cls(REFERENCE_MACHINE, r_stator_end=0.7, offset=0.35)
    for _ in range(max_tries):
        # Hackl generators use np.random under the hood; seed it from `rng`.
        np.random.seed(int(rng.integers(0, 2**31 - 1)))
        params = gen.random_parameters()
        gen.set_parameters(params)
        barriers = gen.generate_barriers()
        if gen.feasible_barriers(barriers):
            flat = _flatten_params(params)
            return flat, barriers
    raise RuntimeError(f"Could not sample feasible {generator_name} in {max_tries} tries")


def _flatten_params(params) -> np.ndarray:
    """Hackl generators return nested tuples; flatten to a 1-D float array."""
    flat: list[float] = []
    for x in params:
        if hasattr(x, "__iter__"):
            for y in x:
                flat.append(float(y))
        else:
            flat.append(float(x))
    return np.array(flat, dtype=float)


def _evaluate_one(args: tuple[str, int]) -> dict | None:
    """Worker for ProcessPoolExecutor. Returns a record dict or None on failure."""
    generator_name, seed = args
    try:
        rng = np.random.default_rng(seed)
        flat_params, barriers = _sample_feasible_params(generator_name, rng)
        out = {"params": flat_params, "T_proxy": {}, "W_d_fine": 0.0, "W_q_fine": 0.0}
        for gname, granularity in GRANULARITIES.items():
            net = build_network(REFERENCE_MACHINE, granularity, barrier_polylines=barriers)
            res = lumped_torque_proxy_saturated(net, mmf_amp=200.0)
            out["T_proxy"][gname] = float(res.T_proxy)
            if gname == "FINE":
                out["W_d_fine"] = float(res.W_d)
                out["W_q_fine"] = float(res.W_q)
        return out
    except Exception:
        return None


def build_library(
    generator_name: str,
    n_samples: int,
    base_seed: int = 0,
    n_workers: int | None = None,
    verbose: bool = True,
) -> LumpedLibrary:
    """Generate `n_samples` feasible designs and evaluate at all three granularities.

    Parallelised over processes. `base_seed` makes the sample reproducible.
    """
    if generator_name not in GENERATOR_REGISTRY:
        raise ValueError(f"unknown generator: {generator_name!r}")

    seeds = [base_seed + i for i in range(n_samples)]
    args = [(generator_name, s) for s in seeds]

    rows: list[dict] = []
    if n_workers is None or n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(_evaluate_one, a): a for a in args}
            for k, fut in enumerate(as_completed(futures)):
                rec = fut.result()
                if rec is not None:
                    rows.append(rec)
                if verbose and (k + 1) % max(1, n_samples // 20) == 0:
                    print(f"  {k+1}/{n_samples} ({len(rows)} valid)")
    else:
        for k, a in enumerate(args):
            rec = _evaluate_one(a)
            if rec is not None:
                rows.append(rec)
            if verbose and (k + 1) % max(1, n_samples // 20) == 0:
                print(f"  {k+1}/{n_samples} ({len(rows)} valid)")

    if not rows:
        raise RuntimeError("no feasible samples generated")

    # Stack into arrays.
    D = rows[0]["params"].shape[0]
    params = np.stack([r["params"] for r in rows])              # (N, D)
    T_proxy = {g: np.array([r["T_proxy"][g] for r in rows]) for g in GRANULARITIES}
    W_d = np.array([r["W_d_fine"] for r in rows])
    W_q = np.array([r["W_q_fine"] for r in rows])

    return LumpedLibrary(
        generator_name=generator_name,
        params=params,
        T_proxy=T_proxy,
        W_d_fine=W_d,
        W_q_fine=W_q,
    )


def save_library(lib: LumpedLibrary, path: str | Path) -> None:
    """Write the library to a single .npz file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generator_name": np.array(lib.generator_name),
        "params": lib.params,
        "W_d_fine": lib.W_d_fine,
        "W_q_fine": lib.W_q_fine,
    }
    for g, arr in lib.T_proxy.items():
        payload[f"T_proxy_{g}"] = arr
    np.savez_compressed(path, **payload)


def load_library(path: str | Path) -> LumpedLibrary:
    """Load a previously-saved library."""
    data = np.load(path, allow_pickle=False)
    T_proxy = {
        g: data[f"T_proxy_{g}"]
        for g in GRANULARITIES
        if f"T_proxy_{g}" in data
    }
    return LumpedLibrary(
        generator_name=str(data["generator_name"]),
        params=data["params"],
        T_proxy=T_proxy,
        W_d_fine=data["W_d_fine"],
        W_q_fine=data["W_q_fine"],
    )
