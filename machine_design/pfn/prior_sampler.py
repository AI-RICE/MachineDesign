"""In-context PFN task sampler.

Each call to `PriorSampler.sample(...)` returns one in-context task as a
`PFNTask(x_context, y_context, x_target, y_target)`. Inputs come straight
from a pre-built `LumpedLibrary` (no live lumped solver), so the sampler
runs at ≥10⁴ tasks/s on CPU.

Granularity amortisation (per CLAUDE.md §4): each task is evaluated at a
randomly-chosen granularity from `{COARSE, MEDIUM, FINE}`. The library
stores T at all three so the choice is just a column lookup.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .library import GRANULARITIES, LumpedLibrary


@dataclass(frozen=True)
class PFNTask:
    """One in-context task for PFN training."""
    x_context: np.ndarray   # (n_ctx, D)
    y_context: np.ndarray   # (n_ctx,)
    x_target: np.ndarray    # (n_tgt, D)
    y_target: np.ndarray    # (n_tgt,)
    granularity: str        # which T_proxy column was used


class PriorSampler:
    """Pre-loaded library → fast in-context task draws.

    Parameters
    ----------
    library : LumpedLibrary
        Pre-built library for one parameterisation.
    granularity : {'random', 'COARSE', 'MEDIUM', 'FINE'}
        If ``'random'`` (default), each task picks a granularity uniformly.
        Otherwise the named granularity is used for every task — useful for
        ablation studies (granularity-fixed PFN vs. granularity-amortised).
    """

    def __init__(self, library: LumpedLibrary, granularity: str = "random") -> None:
        self.library = library
        if granularity not in {"random", *GRANULARITIES}:
            raise ValueError(f"granularity must be 'random' or one of {set(GRANULARITIES)}")
        self.granularity_mode = granularity
        # y is normalised PER TASK (PFNs4BO standard) but using CONTEXT-ONLY
        # mean/std (NOT context+target) — see sample() for why target-inclusive
        # normalisation leaks the target at small n_target.
        # This decouples training-time scale from the absolute units of the
        # underlying lumped solver (or, at inference, any other oracle), so
        # the same PFN can be applied to FEA T_mean (~5 N·m) or lumped
        # T_proxy (~1e12) without recalibration. Per-context normalisation
        # at inference mirrors this — see machine_design/pfn/surrogate.py.

    @property
    def input_dim(self) -> int:
        return self.library.params.shape[1]

    def _pick_granularity(self, rng: np.random.Generator) -> str:
        if self.granularity_mode != "random":
            return self.granularity_mode
        idx = int(rng.integers(0, len(GRANULARITIES)))
        return list(GRANULARITIES.keys())[idx]

    def sample(
        self,
        rng: np.random.Generator,
        n_context: int = 32,
        n_target: int = 1,
        normalise: bool = True,
    ) -> PFNTask:
        """Draw one task.

        `n_context` + `n_target` rows are sampled without replacement from
        the library. The PFN convention is `n_target = 1` (one query per
        forward pass); set `n_target > 1` for batched-target training.
        """
        N = len(self.library)
        n_total = n_context + n_target
        if n_total > N:
            raise ValueError(f"requested {n_total} rows but library has {N}")
        idx = rng.choice(N, size=n_total, replace=False)
        gran = self._pick_granularity(rng)
        y_all = self.library.T_proxy[gran][idx]
        x_all = self.library.params[idx]
        if normalise:
            # CONTEXT-ONLY stats: normalising over context+target leaks the
            # target when n_target is small (z-scores sum to zero, so the
            # target is -sum(context z) independent of its x). Match the
            # inference convention, which z-scores by context only.
            yc = y_all[:n_context]
            mean = float(np.mean(yc))
            std = float(np.std(yc) + 1e-12)
            y_all = (y_all - mean) / std
        return PFNTask(
            x_context=x_all[:n_context],
            y_context=y_all[:n_context],
            x_target=x_all[n_context:],
            y_target=y_all[n_context:],
            granularity=gran,
        )

    def sample_batch(
        self,
        rng: np.random.Generator,
        batch_size: int = 64,
        n_context: int = 32,
        n_target: int = 1,
        normalise: bool = True,
    ) -> list[PFNTask]:
        """Draw `batch_size` independent tasks."""
        return [
            self.sample(rng, n_context=n_context, n_target=n_target, normalise=normalise)
            for _ in range(batch_size)
        ]
