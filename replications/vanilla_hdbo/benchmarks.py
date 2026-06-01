"""Embedded synthetic benchmarks from Hvarfner et al. 2024 (Fig. 5):
a low effective-dimensional function (Hartmann-6 or Levy-4) placed in an
ambient D-dimensional unit cube, with the remaining dimensions inert.

Convention (matches the paper's "X (dD)" embedded tasks):
  * search space is [0,1]^D,
  * a fixed random subset of `d_eff` ambient coordinates is "active",
  * the active coordinates are mapped to the test function's native domain
    and evaluated; inactive coordinates do not affect the value,
  * we MINIMISE (Hartmann-6 min = -3.32237, Levy-4 min = 0),
  * log regret = log10( best_so_far - f_opt ).

The active-subset choice is seeded so a run is reproducible but differs across
repetitions (as in embedded-benchmark practice).
"""

from __future__ import annotations

import torch
from botorch.test_functions import Hartmann, Levy

_BASE = {
    "hartmann6": (Hartmann, 6, -3.32237),
    "levy4": (Levy, 4, 0.0),
}


class EmbeddedTestFunction:
    def __init__(self, name: str, dim: int, seed: int = 0, dtype=torch.double):
        if name not in _BASE:
            raise ValueError(f"unknown benchmark {name!r}; choose from {list(_BASE)}")
        cls, d_eff, opt = _BASE[name]
        if dim < d_eff:
            raise ValueError(f"ambient dim {dim} < effective dim {d_eff}")
        self.name = name
        self.dim = dim
        self.d_eff = d_eff
        self.f_opt = opt
        self.dtype = dtype
        self._f = cls(dim=d_eff)  # native domain in self._f.bounds, shape (2, d_eff)
        # fixed random active-coordinate subset (seeded)
        g = torch.Generator().manual_seed(seed)
        self.active = torch.randperm(dim, generator=g)[:d_eff].sort().values
        self._bounds = self._f.bounds.to(dtype)  # (2, d_eff)

    @property
    def bounds(self) -> torch.Tensor:
        """Search-space bounds: [0,1]^D as a (2, D) tensor."""
        return torch.stack([torch.zeros(self.dim, dtype=self.dtype), torch.ones(self.dim, dtype=self.dtype)])

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        """X in [0,1]^D, shape (..., D) -> f values, shape (...,). Minimisation."""
        X = X.to(self.dtype)
        u = X[..., self.active]  # (..., d_eff) in [0,1]
        lo, hi = self._bounds[0], self._bounds[1]
        x_native = lo + u * (hi - lo)
        return self._f(x_native)  # botorch test funcs return value to be minimised

    def log_regret(self, best_so_far: float) -> float:
        return float(torch.log10(torch.tensor(best_so_far - self.f_opt)).clamp_min(-10.0))
