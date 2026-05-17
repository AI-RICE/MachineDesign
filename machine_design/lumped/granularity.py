"""Mesh resolution descriptor.

The lumped-reluctance network exists at a chosen resolution: how many yoke
nodes per pole, how many tooth nodes, how many airgap segments, how many
rotor flux-tube nodes per inter-barrier channel. The PFN is meta-trained
over random resolutions so it amortises across mesh granularity (see
CLAUDE.md §4, "Granularity amortization").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Granularity:
    """Resolution of the lumped-reluctance graph for a single pole sector.

    `n_col` is the shared angular sample count used by every "ring": the
    airgap arc, the rotor surface arc, and each channel midline curve. It
    must be odd and ≥ 3 so the column angles `linspace(0, 90, n_col)` include
    the three rails {0°, 45°, 90°} as exact samples. The rails are the only
    angular positions where cross-channel edges live.
    """

    n_yoke: int          # stator yoke nodes per pole
    n_teeth: int         # stator tooth nodes per pole
    n_col: int           # shared column count; odd, >= 3
    n_shaft: int = 1     # rotor shaft nodes per pole sector

    def __post_init__(self) -> None:
        if self.n_col < 3 or self.n_col % 2 == 0:
            raise ValueError(f"n_col must be odd and >= 3, got {self.n_col}")


# Standard granularities used for illustrations and ablations.
GRANULARITY_COARSE = Granularity(n_yoke=1, n_teeth=9,  n_col=5)
GRANULARITY_MEDIUM = Granularity(n_yoke=2, n_teeth=9,  n_col=9)
GRANULARITY_FINE   = Granularity(n_yoke=4, n_teeth=18, n_col=17)


_N_YOKE_CHOICES = (1, 2, 4)
_N_TEETH_CHOICES = (9, 18, 36)
_N_COL_CHOICES = (5, 9, 17, 33)


def sample_granularity(rng: np.random.Generator) -> Granularity:
    """Draw a granularity from the meta-training distribution.

    Uniform over the discrete choices declared in CLAUDE.md §4.
    """
    def _pick(choices: tuple[int, ...]) -> int:
        return int(choices[rng.integers(0, len(choices))])

    return Granularity(
        n_yoke=_pick(_N_YOKE_CHOICES),
        n_teeth=_pick(_N_TEETH_CHOICES),
        n_col=_pick(_N_COL_CHOICES),
    )
