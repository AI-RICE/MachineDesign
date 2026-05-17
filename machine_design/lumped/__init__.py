"""Lumped-reluctance model of the reference SynRM.

This subpackage encodes a physically-motivated, FEA-data-uninformed prior over
torque/ripple as a function of Hackl-style barrier parameters. The PFN
meta-trains on samples drawn from this prior.

See `applications/ReluctanceDrive/CLAUDE.md` §4 (prior spec) and §11
(data-hygiene protocol). Constants are cited in `REFERENCES.md`.
"""

from .geometry import MachineSpec, REFERENCE_MACHINE
from .granularity import (
    GRANULARITY_COARSE,
    GRANULARITY_FINE,
    GRANULARITY_MEDIUM,
    Granularity,
    sample_granularity,
)
from .network import LumpedNetwork, build_network
from .visualize import plot_granularity_grid, plot_network

__all__ = [
    "MachineSpec",
    "REFERENCE_MACHINE",
    "GRANULARITY_COARSE",
    "GRANULARITY_MEDIUM",
    "GRANULARITY_FINE",
    "Granularity",
    "sample_granularity",
    "LumpedNetwork",
    "build_network",
    "plot_network",
    "plot_granularity_grid",
]
