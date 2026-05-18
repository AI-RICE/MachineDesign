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
from .bh import mu_r_bh, mu_r_bh_array
from .cross_section import edge_cross_section_m2
from .material import DEFAULT_PERP_WIDTH_M, MU_0, MU_IRON, MU_R_IRON_LINEAR
from .saturation import lumped_torque_proxy_saturated, solve_with_saturation
from .network import LumpedNetwork, build_network
from .reluctance import compute_edge_reluctances
from .solve import assemble_admittance, coenergy, solve_potentials
from .torque import LumpedTorqueResult, lumped_torque_proxy
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
    "MU_0",
    "MU_IRON",
    "MU_R_IRON_LINEAR",
    "DEFAULT_PERP_WIDTH_M",
    "edge_cross_section_m2",
    "compute_edge_reluctances",
    "assemble_admittance",
    "solve_potentials",
    "coenergy",
    "lumped_torque_proxy",
    "LumpedTorqueResult",
    "mu_r_bh",
    "mu_r_bh_array",
    "solve_with_saturation",
    "lumped_torque_proxy_saturated",
]
