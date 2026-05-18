"""Lumped-network torque proxy.

For SynRM at MTPA, `T_mean ∝ L_d − L_q`. With a fixed peak MMF and a
linear network, `L ∝ W` (coenergy), so

    T_proxy = W_d − W_q

is a monotone-related scalar suitable for Spearman rank correlation against
FEA-evaluated `T_mean`. Magnitudes are not physically calibrated in v1; only
ranks across designs are claimed. T_ripple is left for v2 (would require a
rotating-MMF sweep over one electrical period).
"""

from __future__ import annotations

from dataclasses import dataclass

from .network import LumpedNetwork
from .reluctance import compute_edge_reluctances
from .solve import coenergy, solve_potentials


@dataclass(frozen=True)
class LumpedTorqueResult:
    W_d: float    # coenergy under d-aligned MMF
    W_q: float    # coenergy under q-aligned MMF
    T_proxy: float    # W_d - W_q


def lumped_torque_proxy(net: LumpedNetwork, mmf_amp: float = 1.0) -> LumpedTorqueResult:
    """Compute (W_d, W_q, T_proxy = W_d − W_q) for one design."""
    R_per_edge = compute_edge_reluctances(net)

    psi_d, _, edge_pairs = solve_potentials(net, R_per_edge, mmf_mode="d", mmf_amp=mmf_amp)
    W_d = coenergy(psi_d, edge_pairs)

    psi_q, _, _ = solve_potentials(net, R_per_edge, mmf_mode="q", mmf_amp=mmf_amp)
    W_q = coenergy(psi_q, edge_pairs)

    return LumpedTorqueResult(W_d=W_d, W_q=W_q, T_proxy=W_d - W_q)
