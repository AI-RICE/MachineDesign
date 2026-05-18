"""Iterative non-linear solver: μ-iron varies with local flux density (v3).

The linear v2 solver uses `μ_r = 1000` everywhere. v3 evaluates the actual
flux density per iron edge and looks up `μ_r(B)` from the M350-50A B-H
curve (`bh.mu_r_bh`), then re-solves until convergence.

Iron edges in heavily-utilised regions (typically the d-axis iron channels
of high-torque designs) saturate — their effective `μ_r` drops sharply,
their reluctance climbs, and the model's predictions for those designs
shift downward in the rank. This is exactly the mechanism that v1/v2 are
blind to.
"""

from __future__ import annotations

import numpy as np

from .bh import mu_r_bh_array
from .material import MU_0, MU_IRON
from .network import LumpedNetwork
from .reluctance import compute_edge_reluctances, edge_geometry_lengths
from .solve import coenergy, solve_potentials


# Iron edge kinds (everything that carries flux through iron — the iron/air
# split inside the edge is handled by `length_iron_m` / `length_air_m`).
_IRON_KINDS = frozenset({
    "iron_yoke", "iron_tooth", "yoke_to_tooth",
    "iron_rotor", "iron_surface", "shaft_link",
})


def solve_with_saturation(
    net: LumpedNetwork,
    mmf_mode: str,
    mmf_amp: float = 200.0,
    max_iter: int = 8,
    tol: float = 1e-3,
    relax: float = 0.5,
) -> tuple[np.ndarray, list[str], list[tuple[int, int, float]]]:
    """Fixed-point iteration: ψ ↔ μ(B).

    Returns the converged (psi, node_ids, edge_pairs). On non-convergence
    (max_iter hit), returns the last iterate.

    `mmf_amp` is the peak MMF in A-turns; with the M350-50A B-H curve, this
    must be physically calibrated for the saturation lookup to be meaningful.
    For the SIMOTICS GP-VSD4000 reference machine, fundamental MMF per pole
    is approximately `(3/2) · N · kw1 · I_peak ≈ 200 A·turns`.

    `relax` is the under-relaxation factor for μ updates (0 < relax ≤ 1).
    Closer to 1 = faster but riskier; closer to 0 = damped and stable.
    """
    geom = edge_geometry_lengths(net)
    edge_list = list(geom.keys())

    # Initial linear solve.
    mu_per_edge = {e: MU_IRON for e in edge_list}
    R = compute_edge_reluctances(net, mu_iron_per_edge=mu_per_edge)
    psi, node_ids, edge_pairs = solve_potentials(net, R, mmf_mode=mmf_mode, mmf_amp=mmf_amp)
    idx = {nid: i for i, nid in enumerate(node_ids)}

    for it in range(max_iter):
        # Compute B in each iron edge from current ψ + R.
        new_mu = dict(mu_per_edge)
        max_rel_change = 0.0
        for u, v in edge_list:
            d = net.graph.edges[u, v]
            if d["kind"] not in _IRON_KINDS:
                continue
            length_iron_m, length_air_m, A = geom[(u, v)]
            if length_iron_m <= 1e-12:
                continue
            R_uv = R[(u, v)]
            flux = (psi[idx[u]] - psi[idx[v]]) / R_uv          # Wb
            B = abs(flux) / max(A, 1e-12)                       # T
            mu_r_new = float(mu_r_bh_array(np.array([B]))[0])
            mu_new = mu_r_new * MU_0
            mu_old = mu_per_edge[(u, v)]
            # Under-relax.
            mu_blend = mu_old + relax * (mu_new - mu_old)
            new_mu[(u, v)] = mu_blend
            rel = abs(mu_blend - mu_old) / max(mu_old, 1e-30)
            if rel > max_rel_change:
                max_rel_change = rel

        if max_rel_change < tol:
            break

        mu_per_edge = new_mu
        R = compute_edge_reluctances(net, mu_iron_per_edge=mu_per_edge)
        psi, node_ids, edge_pairs = solve_potentials(net, R, mmf_mode=mmf_mode, mmf_amp=mmf_amp)
        idx = {nid: i for i, nid in enumerate(node_ids)}

    return psi, node_ids, edge_pairs


def lumped_torque_proxy_saturated(net: LumpedNetwork, mmf_amp: float = 200.0):
    """Saturated torque proxy: (W_d, W_q, T_proxy = W_d - W_q)."""
    from .torque import LumpedTorqueResult  # local import; avoids cycle

    psi_d, _, edges_d = solve_with_saturation(net, mmf_mode="d", mmf_amp=mmf_amp)
    W_d = coenergy(psi_d, edges_d)
    psi_q, _, edges_q = solve_with_saturation(net, mmf_mode="q", mmf_amp=mmf_amp)
    W_q = coenergy(psi_q, edges_q)
    return LumpedTorqueResult(W_d=W_d, W_q=W_q, T_proxy=W_d - W_q)
