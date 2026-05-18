"""Static reluctance-network solver.

Treats the lumped graph as a magnetic Laplacian: at each node ψ is the
magnetic scalar potential, each edge contributes admittance `1/R` to the
discrete Poisson operator. Dirichlet boundary conditions are applied at
the q-axis boundary nodes (ψ = 0) and at the stator-tooth nodes
(ψ = applied MMF profile). The remaining nodes' ψ values are solved from
KCL.

The magnetic coenergy `W = 0.5 · Σ (ψ_u − ψ_v)² / R_uv` is the natural
energy functional and serves as our `L ∝ W` proxy for d-axis / q-axis
inductance comparisons (CLAUDE.md §4, M1 priority note).
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve

from .network import LumpedNetwork


def _shaft_node_id(net: LumpedNetwork) -> str:
    """Single ground node (the rotor shaft).

    With Option A BC, only one reference is needed for system invertibility.
    The d-mode MMF naturally vanishes at the q-axes (sin = 0 there) so no
    extra boundary pinning is required; the q-mode would conflict with a
    ψ=0 boundary at the q-axes (cos = ±1 there) and so is left free —
    relaxing the boundary here means the q-mode coenergy reflects the actual
    network response, not the BC artefact.
    """
    for nid, d in net.graph.nodes(data=True):
        if d["kind"] == "shaft":
            return nid
    # Fallback: pick the first node.
    return next(iter(net.graph.nodes()))


def _tooth_node_ids(net: LumpedNetwork) -> list[str]:
    """IDs of stator-tooth nodes, sorted by angle."""
    teeth = [(nid, d["angle_deg"]) for nid, d in net.graph.nodes(data=True) if d["kind"] == "tooth"]
    teeth.sort(key=lambda t: t[1])
    return [t[0] for t in teeth]


def _airgap_node_ids(net: LumpedNetwork) -> list[str]:
    """IDs of airgap nodes, sorted by angle."""
    ag = [(nid, d["angle_deg"]) for nid, d in net.graph.nodes(data=True) if d["kind"] == "airgap"]
    ag.sort(key=lambda t: t[1])
    return [t[0] for t in ag]


def _mmf_profile(angles_deg: np.ndarray, mode: str, amp: float = 1.0) -> np.ndarray:
    """Spatial MMF profile across one pole sector [0°, 90°].

    `mode="d"`: half-sine peaking at the d-axis (45°). Vanishes at both
    q-axes — consistent with the ψ=0 Dirichlet boundary, so this is the
    canonical "L_d" excitation.

    `mode="q"`: half-cosine, +1 at θ=0° and −1 at θ=90°. Inconsistent with
    the boundary (the BC then suppresses the boundary values), but the
    resulting coenergy is a monotone-related proxy for q-axis inductance
    across designs.
    """
    rad = np.deg2rad(angles_deg)
    if mode == "d":
        return amp * np.sin(rad * 2.0)        # sin(θ·π/90) for θ in deg
    if mode == "q":
        return amp * np.cos(rad * 2.0)
    raise ValueError(f"unknown MMF mode: {mode!r}")


def assemble_admittance(net: LumpedNetwork, R_per_edge: dict[tuple[str, str], float]):
    """Return (node_ids, Y, edge_pairs) where Y is the dense magnetic Laplacian
    (sparse CSR), `node_ids` lists node id strings in matrix-index order, and
    `edge_pairs` is `[(i, j, R_uv), …]` for coenergy computation.
    """
    node_ids = list(net.graph.nodes())
    idx = {nid: i for i, nid in enumerate(node_ids)}
    n = len(node_ids)
    Y = lil_matrix((n, n), dtype=float)
    edge_pairs: list[tuple[int, int, float]] = []
    for (u, v), R in R_per_edge.items():
        if R <= 0.0:
            continue
        i = idx[u]
        j = idx[v]
        g = 1.0 / R
        Y[i, i] += g
        Y[j, j] += g
        Y[i, j] -= g
        Y[j, i] -= g
        edge_pairs.append((i, j, R))
    return node_ids, Y.tocsr(), edge_pairs


def solve_potentials(
    net: LumpedNetwork,
    R_per_edge: dict[tuple[str, str], float],
    mmf_mode: str,
    mmf_amp: float = 1.0,
) -> tuple[np.ndarray, list[str], list[tuple[int, int, float]]]:
    """Solve for the magnetic scalar potential ψ at every node.

    Returns (psi, node_ids, edge_pairs):
    - `psi[i]` is the potential at `node_ids[i]`
    - `edge_pairs[k] = (i, j, R)` for coenergy computation downstream
    """
    node_ids, Y, edge_pairs = assemble_admittance(net, R_per_edge)
    n = len(node_ids)
    idx = {nid: i for i, nid in enumerate(node_ids)}

    # Apply the MMF wave as a *current injection* at the airgap nodes —
    # the discrete-Laplacian dual of "tangential MMF source between
    # consecutive slots". The total flux entering the airgap-half from the
    # stator side is forced to equal the spatial profile, which is the
    # physically correct excitation for the rotor magnetic circuit.
    # The system needs one Dirichlet point to be invertible; we anchor at
    # the shaft (the physical rotor ground).
    shaft_id = _shaft_node_id(net)
    ag_ids = _airgap_node_ids(net)
    ag_angles = np.array([net.graph.nodes[t]["angle_deg"] for t in ag_ids])
    ag_mmf = _mmf_profile(ag_angles, mode=mmf_mode, amp=mmf_amp)

    psi = np.zeros(n)
    fixed: dict[int, float] = {idx[shaft_id]: 0.0}
    rhs_extra = np.zeros(n)
    for nid, m in zip(ag_ids, ag_mmf):
        rhs_extra[idx[nid]] += float(m)

    free_mask = np.ones(n, dtype=bool)
    for i in fixed:
        free_mask[i] = False
        psi[i] = fixed[i]
    free_idx = np.where(free_mask)[0]

    if free_idx.size == 0:
        return psi, node_ids, edge_pairs

    Y_ff = Y[free_idx][:, free_idx]
    Y_fd = Y[free_idx][:, ~free_mask]
    psi_dirichlet = psi[~free_mask]

    rhs = -Y_fd @ psi_dirichlet + rhs_extra[free_idx]
    psi_free = spsolve(Y_ff.tocsc(), rhs)
    psi[free_idx] = psi_free
    return psi, node_ids, edge_pairs


def coenergy(psi: np.ndarray, edge_pairs: list[tuple[int, int, float]]) -> float:
    """Magnetic coenergy `W = 0.5 · Σ (ψ_u − ψ_v)² / R_uv`."""
    if not edge_pairs:
        return 0.0
    pairs = np.array(edge_pairs)
    i = pairs[:, 0].astype(int)
    j = pairs[:, 1].astype(int)
    R = pairs[:, 2]
    return 0.5 * float(np.sum((psi[i] - psi[j]) ** 2 / R))
