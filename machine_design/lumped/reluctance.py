"""Per-edge reluctance computation.

`compute_edge_reluctances(net)` returns a dict `(u, v) → R` (Wb⁻¹ · A·turns)
for every edge in the lumped graph. The split between iron and air is
derived from the edge geometry intersected with the barrier polylines —
**not** from the constructor-time `kind` — so chord segments that clip a
barrier sliver pay the correct hybrid reluctance.

v1 uses a single default cross-section (`material.DEFAULT_PERP_WIDTH_M`)
for every edge. Per-edge geometric cross-sections are a v2 refinement.
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import LineString, Polygon

from .cross_section import edge_cross_section_m2
from .material import DEFAULT_PERP_WIDTH_M, EDGE_PERP_WIDTH_M, MU_0, MU_IRON
from .network import LumpedNetwork


def _edge_geometry(net: LumpedNetwork, u: str, v: str, d: dict) -> tuple[np.ndarray, float]:
    """Return (polyline, total_length) for an edge.

    Polyline-attached edges (e.g. channel midlines) use their stored polyline;
    everything else is a straight chord between endpoint coordinates.
    """
    poly = d.get("polyline")
    if poly is not None and len(poly) >= 2:
        pts = np.asarray(poly, dtype=float)
    else:
        nu = net.graph.nodes[u]
        nv = net.graph.nodes[v]
        pts = np.array([[nu["x"], nu["y"]], [nv["x"], nv["y"]]], dtype=float)
    segs = np.diff(pts, axis=0)
    length = float(np.hypot(segs[:, 0], segs[:, 1]).sum())
    return pts, length


def _iron_air_split(
    pts: np.ndarray,
    total_length: float,
    barrier_polygons: list[Polygon],
    kind: str,
) -> tuple[float, float]:
    """Decompose an edge's total length into (length_iron, length_air) in metres.

    Rules:
    - `airgap` and `barrier` edges: entirely air.
    - Other edges (iron kinds): iron, minus any portion inside a barrier polygon.
      The intersection is computed in mm (since node coordinates are in mm);
      we convert at the end to metres.
    """
    if total_length <= 0.0:
        return 0.0, 0.0

    if kind in ("airgap", "barrier"):
        return 0.0, total_length * 1e-3   # mm → m

    if not barrier_polygons or len(pts) < 2:
        return total_length * 1e-3, 0.0

    seg = LineString(pts)
    air_mm = 0.0
    for poly in barrier_polygons:
        try:
            inter = seg.intersection(poly)
        except Exception:
            continue
        if inter.is_empty:
            continue
        # `.length` is defined on LineString/MultiLineString and gives 0 for
        # Point/MultiPoint intersections (tangent touches).
        try:
            air_mm += float(getattr(inter, "length", 0.0))
        except Exception:
            continue

    air_mm = min(air_mm, total_length)
    iron_mm = total_length - air_mm
    return iron_mm * 1e-3, air_mm * 1e-3


def compute_edge_reluctances(
    net: LumpedNetwork,
    mu_iron_per_edge: dict[tuple[str, str], float] | None = None,
) -> dict[tuple[str, str], float]:
    """Return {(u, v): R [A·turns/Wb]} for every edge in the network.

    If `mu_iron_per_edge` is given, it overrides `material.MU_IRON` per edge
    (used by `saturation.solve_with_saturation` to set a B-dependent μ on
    iron edges).
    """
    spec = net.spec
    stack_m = spec.stack_length * 1e-3            # mm → m

    barrier_polygons: list[Polygon] = []
    for poly in net.barrier_polylines:
        if poly is None or len(poly) < 4:
            continue
        try:
            barrier_polygons.append(Polygon(poly))
        except Exception:
            continue

    out: dict[tuple[str, str], float] = {}
    for u, v, d in net.graph.edges(data=True):
        pts, length_mm = _edge_geometry(net, u, v, d)
        length_iron_m, length_air_m = _iron_air_split(pts, length_mm, barrier_polygons, d["kind"])
        # v2: per-edge geometric cross-section (m²) based on edge purpose.
        A = edge_cross_section_m2(net, u, v, d)
        if A <= 0.0:
            A = DEFAULT_PERP_WIDTH_M * stack_m
        if length_iron_m + length_air_m <= 1e-12:
            out[(u, v)] = 1e18
            continue
        mu_iron_eff = MU_IRON if mu_iron_per_edge is None else mu_iron_per_edge.get((u, v), MU_IRON)
        R = length_iron_m / (mu_iron_eff * A) + length_air_m / (MU_0 * A)
        out[(u, v)] = max(R, 1e-3)
    return out


def edge_geometry_lengths(net: LumpedNetwork) -> dict[tuple[str, str], tuple[float, float, float]]:
    """For each edge return (length_iron_m, length_air_m, A_m2).

    Used by the saturation solver to convert flux to B and update μ
    without re-running the polygon-intersection work.
    """
    spec = net.spec
    stack_m = spec.stack_length * 1e-3
    barrier_polygons: list[Polygon] = []
    for poly in net.barrier_polylines:
        if poly is None or len(poly) < 4:
            continue
        try:
            barrier_polygons.append(Polygon(poly))
        except Exception:
            continue

    out: dict[tuple[str, str], tuple[float, float, float]] = {}
    for u, v, d in net.graph.edges(data=True):
        pts, length_mm = _edge_geometry(net, u, v, d)
        length_iron_m, length_air_m = _iron_air_split(pts, length_mm, barrier_polygons, d["kind"])
        A = edge_cross_section_m2(net, u, v, d)
        if A <= 0.0:
            A = DEFAULT_PERP_WIDTH_M * stack_m
        out[(u, v)] = (length_iron_m, length_air_m, A)
    return out
