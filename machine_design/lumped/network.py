"""Lumped-reluctance graph topology.

Builds the graph (nodes + edges + geometry) over a single pole sector of the
reference SynRM, at a chosen `Granularity`. No reluctance values, no MMF
values, no solve — those land in `solve.py` / `torque.py` (M1).

Design principles
-----------------
- **Nodes live inside iron.** Rotor flux-tube nodes are placed at the radial
  mid-point of the iron interval at their angular column. When barriers
  don't reach a given column, two adjacent channels share the same iron
  interval and their flux-tube nodes coincide (zero-length cross-barrier
  edge with `barrier_crossings = []` — M1 treats it as iron).
- **Edges follow iron paths.** Within-channel edges are polylines that
  trace the channel midline; cross-channel and airgap–rotor edges are
  purely radial single-column segments. Both produce honest crossing
  lists when intersected with barrier polylines.
- **Shared angular columns.** Airgap and rotor share `n_col` columns,
  so the airgap→rotor link at column k is the radial segment
  `airgap[k] → fluxtube[outer][k]` — no angular offset, no spurious
  barrier hit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import networkx as nx
import numpy as np
from scipy.ndimage import gaussian_filter1d
from shapely.geometry import LineString

from .geometry import MachineSpec
from .granularity import Granularity


NodeKind = str   # "yoke" | "tooth" | "airgap" | "surface" | "fluxtube" | "shaft"
EdgeKind = str   # "iron_yoke" | "iron_tooth" | "iron_rotor" | "iron_surface"
                 # | "yoke_to_tooth" | "airgap" | "barrier" | "shaft_link"


@dataclass(frozen=True)
class Node:
    id: str
    kind: NodeKind
    x: float
    y: float
    meta: dict = field(default_factory=dict)


@dataclass
class LumpedNetwork:
    spec: MachineSpec
    granularity: Granularity
    graph: nx.Graph
    pole_sector_deg: tuple[float, float] = (0.0, 90.0)
    barrier_polylines: list[np.ndarray] = field(default_factory=list)

    def nodes_by_kind(self, kind: NodeKind) -> list[str]:
        return [n for n, d in self.graph.nodes(data=True) if d["kind"] == kind]

    def edges_by_kind(self, kind: EdgeKind) -> list[tuple[str, str]]:
        return [(u, v) for u, v, d in self.graph.edges(data=True) if d["kind"] == kind]

    def topology_mismatches(self) -> dict[str, int]:
        """Counts of edges whose `kind` contradicts the geometric crossings.

        With the iron-aware topology, the only expected mismatch is
        `barrier_no_cross` at columns where a barrier doesn't extend — those
        are correctly-classified-as-iron edges that we still call "barrier"
        for the 2-D grid topology to remain uniform.
        """
        counts = {"iron_should_be_air": 0, "barrier_no_cross": 0, "airgap_crosses": 0}
        for _, _, d in self.graph.edges(data=True):
            crossings = d.get("barrier_crossings", [])
            kind = d["kind"]
            if kind == "barrier" and not crossings:
                counts["barrier_no_cross"] += 1
            elif (kind.startswith("iron_") or kind in ("yoke_to_tooth", "shaft_link")) and crossings:
                counts["iron_should_be_air"] += 1
            elif kind == "airgap" and crossings:
                counts["airgap_crosses"] += 1
        return counts


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _polar_to_xy(r: float, deg: float) -> tuple[float, float]:
    rad = np.deg2rad(deg)
    return r * np.cos(rad), r * np.sin(rad)


def _evenly_spaced_angles(n: int, span: tuple[float, float]) -> np.ndarray:
    a, b = span
    if n <= 0:
        return np.empty(0)
    return a + (b - a) * (np.arange(n) + 0.5) / n


def _ring_nodes(
    prefix: str,
    kind: NodeKind,
    radius: float,
    angles_deg: Iterable[float],
    meta_per_node: Iterable[dict] | None = None,
) -> list[Node]:
    angles = list(angles_deg)
    metas = list(meta_per_node) if meta_per_node is not None else [{} for _ in angles]
    out: list[Node] = []
    for i, (deg, m) in enumerate(zip(angles, metas)):
        x, y = _polar_to_xy(radius, deg)
        out.append(Node(id=f"{prefix}{i}", kind=kind, x=x, y=y, meta={"angle_deg": float(deg), **m}))
    return out


# ---------------------------------------------------------------------------
# Barrier crossings on a radial ray
# ---------------------------------------------------------------------------


def _barrier_crossings(
    barrier_polylines: list[np.ndarray] | None, theta_deg: float, r_max: float
) -> list[tuple[float, float] | None]:
    """For each barrier, return the `(r_inner, r_outer)` crossings of the
    radial ray at `theta_deg`, or None if the barrier doesn't cross.
    """
    if not barrier_polylines:
        return []
    rad = np.deg2rad(theta_deg)
    far = 1.5 * r_max
    ray = LineString([(0.0, 0.0), (far * np.cos(rad), far * np.sin(rad))])
    out: list[tuple[float, float] | None] = []
    for poly in barrier_polylines:
        try:
            bline = LineString(poly)
        except Exception:
            out.append(None)
            continue
        inter = ray.intersection(bline)
        if inter.is_empty:
            out.append(None)
            continue
        radii: list[float] = []
        if inter.geom_type == "Point":
            radii = [float(np.hypot(inter.x, inter.y))]
        elif inter.geom_type == "MultiPoint":
            radii = sorted(float(np.hypot(p.x, p.y)) for p in inter.geoms)
        else:
            xs, ys = (inter.xy if hasattr(inter, "xy") else ([], []))
            radii = sorted(float(np.hypot(x, y)) for x, y in zip(xs, ys))
        if len(radii) >= 2:
            out.append((radii[0], radii[-1]))
        elif len(radii) == 1:
            out.append((radii[0], radii[0]))
        else:
            out.append(None)
    return out


def _channel_node_radii_raw_at_angle(
    barrier_polylines: list[np.ndarray] | None,
    theta_deg: float,
    r_min: float,
    r_max: float,
) -> list[float]:
    """Per-channel midradii at `theta_deg` directly from barrier geometry.

    For each channel c:
    - `below` = r_min if c == 0, else barrier (c-1)'s outer-bezier value at θ
    - `above` = r_max if c == N, else barrier c's inner-bezier value at θ
    - midradius = (below + above) / 2

    Where a barrier doesn't extend at this angle (`_barrier_crossings`
    returns None), substitute `r_max` for the missing bezier — i.e., the
    barrier "ghosts" out to the rotor-surface arc at the angular extent
    boundary. This produces hyperbolic channel midlines that mirror the
    barrier shapes: closest approach at the d-axis, diverging toward `r_max`
    at the q-axes (where barriers terminate at the rotor surface).
    """
    n = len(barrier_polylines) if barrier_polylines else 0
    n_channels = n + 1
    if n == 0:
        return [0.5 * (r_min + r_max)]

    crossings = _barrier_crossings(barrier_polylines, theta_deg, r_max)
    # `c[0]` = inner-bezier intersection (closer to shaft);
    # `c[1]` = outer-bezier intersection (closer to surface).
    # `None` → both substituted with `r_max` (bezier endpoints on the surface arc).
    r_in_b = [r_max if c is None else c[0] for c in crossings]
    r_out_b = [r_max if c is None else c[1] for c in crossings]

    radii: list[float] = []
    for ch in range(n_channels):
        below = r_min if ch == 0 else r_out_b[ch - 1]
        above = r_max if ch == n_channels - 1 else r_in_b[ch]
        radii.append(0.5 * (below + above))
    return radii


# ---------------------------------------------------------------------------
# Edge–barrier crossing detection on an arbitrary polyline
# ---------------------------------------------------------------------------


def _polyline_barrier_crossings(
    points: list[tuple[float, float]],
    barrier_polylines: list[np.ndarray] | None,
) -> list[int]:
    """Indices of barriers whose polyline the edge polyline intersects."""
    if not barrier_polylines or len(points) < 2:
        return []
    eline = LineString(points)
    hits: list[int] = []
    for i, poly in enumerate(barrier_polylines):
        try:
            bline = LineString(poly)
        except Exception:
            continue
        if not eline.intersection(bline).is_empty:
            hits.append(i)
    return hits


def _annotate_edges_with_crossings(g: nx.Graph, barrier_polylines: list[np.ndarray] | None) -> None:
    """Stamp every edge with `barrier_crossings: list[int]` derived from the
    actual segment / polyline geometry. M1's solver should use this attribute
    (not the constructor-time `kind`) to decide iron vs. air reluctance.
    """
    pos = {n: (d["x"], d["y"]) for n, d in g.nodes(data=True)}
    for u, v, d in g.edges(data=True):
        polyline = d.get("polyline")
        if polyline is not None and len(polyline) >= 2:
            pts = [(float(p[0]), float(p[1])) for p in polyline]
        else:
            pts = [pos[u], pos[v]]
        d["barrier_crossings"] = _polyline_barrier_crossings(pts, barrier_polylines)


# ---------------------------------------------------------------------------
# Within-channel midline computation
# ---------------------------------------------------------------------------


# `_channel_midline` removed: within-channel edges are simple chords between
# consecutive column endpoints, so the per-edge geometry is just the two node
# positions and no intermediate sampling. This avoids the artificial radius
# jumps that the dense intermediate sampling exposed near barrier endpoints.


# Smooth per-column radii alias: the raw barrier-derived radii (with merged
# pockets distributed evenly) are already smooth within any barrier-extent
# region and stable across the q-axis fallback. We use the raw function
# directly for every column.
_channel_node_radii_at_angle = _channel_node_radii_raw_at_angle


def _channel_angular_extent(
    channel: int,
    n_channels: int,
    barrier_polylines: list[np.ndarray] | None,
    r_min: float | None = None,
    r_max: float | None = None,
    *,
    surface_tol: float = 0.5,
    n_dense: int = 400,
) -> tuple[float, float]:
    """Angular extent `(theta_lo, theta_hi)` where `channel`'s midline is
    **inside** the iron (not riding the rotor surface).

    - Channel 0 (innermost) and channel N (outermost) always span the full
      pole sector `[0°, 90°]` — their non-barrier bound (`r_min` / `r_max`)
      keeps their midline well-defined throughout.
    - Middle channels' extent is the angular range where the midline radius
      is below `r_max − surface_tol`. Outside this range, both bracketing
      barriers' beziers are at (or beyond) their endpoints and the
      substituted-`r_max` formula gives midline ≈ `r_max` — the "riding the
      surface" tail. Truncating there lets the last interior node connect
      directly to the nearest surface arc node (a single clean iron edge
      from the interior to the surface), instead of running parallel to
      the surface along an arc.
    """
    if channel == 0 or channel == n_channels - 1 or not barrier_polylines:
        return (0.0, 90.0)
    if r_min is None or r_max is None:
        raise ValueError("r_min and r_max required for middle channels")
    dense_angles = np.linspace(0.0, 90.0, n_dense)
    dense_radii = np.array(
        [
            _channel_node_radii_raw_at_angle(barrier_polylines, float(a), r_min, r_max)[channel]
            for a in dense_angles
        ]
    )
    inside = dense_radii < r_max - surface_tol
    if not inside.any():
        # Pathological: channel never leaves the surface. Pin to d-axis as a single point.
        return (45.0, 45.0)
    idx = np.where(inside)[0]
    return float(dense_angles[idx[0]]), float(dense_angles[idx[-1]])


def _channel_smooth_midline(
    channel: int,
    barrier_polylines: list[np.ndarray] | None,
    r_min: float,
    r_max: float,
    n_dense: int = 400,
    sigma: float = 0.0,  # smoothing no longer needed
) -> tuple[np.ndarray, np.ndarray]:
    """Dense channel midline samples for arc-length distribution.

    With the blend-anchor algorithm in `_channel_node_radii_raw_at_angle`,
    midradii are already smooth (no per-angle iron-pocket reallocation), so
    `sigma=0` is the default — no Gaussian filter applied. Kept for
    backwards compatibility if a future channel-shape model needs it.
    """
    angles = np.linspace(0.0, 90.0, n_dense)
    radii = np.array(
        [
            _channel_node_radii_raw_at_angle(barrier_polylines, float(a), r_min, r_max)[channel]
            for a in angles
        ]
    )
    if sigma > 0.0:
        radii = gaussian_filter1d(radii, sigma=sigma, mode="nearest")
    return angles, radii


def _channel_column_angles_by_arclen(
    channel: int,
    n_channels: int,
    n_col: int,
    barrier_polylines: list[np.ndarray] | None,
    r_min: float,
    r_max: float,
    n_dense: int = 400,
) -> tuple[np.ndarray, np.ndarray]:
    """Column angles AND radii for `channel`, distributed evenly in arc length
    along the channel midline, pinned at the d-axis (index `n_half = (n_col-1)//2`).

    For channels 0 and N (boundary channels), the extent is `[0°, 90°]` and
    the three rails (0°, 45°, 90°) are anchored at indices 0, n_half, n_col-1.

    For middle channels, the extent is truncated to where the bracketing
    barriers extend (the upper barrier's polyline angular range). Endpoints
    are pinned at the extent boundaries; only the d-axis anchor is shared
    with the global 45° rail.
    """
    if n_col < 3 or n_col % 2 == 0:
        raise ValueError(f"n_col must be odd and >= 3, got {n_col}")

    theta_lo, theta_hi = _channel_angular_extent(
        channel, n_channels, barrier_polylines, r_min=r_min, r_max=r_max
    )
    n_half = (n_col - 1) // 2

    # Dense sampling restricted to the channel's extent.
    dense_angles = np.linspace(theta_lo, theta_hi, n_dense)
    dense_radii = np.array(
        [
            _channel_node_radii_raw_at_angle(barrier_polylines, float(a), r_min, r_max)[channel]
            for a in dense_angles
        ]
    )
    rads = np.deg2rad(dense_angles)
    dense_pts = np.stack([dense_radii * np.cos(rads), dense_radii * np.sin(rads)], axis=1)

    segs = np.diff(dense_pts, axis=0)
    lengths = np.hypot(segs[:, 0], segs[:, 1])
    cumlen = np.concatenate([[0.0], np.cumsum(lengths)])
    total = float(cumlen[-1])

    if theta_lo <= 45.0 <= theta_hi:
        arc_at_45 = float(np.interp(45.0, dense_angles, cumlen))
    else:
        arc_at_45 = 0.5 * total

    targets_lo = np.linspace(0.0, arc_at_45, n_half + 1)
    targets_hi = np.linspace(arc_at_45, total, n_half + 1)[1:]
    targets = np.concatenate([targets_lo, targets_hi])

    angles = np.interp(targets, cumlen, dense_angles)
    radii_out = np.interp(targets, cumlen, dense_radii)

    # Pin anchors at the boundaries of the channel's extent and at d-axis.
    angles[0] = theta_lo
    angles[-1] = theta_hi
    if theta_lo <= 45.0 <= theta_hi:
        angles[n_half] = 45.0
    for j in (0, n_half, n_col - 1):
        radii_out[j] = float(np.interp(angles[j], dense_angles, dense_radii))
    return angles, radii_out


# ---------------------------------------------------------------------------
# Network construction
# ---------------------------------------------------------------------------


def build_network(
    spec: MachineSpec,
    granularity: Granularity,
    barrier_polylines: list[np.ndarray] | None = None,
) -> LumpedNetwork:
    """Build the lumped-reluctance graph for one pole sector.

    The number of rotor channels equals `len(barrier_polylines) + 1`
    (defaults to 4 = three Hackl barriers + 1).
    """
    g = nx.Graph()
    pole_span = (0.0, 90.0)

    # --- Shared angular columns (airgap + surface + rotor) ---------------
    # Endpoint-inclusive sampling. n_col is odd, so the rails (0°, 45°, 90°)
    # are sampled exactly at indices 0, (n_col-1)//2, n_col-1.
    col_angles = np.linspace(pole_span[0], pole_span[1], granularity.n_col)
    rail_indices = (0, (granularity.n_col - 1) // 2, granularity.n_col - 1)

    n_barriers = len(barrier_polylines) if barrier_polylines is not None else 0
    n_channels = n_barriers + 1

    # --- Per-channel column angles + radii (arc-length within extent) ----
    # Boundary channels (0 and N) span [0°, 90°]; middle channels are
    # truncated to their barrier-defined extent so they don't merge into
    # the rotor surface at the q-axes.
    channel_col_data: list[tuple[np.ndarray, np.ndarray]] = [
        _channel_column_angles_by_arclen(
            ch, n_channels, granularity.n_col, barrier_polylines,
            spec.rotor_r_min, spec.rotor_r_max,
        )
        for ch in range(n_channels)
    ]

    # --- Stator yoke ring -------------------------------------------------
    r_yoke = 0.5 * (spec.stator_r_inner + spec.stator_yoke_r_outer)
    yoke_angles = _evenly_spaced_angles(granularity.n_yoke, pole_span)
    yoke_nodes = _ring_nodes("Y", "yoke", r_yoke, yoke_angles)

    # --- Stator teeth ring (fixed by motor design at n_teeth) -------------
    r_tooth = spec.stator_r_inner + 0.25 * (spec.stator_yoke_r_outer - spec.stator_r_inner)
    tooth_angles = _evenly_spaced_angles(granularity.n_teeth, pole_span)
    phase_tags = ["A", "B", "C"]
    tooth_metas = [{"phase": phase_tags[i % 3]} for i in range(granularity.n_teeth)]
    tooth_nodes = _ring_nodes("T", "tooth", r_tooth, tooth_angles, tooth_metas)

    # --- Airgap ring (one per column) -------------------------------------
    r_airgap = 0.5 * (spec.rotor_r_max + spec.stator_r_inner)
    airgap_nodes = _ring_nodes("G", "airgap", r_airgap, col_angles)

    # --- Surface arc (just inside rotor_r_max, one per column) -----------
    # Small radial offset so surface nodes sit visibly inside the rotor; the
    # surface arc is the rotor's near-skin iron rim that ties together the
    # outer ends of the channel curves and couples to the airgap.
    r_surface = spec.rotor_r_max - 0.01 * (spec.rotor_r_max - spec.rotor_r_min)
    surface_nodes = _ring_nodes("U", "surface", r_surface, col_angles)

    # --- Rotor flux-tube grid: per-channel arc-length-uniform sampling ----
    fluxtube_grid: list[list[Node]] = [[] for _ in range(n_channels)]
    for ch in range(n_channels):
        col_angles_ch, col_radii_ch = channel_col_data[ch]
        for k, (deg, r) in enumerate(zip(col_angles_ch, col_radii_ch)):
            x, y = _polar_to_xy(float(r), float(deg))
            fluxtube_grid[ch].append(
                Node(
                    id=f"F{ch}_{k}",
                    kind="fluxtube",
                    x=x,
                    y=y,
                    meta={"channel": ch, "angle_deg": float(deg), "r": float(r)},
                )
            )
    fluxtube_nodes = [n for row in fluxtube_grid for n in row]

    # --- Shaft -----------------------------------------------------------
    r_shaft_mid = 0.5 * spec.rotor_r_min
    sx, sy = _polar_to_xy(r_shaft_mid, 45.0)
    shaft_node = Node(id="S0", kind="shaft", x=sx, y=sy)

    # --- Register all nodes ----------------------------------------------
    all_nodes = [*yoke_nodes, *tooth_nodes, *airgap_nodes, *surface_nodes, *fluxtube_nodes, shaft_node]
    for n in all_nodes:
        g.add_node(n.id, kind=n.kind, x=n.x, y=n.y, **n.meta)

    # --- Stator edges ----------------------------------------------------
    for i in range(len(yoke_nodes) - 1):
        g.add_edge(yoke_nodes[i].id, yoke_nodes[i + 1].id, kind="iron_yoke")

    if yoke_nodes:
        yoke_ang = np.array([n.meta["angle_deg"] for n in yoke_nodes])
        for t in tooth_nodes:
            j = int(np.argmin(np.abs(yoke_ang - t.meta["angle_deg"])))
            g.add_edge(t.id, yoke_nodes[j].id, kind="yoke_to_tooth")

    for i in range(len(tooth_nodes) - 1):
        g.add_edge(tooth_nodes[i].id, tooth_nodes[i + 1].id, kind="iron_tooth")

    # --- Airgap stator half: tooth ↔ airgap at angular-nearest tooth -----
    if tooth_nodes and airgap_nodes:
        tooth_ang = np.array([n.meta["angle_deg"] for n in tooth_nodes])
        for ag in airgap_nodes:
            j = int(np.argmin(np.abs(tooth_ang - ag.meta["angle_deg"])))
            g.add_edge(ag.id, tooth_nodes[j].id, kind="airgap")

    # --- Airgap rotor half: airgap[k] ↔ surface[k] (pure radial) ---------
    for k in range(len(col_angles)):
        g.add_edge(airgap_nodes[k].id, surface_nodes[k].id, kind="airgap")

    # --- Surface arc consecutive iron edges (near-surface rim) -----------
    for k in range(len(surface_nodes) - 1):
        g.add_edge(surface_nodes[k].id, surface_nodes[k + 1].id, kind="iron_surface")

    # --- Surface ↔ outermost channel midline (angular-nearest) -----------
    # Each surface node ties to the angular-nearest outermost-channel
    # flux-tube node.
    outer_row = fluxtube_grid[-1] if fluxtube_grid else []
    if outer_row:
        outer_ang = np.array([n.meta["angle_deg"] for n in outer_row])
        for s in surface_nodes:
            j = int(np.argmin(np.abs(outer_ang - s.meta["angle_deg"])))
            g.add_edge(s.id, outer_row[j].id, kind="iron_rotor")

    # --- Surface ↔ middle-channel endpoints ------------------------------
    # Truncated middle channels end near the rotor surface (their last node
    # sits on or just below R, where the bracketing barrier terminates).
    # Tie each endpoint to the angular-nearest surface node so the channel
    # has a path back into the airgap circuit at its terminus.
    if surface_nodes:
        surf_ang = np.array([s.meta["angle_deg"] for s in surface_nodes])
        for ch in range(1, n_channels - 1):
            row = fluxtube_grid[ch]
            if not row:
                continue
            for endpoint in (row[0], row[-1]):
                j = int(np.argmin(np.abs(surf_ang - endpoint.meta["angle_deg"])))
                g.add_edge(surface_nodes[j].id, endpoint.id, kind="iron_rotor")

    # --- Within-channel midline edges (chords between adjacent columns) --
    for ch in range(n_channels):
        row = fluxtube_grid[ch]
        for k in range(len(row) - 1):
            g.add_edge(row[k].id, row[k + 1].id, kind="iron_rotor")

    # --- Cross-channel edges at the three rails (0°, 45°, 90°) ----------
    # D-axis (middle rail): every channel has a node, so the chain goes
    # 0 ↔ 1 ↔ 2 ↔ ... ↔ N, crossing each barrier in turn.
    # Q-axes (first/last rails): only channels 0 and N have nodes there
    # (middle channels are truncated to their barrier extent). A single
    # iron edge runs directly from channel 0 to channel N through the
    # open iron.
    def _add_cross_edge(u_node: Node, v_node: Node) -> None:
        xings = _polyline_barrier_crossings(
            [(u_node.x, u_node.y), (v_node.x, v_node.y)], barrier_polylines
        )
        kind = "barrier" if xings else "iron_rotor"
        g.add_edge(u_node.id, v_node.id, kind=kind)

    mid_rail = rail_indices[1]
    for ch in range(n_channels - 1):
        _add_cross_edge(fluxtube_grid[ch][mid_rail], fluxtube_grid[ch + 1][mid_rail])

    if n_channels >= 2:
        for k in (rail_indices[0], rail_indices[2]):
            _add_cross_edge(fluxtube_grid[0][k], fluxtube_grid[-1][k])

    # --- Shaft ↔ innermost-channel rail-column nodes (star) --------------
    if fluxtube_grid:
        for k in rail_indices:
            g.add_edge(shaft_node.id, fluxtube_grid[0][k].id, kind="shaft_link")

    # --- Annotate every edge with its actual barrier crossings -----------
    _annotate_edges_with_crossings(g, barrier_polylines)

    return LumpedNetwork(
        spec=spec,
        granularity=granularity,
        graph=g,
        pole_sector_deg=pole_span,
        barrier_polylines=barrier_polylines or [],
    )
