"""Per-edge cross-section (v2).

`edge_cross_section_m2(net, u, v, d)` returns the perpendicular cross-section
in m² for an edge. v2 replaces the constant `EDGE_PERP_WIDTH_M` table with
geometry-driven values:

- **Airgap / barrier** edges (radial): width = local angular extent × radius.
- **Within-channel iron** edges (tangential, between two fluxtubes in the
  same channel): width = local iron-pocket radial thickness at the edge
  midpoint, derived from the barrier-bracketed extent at that angle.
- **Surface ↔ channel** iron edges (radial, short hop from surface arc to
  channel midline): width = small angular slice × surface radius.
- **Surface rim, yoke ring, tooth ring** (tangential): width = radial
  thickness of the iron strip carrying tangential flux.
- **Yoke ↔ tooth** (radial tooth body): width = tooth body width.
- **Shaft link** (radial through inner rotor): width = inner channel
  angular extent × mean radius (multiple parallel paths).

Anything not classified falls back to `DEFAULT_PERP_WIDTH_M`.
"""

from __future__ import annotations

import math

import numpy as np

from .material import DEFAULT_PERP_WIDTH_M
from .network import LumpedNetwork


def _node_angle(net: LumpedNetwork, nid: str) -> float:
    d = net.graph.nodes[nid]
    a = d.get("angle_deg")
    if a is not None:
        return float(a)
    return math.degrees(math.atan2(d["y"], d["x"]))


def _node_radius(net: LumpedNetwork, nid: str) -> float:
    d = net.graph.nodes[nid]
    return float(math.hypot(d["x"], d["y"]))


def _barrier_radii_at_angle(net: LumpedNetwork, theta_deg: float) -> list[tuple[float, float] | None]:
    """Return per-barrier (r_in, r_out) at this angle (or None where the
    barrier doesn't cross). Re-uses network's `_barrier_crossings` to stay
    consistent with how barriers are interpreted elsewhere.
    """
    from .network import _barrier_crossings  # local import to avoid cycle at module load
    return _barrier_crossings(net.barrier_polylines, theta_deg, net.spec.rotor_r_max)


def _channel_iron_thickness_mm(net: LumpedNetwork, channel: int, theta_deg: float) -> float:
    """Radial thickness (mm) of the iron pocket bracketing `channel` at angle θ.

    Below bound: `r_min` (channel 0) or barrier (c-1)'s outer-bezier crossing.
    Above bound: `r_max` (channel N) or barrier c's inner-bezier crossing.
    A barrier that doesn't extend gets substituted with `r_max` (same ghost
    rule as in `network._channel_node_radii_raw_at_angle`).
    """
    spec = net.spec
    n = len(net.barrier_polylines)
    if n == 0:
        return spec.rotor_r_max - spec.rotor_r_min
    crossings = _barrier_radii_at_angle(net, theta_deg)
    r_in_b = [spec.rotor_r_max if c is None else c[0] for c in crossings]
    r_out_b = [spec.rotor_r_max if c is None else c[1] for c in crossings]
    below = spec.rotor_r_min if channel == 0 else r_out_b[channel - 1]
    above = spec.rotor_r_max if channel == n else r_in_b[channel]
    return max(0.1, float(above - below))   # guard against degenerate widths


def edge_cross_section_m2(net: LumpedNetwork, u: str, v: str, d: dict) -> float:
    """Perpendicular cross-section (m²) for the edge `(u, v)`."""
    spec = net.spec
    stack_m = spec.stack_length * 1e-3
    kind = d["kind"]
    nu = net.graph.nodes[u]
    nv = net.graph.nodes[v]

    # Common helpers
    ang_u = _node_angle(net, u)
    ang_v = _node_angle(net, v)
    r_u = _node_radius(net, u)
    r_v = _node_radius(net, v)
    r_mid_mm = 0.5 * (r_u + r_v)
    ang_mid_deg = 0.5 * (ang_u + ang_v)
    slot_pitch_rad = math.radians(360.0 / spec.n_slots)
    pole_span_rad = math.radians(90.0)

    if kind == "iron_yoke":
        # Tangential through the stator yoke. Width = yoke radial height.
        yoke_height_mm = spec.stator_yoke_r_outer - (spec.stator_r_inner + 0.25 * (spec.stator_yoke_r_outer - spec.stator_r_inner))
        width_mm = max(2.0, yoke_height_mm * 0.6)   # yoke iron is ~60% of the yoke region
        return width_mm * 1e-3 * stack_m

    if kind == "iron_tooth":
        # Slot-leakage path between consecutive teeth. Width ≈ slot opening width
        # × half-tooth-height — conservative narrow value.
        return 1.5e-3 * stack_m

    if kind == "yoke_to_tooth":
        # Tooth body (radial). Width = approximate tooth angular pitch − slot opening.
        # Tooth body is roughly half the slot pitch arc.
        arc_mm = 0.5 * slot_pitch_rad * r_mid_mm
        return max(2.0e-3, arc_mm * 1e-3) * stack_m

    if kind == "shaft_link":
        # Radial through the inner rotor; the shaft connects to all rail-column
        # inner-channel nodes (a star). Width ≈ inner channel angular extent × r.
        # Use one third of the pole's angular extent (because there are 3 star arms).
        arc_mm = (pole_span_rad / 3.0) * r_mid_mm
        return max(2.0e-3, arc_mm * 1e-3) * stack_m

    if kind == "airgap":
        # Radial across (a half of) the airgap. Width = one slot-pitch arc at
        # the airgap radius, so each tooth couples to its angular share of
        # airgap.
        arc_mm = slot_pitch_rad * r_mid_mm
        return arc_mm * 1e-3 * stack_m

    if kind == "barrier":
        # Radial across a rotor barrier. Width = local barrier angular extent
        # at this rail's angle. For Hackl barriers, the inner-bezier and
        # outer-bezier converge near r_max at the angular extent endpoints, so
        # the local barrier "width" depends on θ. A simple proxy: barrier
        # angular extent ≈ (phi_outer − phi_inner) at this rail, weighted by
        # depth. Lacking generator metadata, use a conservative default
        # proportional to barrier polyline angular range / number_of_barriers.
        bridx = d.get("barrier_index", 0)
        try:
            poly = net.barrier_polylines[bridx]
            ang = np.rad2deg(np.arctan2(poly[:, 1], poly[:, 0]))
            angular_extent_rad = math.radians(float(ang.max() - ang.min())) / max(1.0, float(len(net.barrier_polylines)))
        except Exception:
            angular_extent_rad = pole_span_rad / 9.0   # ~10° fallback
        arc_mm = angular_extent_rad * r_mid_mm
        return max(2.0e-3, arc_mm * 1e-3) * stack_m

    if kind == "iron_surface":
        # Tangential just-inside-surface rim. Width ≈ the thin shell between
        # the bezier endpoint radius `R` and r_max. For Hackl with r_stator_end
        # = 0.7 mm, R ≈ r_max - 0.7, so the rim is ~0.5–1 mm.
        return 1.0e-3 * stack_m

    if kind == "iron_rotor":
        # Two sub-cases: within-channel (both nodes are fluxtubes in same
        # channel) vs surface↔channel hop.
        is_u_fluxtube = nu["kind"] == "fluxtube"
        is_v_fluxtube = nv["kind"] == "fluxtube"
        if is_u_fluxtube and is_v_fluxtube:
            ch_u = nu.get("channel")
            ch_v = nv.get("channel")
            if ch_u is not None and ch_u == ch_v:
                # Within-channel: width = local iron-pocket radial thickness
                thickness_mm = _channel_iron_thickness_mm(net, int(ch_u), ang_mid_deg)
                return thickness_mm * 1e-3 * stack_m
            # Cross-channel iron at q-axes (channel 0 ↔ channel N when no barriers)
            arc_mm = (pole_span_rad / 6.0) * r_mid_mm   # local angular share
            return max(2.0e-3, arc_mm * 1e-3) * stack_m
        # Surface ↔ channel: short radial hop. Width = local angular slice.
        arc_mm = (pole_span_rad / max(1.0, float(net.granularity.n_col))) * r_mid_mm
        return max(2.0e-3, arc_mm * 1e-3) * stack_m

    return DEFAULT_PERP_WIDTH_M * stack_m
