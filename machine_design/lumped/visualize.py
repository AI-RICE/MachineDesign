"""Publication-quality rendering of the lumped-reluctance graph.

Renders nodes (yoke, tooth, airgap, fluxtube, shaft) and edges (iron paths,
airgap, barriers, shaft link) on top of the rotor cross-section, restricted
to one pole sector (0°–90°).
"""

from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .network import LumpedNetwork


# --- Styling ---------------------------------------------------------------

_NODE_STYLE: dict[str, dict] = {
    "yoke":     {"color": "#3b4f6b", "marker": "s", "size": 80, "label": "stator yoke"},
    "tooth":    {"color": "#1f78b4", "marker": "o", "size": 70, "label": "stator tooth (MMF)"},
    "airgap":   {"color": "#33a02c", "marker": "^", "size": 40, "label": "airgap segment"},
    "surface":  {"color": "#ff7f00", "marker": "v", "size": 45, "label": "rotor surface"},
    "fluxtube": {"color": "#b15928", "marker": "o", "size": 55, "label": "rotor flux-tube"},
    "shaft":    {"color": "#6a3d9a", "marker": "D", "size": 90, "label": "shaft"},
}

_EDGE_STYLE: dict[str, dict] = {
    "iron_yoke":     {"color": "#3b4f6b", "lw": 1.5, "ls": "-",  "label": "yoke (iron)"},
    "iron_tooth":    {"color": "#1f78b4", "lw": 1.0, "ls": ":",  "label": "tooth leakage"},
    "yoke_to_tooth": {"color": "#1f78b4", "lw": 1.5, "ls": "-",  "label": "yoke ↔ tooth"},
    "iron_rotor":    {"color": "#b15928", "lw": 1.5, "ls": "-",  "label": "rotor iron"},
    "iron_surface":  {"color": "#ff7f00", "lw": 1.5, "ls": "-",  "label": "surface rim"},
    "airgap":        {"color": "#33a02c", "lw": 1.0, "ls": "--", "label": "airgap reluctance"},
    "barrier":       {"color": "#e31a1c", "lw": 1.8, "ls": "-",  "label": "barrier (air)"},
    "shaft_link":    {"color": "#6a3d9a", "lw": 1.5, "ls": "-",  "label": "shaft link"},
}


# --- Drawing helpers -------------------------------------------------------


def _draw_pole_sector_outline(ax: Axes, net: LumpedNetwork) -> None:
    spec = net.spec
    a0, a1 = net.pole_sector_deg

    # Concentric arcs for shaft, rotor-outer, stator-inner, stator-outer.
    arcs_r_kind = [
        (spec.rotor_r_min,         "shaft"),
        (spec.rotor_r_max,         "rotor surface"),
        (spec.stator_r_inner,      "stator bore"),
        (spec.stator_yoke_r_outer, "stator yoke outer"),
    ]
    deg = np.linspace(a0, a1, 200)
    rad = np.deg2rad(deg)
    for r, _label in arcs_r_kind:
        ax.plot(r * np.cos(rad), r * np.sin(rad), color="#a0a0a0", lw=0.6, zorder=0)

    # Radial sector boundaries.
    for ang in (a0, a1):
        r_end = spec.stator_yoke_r_outer
        ax.plot(
            [0, r_end * np.cos(np.deg2rad(ang))],
            [0, r_end * np.sin(np.deg2rad(ang))],
            color="#a0a0a0",
            lw=0.6,
            zorder=0,
        )

    # d-axis dashed line (45°) for reference.
    r_end = spec.rotor_r_max
    ax.plot(
        [0, r_end * np.cos(np.deg2rad(45))],
        [0, r_end * np.sin(np.deg2rad(45))],
        color="#cccccc",
        lw=0.5,
        ls=":",
        zorder=0,
    )


def _draw_barriers(ax: Axes, net: LumpedNetwork) -> None:
    for polyline in net.barrier_polylines:
        if polyline is None or len(polyline) == 0:
            continue
        ax.plot(polyline[:, 0], polyline[:, 1], color="#888888", lw=0.7, alpha=0.7, zorder=1)


def _draw_air_grid_points(ax: Axes, net: LumpedNetwork) -> None:
    """Mark the per-column air-barrier crossing points (the radial bounds of
    each barrier's air interval at every angular column). These are the
    "grid points of the air" that the iron-channel midradii are derived from.
    """
    # Lazy import to avoid circular reference at module load.
    from .network import _barrier_crossings

    spec = net.spec
    a0, a1 = net.pole_sector_deg
    col_angles = np.linspace(a0, a1, net.granularity.n_col)
    xs: list[float] = []
    ys: list[float] = []
    for theta in col_angles:
        crossings = _barrier_crossings(net.barrier_polylines, float(theta), spec.rotor_r_max)
        rad = np.deg2rad(theta)
        for c in crossings:
            if c is None:
                continue
            r_in, r_out = c
            xs.extend([r_in * np.cos(rad), r_out * np.cos(rad)])
            ys.extend([r_in * np.sin(rad), r_out * np.sin(rad)])
    if xs:
        ax.scatter(
            xs, ys, c="#e31a1c", marker="x", s=18, linewidths=0.9, zorder=4, label="air grid (barrier crossings)"
        )


def _draw_edges(ax: Axes, net: LumpedNetwork) -> None:
    g = net.graph
    pos = {n: (d["x"], d["y"]) for n, d in g.nodes(data=True)}
    for u, v, d in g.edges(data=True):
        kind = d["kind"]
        crossings = d.get("barrier_crossings", [])
        intent_is_air = kind == "barrier"
        derived_is_air = bool(crossings)

        # Choose the geometry. Within-channel midlines are stored as a
        # polyline attribute; everything else is the straight segment
        # between endpoint positions.
        poly = d.get("polyline")
        if poly is not None and len(poly) >= 2:
            xs = [float(p[0]) for p in poly]
            ys = [float(p[1]) for p in poly]
        else:
            xs = [pos[u][0], pos[v][0]]
            ys = [pos[u][1], pos[v][1]]

        if derived_is_air == intent_is_air:
            style = _EDGE_STYLE.get(kind, {"color": "k", "lw": 1.0, "ls": "-"})
            color = style["color"]
            lw = style["lw"]
            ls = style["ls"]
            alpha = 0.85
        else:
            color = "k"
            lw = 2.2
            ls = (0, (3, 1))
            alpha = 1.0

        ax.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=2, alpha=alpha)


def _draw_nodes(ax: Axes, net: LumpedNetwork) -> None:
    g = net.graph
    for kind, style in _NODE_STYLE.items():
        xs, ys = [], []
        for _, d in g.nodes(data=True):
            if d["kind"] == kind:
                xs.append(d["x"])
                ys.append(d["y"])
        if xs:
            ax.scatter(
                xs,
                ys,
                c=style["color"],
                marker=style["marker"],
                s=style["size"],
                edgecolors="white",
                linewidths=0.8,
                zorder=3,
            )


def _legend_handles() -> list:
    handles = []
    for _kind, style in _NODE_STYLE.items():
        handles.append(
            plt.Line2D(
                [0], [0],
                marker=style["marker"], color="w",
                markerfacecolor=style["color"], markersize=8,
                markeredgecolor="white", label=style["label"],
            )
        )
    for _kind, style in _EDGE_STYLE.items():
        handles.append(
            plt.Line2D([0], [0], color=style["color"], lw=style["lw"], ls=style["ls"], label=style["label"])
        )
    handles.append(
        plt.Line2D([0], [0], color="k", lw=2.2, ls=(0, (3, 1)), label="kind ≠ geometry")
    )
    handles.append(
        plt.Line2D(
            [0], [0], marker="x", color="w",
            markeredgecolor="#e31a1c", markersize=7, lw=0,
            label="air grid (barrier crossings)",
        )
    )
    return handles


def _legend(ax: Axes) -> None:
    ax.legend(handles=_legend_handles(), loc="upper left", fontsize=7, frameon=True, framealpha=0.9, ncol=1)


# --- Public API ------------------------------------------------------------


def plot_network(
    net: LumpedNetwork,
    ax: Axes | None = None,
    show_legend: bool = True,
    title: str | None = None,
    show_air_grid: bool = False,
) -> tuple[Figure, Axes]:
    """Render one pole sector of `net` to a matplotlib axis.

    `show_air_grid` overlays the per-column barrier-crossing points (the air
    grid that drives the iron-channel midradius computation).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
    else:
        fig = ax.figure

    _draw_pole_sector_outline(ax, net)
    _draw_barriers(ax, net)
    _draw_edges(ax, net)
    _draw_nodes(ax, net)
    if show_air_grid:
        _draw_air_grid_points(ax, net)

    if show_legend:
        _legend(ax)

    if title is None:
        gr = net.granularity
        title = (
            f"Lumped reluctance network — one pole\n"
            f"yoke={gr.n_yoke}, teeth={gr.n_teeth}, airgap={gr.n_airgap}, "
            f"flux-tube/channel={gr.n_fluxtube}"
        )
    ax.set_title(title, fontsize=10)

    # Square axes, padded slightly beyond the stator yoke radius.
    rmax = net.spec.stator_yoke_r_outer * 1.05
    ax.set_xlim(-0.05 * rmax, rmax)
    ax.set_ylim(-0.05 * rmax, rmax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.grid(True, alpha=0.15)

    return fig, ax


def plot_granularity_grid(
    nets: list[LumpedNetwork],
    titles: list[str] | None = None,
    figsize: tuple[float, float] = (17, 5.5),
) -> Figure:
    """Render a row of networks side by side for granularity ablation figures.

    A dedicated right-hand panel holds the legend so it never overlaps with
    any plot.
    """
    n = len(nets)
    # n network panels of equal width + 1 narrower legend panel.
    width_ratios = [1.0] * n + [0.45]
    fig, axes = plt.subplots(
        1, n + 1, figsize=figsize, gridspec_kw={"width_ratios": width_ratios}
    )
    if n + 1 == 1:
        axes = [axes]
    for i, (net, ax) in enumerate(zip(nets, axes[:n])):
        ttl = titles[i] if titles is not None and i < len(titles) else None
        plot_network(net, ax=ax, show_legend=False, title=ttl)

    legend_ax = axes[-1]
    legend_ax.axis("off")
    legend_ax.legend(
        handles=_legend_handles(),
        loc="center left",
        fontsize=8,
        frameon=True,
        framealpha=0.95,
        ncol=1,
        title="Lumped network legend",
        title_fontsize=9,
    )
    fig.tight_layout()
    return fig
