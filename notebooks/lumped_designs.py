"""Lumped network on the three Hackl-family parameterisations at FINE granularity.

3×3 grid: rows = {OneLambda 7-D, SixLambdas 12-D, 3BrokenLines 13-D},
columns = three random parameter samples per method. Each panel also shows
the per-column barrier-crossing points (the "air grid" that the iron channel
midradii are derived from).

Run from the MachineDesign repo root:

    python notebooks/lumped_designs.py

Output: `notebooks/lumped_designs.{pdf,png}`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from machine_design.generators import (
    HacklGenerator_3BrokenLines,
    HacklGenerator_OneLambda,
    HacklGenerator_SixLambdas,
)
from machine_design.lumped import (
    GRANULARITY_FINE,
    REFERENCE_MACHINE,
    build_network,
)
from machine_design.lumped.visualize import _legend_handles, plot_network


GENERATOR_ROWS = [
    ("OneLambda (7-D)", HacklGenerator_OneLambda),
    ("SixLambdas (12-D)", HacklGenerator_SixLambdas),
    ("3BrokenLines (13-D)", HacklGenerator_3BrokenLines),
]
COLUMN_SEEDS = (0, 7, 42)


def sample_barriers(generator_cls, seed: int) -> list[np.ndarray]:
    """Deterministic random barrier sample for a given generator class."""
    rng_state = np.random.get_state()
    np.random.seed(seed)
    try:
        gen = generator_cls(REFERENCE_MACHINE, r_stator_end=0.7, offset=0.35)
        gen.set_parameters(gen.random_parameters())
        barriers = gen.generate_barriers()
    finally:
        np.random.set_state(rng_state)
    return barriers


def main() -> None:
    n_rows = len(GENERATOR_ROWS)
    n_cols = len(COLUMN_SEEDS)
    width_ratios = [1.0] * n_cols + [0.55]
    fig, axes = plt.subplots(
        n_rows, n_cols + 1,
        figsize=(5.0 * n_cols + 3.0, 5.0 * n_rows),
        gridspec_kw={"width_ratios": width_ratios},
    )

    for r, (row_name, gen_cls) in enumerate(GENERATOR_ROWS):
        for c, seed in enumerate(COLUMN_SEEDS):
            ax = axes[r, c]
            barriers = sample_barriers(gen_cls, seed)
            net = build_network(REFERENCE_MACHINE, GRANULARITY_FINE, barrier_polylines=barriers)
            title = f"{row_name}\nseed={seed}"
            plot_network(net, ax=ax, show_legend=False, title=title, show_air_grid=True)
            m = net.topology_mismatches()
            n_edges = net.graph.number_of_edges()
            note = (f"edges={n_edges}  iron→air={m['iron_should_be_air']}"
                    f"  barr0={m['barrier_no_cross']}  ag×={m['airgap_crosses']}")
            ax.text(
                0.02, 0.98, note, transform=ax.transAxes,
                fontsize=7, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="#cccccc"),
            )

        # legend in the right-most column of the first row only
        if r == 0:
            legend_ax = axes[r, n_cols]
            legend_ax.axis("off")
            legend_ax.legend(
                handles=_legend_handles(),
                loc="upper left",
                fontsize=8,
                frameon=True,
                framealpha=0.95,
                ncol=1,
                title="Lumped network legend",
                title_fontsize=9,
            )
        else:
            axes[r, n_cols].axis("off")

    fig.suptitle(
        "Lumped reluctance network — FINE granularity\n"
        "rows: barrier parameterisation, columns: random parameter sample",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out_dir = Path(__file__).resolve().parent
    pdf_path = out_dir / "lumped_designs.pdf"
    png_path = out_dir / "lumped_designs.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
