"""Render the lumped-reluctance network for one pole at multiple granularities.

Produces a single PDF figure (3 panels: coarse / medium / fine) for the 6λ
Hackl barrier parameterization on the ICEM2026 reference machine. No solver,
no ANSYS, no FEA data — just the graph topology overlaid on the rotor
cross-section.

Run from the MachineDesign repo root:

    python notebooks/lumped_sketch.py

Output: `notebooks/lumped_sketch.pdf` and `.png` next to this script.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from machine_design.generators import HacklGenerator_SixLambdas
from machine_design.lumped import (
    GRANULARITY_COARSE,
    GRANULARITY_FINE,
    GRANULARITY_MEDIUM,
    REFERENCE_MACHINE,
    build_network,
)
from machine_design.lumped.visualize import plot_granularity_grid

# `BarrierGenerator` only reads `rotor_r_max`/`rotor_r_min` from its first arg,
# so the `MachineSpec` works in place of a full `Design`.


def make_barriers(seed: int = 0) -> list[np.ndarray]:
    """Build a deterministic SixLambdas barrier set for the illustration."""
    rng_state = np.random.get_state()
    np.random.seed(seed)
    try:
        # r_stator_end is the radial margin between rotor_r_max and the outermost
        # barrier control radius. The ICEM run scripts use 0.7 mm; replicate that.
        gen = HacklGenerator_SixLambdas(REFERENCE_MACHINE, r_stator_end=0.7, offset=0.35)
        params = gen.random_parameters()
        gen.set_parameters(params)
        barriers = gen.generate_barriers()
    finally:
        np.random.set_state(rng_state)
    return barriers


def main() -> None:
    barriers = make_barriers(seed=0)

    nets = [
        build_network(REFERENCE_MACHINE, GRANULARITY_COARSE, barrier_polylines=barriers),
        build_network(REFERENCE_MACHINE, GRANULARITY_MEDIUM, barrier_polylines=barriers),
        build_network(REFERENCE_MACHINE, GRANULARITY_FINE, barrier_polylines=barriers),
    ]
    titles = ["Coarse", "Medium", "Fine"]

    fig = plot_granularity_grid(nets, titles=titles)
    out_dir = Path(__file__).resolve().parent
    pdf_path = out_dir / "lumped_sketch.pdf"
    png_path = out_dir / "lumped_sketch.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")

    # Topology vs. geometry sanity check. M1's reluctance values come from the
    # `barrier_crossings` attribute on each edge, not from the constructor's
    # `kind`. Mismatches mean the straight-chord edge enters or skips a
    # barrier — for M0 we just count and report them.
    print()
    print("Topology mismatches per granularity (kind contradicts geometry):")
    for title, net in zip(titles, nets):
        m = net.topology_mismatches()
        n_edges = net.graph.number_of_edges()
        print(f"  {title:<6} edges={n_edges:>4}  "
              f"iron_should_be_air={m['iron_should_be_air']:>3}  "
              f"barrier_no_cross={m['barrier_no_cross']:>3}  "
              f"airgap_crosses={m['airgap_crosses']:>3}")


if __name__ == "__main__":
    main()
