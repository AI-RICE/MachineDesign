"""RadialSpline geometry gate (no FEA) — §11.1 / §12 step 2.

Acceptance checks for the new unified parameterisation:
  1. random designs are feasible and look like plausible SynRM rotors,
  2. warm-start round-trips the existing Hackl designs (encode -> decode).

Outputs PNGs next to this script:
  RadialSpline_random.png      — grid of random designs (rib shown)
  RadialSpline_warmstart.png   — Hackl design vs RadialSpline refit overlay

Run:
  .venv/bin/python notebooks/radialspline_geometry_gate.py
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from machine_design.generators import (  # noqa: E402
    HacklGenerator_3BrokenLines,
    HacklGenerator_OneLambda,
    HacklGenerator_SixLambdas,
    RadialSplineGenerator,
)
from machine_design.lumped.geometry import REFERENCE_MACHINE  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def draw_rotor(ax, gen, barriers, title=""):
    """Pole sector: shaft, rotor surface arc, filled air barriers."""
    a = np.linspace(0, np.pi / 2, 100)
    ax.plot(gen.r_min * np.cos(a), gen.r_min * np.sin(a), "k-", lw=0.8)
    ax.plot(gen.r_max * np.cos(a), gen.r_max * np.sin(a), "k-", lw=1.2)
    for r in (gen.r_min, gen.r_max):
        ax.plot([0, 0], [r * 0, r * 0], "k-")
    ax.plot([gen.r_min, gen.r_max], [0, 0], "k-", lw=0.8)
    ax.plot([0, 0], [gen.r_min, gen.r_max], "k-", lw=0.8)
    for b in barriers:
        ax.fill(b[:, 0], b[:, 1], facecolor="#4488cc", edgecolor="#113355", lw=0.5, alpha=0.85)
    ax.set_aspect("equal")
    ax.set_xlim(-2, gen.r_max + 2)
    ax.set_ylim(-2, gen.r_max + 2)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def fig_random(gen, n=9, seed=0):
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    n_feas = 0
    for ax in axes.ravel():
        gen.set_parameters(gen.random_X(rng))
        bars = gen.generate_barriers()
        feas = gen.feasible_barriers(bars)
        n_feas += feas
        ribbed = gen.split_barriers(bars)  # show central rib
        draw_rotor(ax, gen, ribbed, f"feasible={feas}")
    fig.suptitle(f"RadialSpline random designs (rib shown). feasible {n_feas}/{n}", fontsize=11)
    fig.tight_layout()
    out = os.path.join(HERE, "RadialSpline_random.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def fig_warmstart(gen, seed=3):
    families = [
        ("OneLambda", HacklGenerator_OneLambda),
        ("SixLambdas", HacklGenerator_SixLambdas),
        ("3BrokenLines", HacklGenerator_3BrokenLines),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    rng = np.random.default_rng(seed)
    for ax, (name, cls) in zip(axes, families):
        hk = cls(REFERENCE_MACHINE, r_stator_end=0.7, offset=0.35)
        np.random.seed(int(rng.integers(1e9)))
        hk.set_parameters(hk.random_parameters())
        hbars = hk.generate_barriers()
        X = gen.fit_barriers(hbars)
        gen.set_parameters(X)
        rbars = gen.generate_barriers()
        a = np.linspace(0, np.pi / 2, 100)
        ax.plot(gen.r_min * np.cos(a), gen.r_min * np.sin(a), "k-", lw=0.8)
        ax.plot(gen.r_max * np.cos(a), gen.r_max * np.sin(a), "k-", lw=1.2)
        for b in rbars:
            ax.fill(b[:, 0], b[:, 1], facecolor="#4488cc", edgecolor="none", alpha=0.5)
        for b in hbars:
            ax.plot(b[:, 0], b[:, 1], "r-", lw=1.0)
        ax.set_aspect("equal")
        ax.set_xlim(-2, gen.r_max + 2)
        ax.set_ylim(-2, gen.r_max + 2)
        ax.set_title(f"{name}: Hackl (red) vs RadialSpline refit (blue)", fontsize=9)
        ax.axis("off")
    fig.suptitle("Warm-start round-trip: existing parameterisations -> RadialSpline", fontsize=11)
    fig.tight_layout()
    out = os.path.join(HERE, "RadialSpline_warmstart.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main():
    gen = RadialSplineGenerator(REFERENCE_MACHINE)
    lo, _ = gen.bounds
    print(f"RadialSpline: D={lo.shape[0]}, N={gen.N}, K={gen.K}")

    # feasibility rate
    rng = np.random.default_rng(0)
    M = 1000
    n_ok = 0
    for _ in range(M):
        gen.set_parameters(gen.random_X(rng))
        n_ok += gen.feasible_barriers(gen.generate_barriers())
    print(f"random feasibility: {n_ok}/{M} ({100 * n_ok / M:.1f}%)")

    print("wrote", fig_random(gen))
    print("wrote", fig_warmstart(gen))


if __name__ == "__main__":
    main()
