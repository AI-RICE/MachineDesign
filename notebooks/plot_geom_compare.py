"""Side-by-side rotor geometry of three designs near each other on the front:
SixLambdas (idx54), ThreeBrokenLines (idx138), and our Bezier BO design (pilot2
eval 35). Shows whether the free-form Bezier finds a DIFFERENT geometry for
similar (T_mean, ripple).

  .venv/bin/python notebooks/plot_geom_compare.py
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from machine_design.bezier_bo import warmstart_box  # noqa: E402
from machine_design.bezier_generator import BezierSupersetGenerator  # noqa: E402
from machine_design.generators import (  # noqa: E402
    HacklGenerator_3BrokenLines,
    HacklGenerator_SixLambdas,
)
from machine_design.lumped.geometry import REFERENCE_MACHINE  # noqa: E402
from notebooks.run_bar_converged import build_bar_list  # noqa: E402

Xs, shorts, _, _, _ = build_bar_list("../MachineDesign/results", 200)


def hackl_bars(cls, X):
    g = cls(REFERENCE_MACHINE, r_stator_end=0.7, offset=0.35)
    g.set_parameters(g.X_to_params(np.asarray(X, float)))
    return g.generate_barriers(), g.r_min, g.r_max


def our_bars():
    d = np.load(os.path.join(HERE, "Bezier_warmstart_converged.npz"), allow_pickle=True)
    g0 = BezierSupersetGenerator(REFERENCE_MACHINE, M=int(d["M"]))
    lo, span = warmstart_box(d["X_bz"], g0)
    e = np.load("/tmp/pilotdata/eval_0035.npz", allow_pickle=True)
    g = BezierSupersetGenerator(REFERENCE_MACHINE, M=int(d["M"]), n_per=320)
    g.set_parameters(lo + e["X_norm"].astype(float) * span)
    return g.generate_barriers(), g.r_min, g.r_max


def draw(ax, bars, rmin, rmax, title):
    th = np.linspace(0, np.pi / 2, 100)
    # iron sector (light gray), shaft hole (white)
    sec = np.concatenate([np.column_stack([rmax * np.cos(th), rmax * np.sin(th)]),
                          np.column_stack([rmin * np.cos(th[::-1]), rmin * np.sin(th[::-1])])])
    ax.add_patch(Polygon(sec, closed=True, facecolor="0.85", edgecolor="0.4", lw=1.2, zorder=1))
    for b in bars:  # air barriers
        ax.add_patch(Polygon(np.asarray(b)[:, :2], closed=True, facecolor="white",
                             edgecolor="tab:blue", lw=1.0, zorder=2))
    ax.plot(rmax * np.cos(th), rmax * np.sin(th), "k-", lw=0.8)
    ax.set_aspect("equal"); ax.set_xlim(-1, rmax + 1); ax.set_ylim(-1, rmax + 1)
    ax.set_title(title, fontsize=10); ax.axis("off")


b6, r0, r1 = hackl_bars(HacklGenerator_SixLambdas, Xs[54])
b3, _, _ = hackl_bars(HacklGenerator_3BrokenLines, Xs[138])
bo, ro0, ro1 = our_bars()

fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
draw(axes[0], b6, r0, r1, "SixLambdas (idx54)\nT=4.435 N·m, ripple 3.85%")
draw(axes[1], b3, r0, r1, "ThreeBrokenLines (idx138)\nT=4.450 N·m, ripple 4.21%")
draw(axes[2], bo, ro0, ro1, "OURS — Bezier BO (eval35)\nT=4.441 N·m, ripple 5.01% (hi-fid)")
fig.suptitle("Rotor flux-barrier geometry — three designs adjacent on the Pareto front", fontsize=12)
out = os.path.join(os.path.dirname(HERE), "docs", "figures", "front_geometry_compare.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(out, dpi=130)
print(f"saved {out}")
