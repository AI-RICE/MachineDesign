"""3x2 regime grid of rotor flux-barrier geometries.
Rows = regimes (no-limit / current-limited / voltage-limited);
columns = G* (dq1-only) and joint (dq1+dq3). Each joint panel overlays its own
regime's G* (grey) so the coupling move is visible per regime. Reads
geom_shapes6.json (built by extract_geom.py)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "..", "geom_shapes6.json")))
rmin, rmax = d["rotor_r_min"], d["rotor_r_max"]
C = d["cases"]

ROWS = [("nolim", "tab:blue"), ("curr", "tab:green"), ("volt", "tab:red")]

allpts = np.vstack([np.array(p) for c in C.values() for p in c["barriers"]])
th = np.arctan2(allpts[:, 1], allpts[:, 0]); th0, th1 = th.min(), th.max()
xs, ys = allpts[:, 0], allpts[:, 1]; pad = 2

fig, axes = plt.subplots(3, 2, figsize=(9.5, 13), subplot_kw={"aspect": "equal"})


def wedge(ax):
    a = np.linspace(th0, th1, 100)
    for r in (rmin, rmax):
        ax.plot(r * np.cos(a), r * np.sin(a), color="0.5", lw=0.9)
    for t in (th0, th1):
        ax.plot([rmin * np.cos(t), rmax * np.cos(t)], [rmin * np.sin(t), rmax * np.sin(t)], color="0.5", lw=0.9)


def draw(ax, bars, color, base=None):
    wedge(ax)
    if base is not None:
        for p in base:
            p = np.array(p); ax.plot(p[:, 0], p[:, 1], color="0.55", lw=1.0, zorder=2)
    for p in bars:
        p = np.array(p)
        ax.add_patch(Polygon(p, closed=True, facecolor=color, edgecolor=color, alpha=0.18, lw=2.0, zorder=3))
    ax.set_xlim(xs.min() - pad, xs.max() + pad); ax.set_ylim(ys.min() - pad, ys.max() + pad)
    ax.axis("off")


for r, (reg, color) in enumerate(ROWS):
    g, j = C[f"{reg}_gstar"], C[f"{reg}_joint"]
    draw(axes[r, 0], g["barriers"], color)
    draw(axes[r, 1], j["barriers"], color, base=g["barriers"])
    axes[r, 0].set_title(g["role"], fontsize=10.5)
    axes[r, 1].set_title(j["role"] + "   (grey = $G^\\star$)", fontsize=10.5)
    axes[r, 0].text(0.5, -0.02, g["note"], transform=axes[r, 0].transAxes, ha="center", va="top", fontsize=8.5)
    axes[r, 1].text(0.5, -0.02, j["note"], transform=axes[r, 1].transAxes, ha="center", va="top", fontsize=8.5)
    # regime label on the row's left
    axes[r, 0].text(-0.08, 0.5, g["regime"], transform=axes[r, 0].transAxes, rotation=90,
                    ha="center", va="center", fontsize=11, fontweight="bold")

fig.suptitle("Rotor flux-barrier geometry across regimes: $G^\\star$ ($dq1$) vs joint ($dq1{+}dq3$)\n"
             "each regime has its own $G^\\star$; grey baseline = that regime's $G^\\star$", fontsize=12)
fig.tight_layout(rect=[0.02, 0, 1, 0.95])
for ext in ("png", "pdf"):
    out = os.path.join(HERE, "..", "..", "paper", f"geom_grid.{ext}")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)
