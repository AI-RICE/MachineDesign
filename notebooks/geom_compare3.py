"""Compare the rotor flux-barrier geometry of the three 60-slot min-loss designs
(T>=20 Nm, ripple<=5%, 50 Hz, common stator):
  3-phase (dq1)            -> loss 16.61
  5-phase dq1-only         -> loss 5.82
  5-phase joint (dq1+dq3)  -> loss 5.85
Barriers are read from the CORRECTLY-decoded dumps (wide=False, real rotor radii):
  - the two 5-phase rotors from xcheck.json (built by crosscheck.py)
  - the 3-phase rotor decoded here with the matching mock generator (wide=False)
1x4 grid: three machines + outline overlay. Writes ../../paper/geom3.{png,pdf}.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

import h0h1_par as P
import h0h1_study as H

xc = json.load(open("xcheck.json"))
rmin, rmax = xc["rotor_r_min"], xc["rotor_r_max"]


class _D:
    rotor_r_min, rotor_r_max = rmin, rmax


gen = P.make_generator(_D(), False)            # wide=False, matches the runs
lb, ub = H.geom_bounds_arrays(gen)

D3 = json.load(open("results/minloss60_3f/dq1_best.json"))
D5 = json.load(open("results/minloss60_5f_dq1/dq1_best.json"))
DJ = json.load(open("results/minloss60_5f_joint/joint_best.json"))
bars_3f = [np.asarray(b, float)[:, :2] for b in H.build_barriers(gen, np.asarray(D3["geom_norm"], float), lb, ub)]

cases = [
    dict(key="3f", color="tab:red", title="3-phase (dq1)", bars=bars_3f, d=D3),
    dict(key="5f", color="tab:blue", title="5-phase dq1-only",
         bars=[np.asarray(b, float) for b in xc["geoms"]["G_dq1"]], d=D5),
    dict(key="5fj", color="tab:green", title="5-phase joint (dq1+dq3)",
         bars=[np.asarray(b, float) for b in xc["geoms"]["G_joint"]], d=DJ),
]
for c in cases:
    dq = c["d"]["dq"]
    c["note"] = (f"loss={c['d']['loss']:.2f}  |I|={c['d']['Irms_equiv']:.2f} A\n"
                 f"T={c['d']['T']:.1f} Nm  rip={c['d']['ripple']:.1f}%\n"
                 f"dq=[{dq[0]:.2f}, {dq[1]:.2f}, {dq[2]:.2f}, {dq[3]:.2f}]")

allpts = np.vstack([b for c in cases for b in c["bars"]])
th = np.arctan2(allpts[:, 1], allpts[:, 0]); th0, th1 = th.min(), th.max()
xs, ys = allpts[:, 0], allpts[:, 1]; pad = 2


def wedge(ax):
    a = np.linspace(th0, th1, 100)
    for r in (rmin, rmax):
        ax.plot(r * np.cos(a), r * np.sin(a), color="0.5", lw=0.9)
    for t in (th0, th1):
        ax.plot([rmin * np.cos(t), rmax * np.cos(t)], [rmin * np.sin(t), rmax * np.sin(t)], color="0.5", lw=0.9)
    ax.set_xlim(xs.min() - pad, xs.max() + pad); ax.set_ylim(ys.min() - pad, ys.max() + pad)
    ax.axis("off")


fig, axes = plt.subplots(1, 4, figsize=(17, 5.2), subplot_kw={"aspect": "equal"})
for ax, c in zip(axes[:3], cases):
    wedge(ax)
    for p in c["bars"]:
        ax.add_patch(Polygon(p, closed=True, facecolor=c["color"], edgecolor=c["color"], alpha=0.20, lw=2.0, zorder=3))
    ax.set_title(c["title"], fontsize=12, fontweight="bold", color=c["color"])
    ax.text(0.5, -0.04, c["note"], transform=ax.transAxes, ha="center", va="top", fontsize=9.5)

ax = axes[3]; wedge(ax)
for c in cases:
    for p in c["bars"]:
        ax.plot(np.append(p[:, 0], p[0, 0]), np.append(p[:, 1], p[0, 1]),
                color=c["color"], lw=1.6, zorder=3, label=c["title"])
h, l = ax.get_legend_handles_labels(); seen = dict(zip(l, h))
ax.legend(seen.values(), seen.keys(), fontsize=8.5, loc="lower center", framealpha=0.9)
ax.set_title("overlay (outlines)", fontsize=12, fontweight="bold")

fig.suptitle("Rotor flux-barrier geometry — fair min-loss inner point on the common 60-slot stator\n"
             "(T$\\geq$20 Nm, ripple$\\leq$5%, 50 Hz; same 12-dim SixLambda rotor space; wide=False)",
             fontsize=12.5)
fig.tight_layout(rect=[0, 0, 1, 0.92])
HERE = os.path.dirname(os.path.abspath(__file__))
for ext in ("png", "pdf"):
    out = os.path.join(HERE, "..", "..", "paper", f"geom3.{ext}")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)
