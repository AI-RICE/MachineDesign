"""Plot the Step-5 pilot2 result in objective space against the Step-1 bar:
bar cloud + bar Pareto front (Hackl, converged), pilot2 warm-start vs BO designs
(Bezier decode), and the high-fidelity bias-corrected points. Tells the story:
BO explores within the ~1.8% T_mean band, the n_per=160 decode bias shifts it
left, and bias-corrected it only ties the bar's low-ripple corner.

  .venv/bin/python notebooks/plot_pilot.py
"""

import csv
import os
import re

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def load_bar():
    Tm, Tr, front = [], [], []
    for r in csv.DictReader(open(os.path.join(ROOT, "docs/tables/bar_converged.csv"))):
        if r["status"] != "ok":
            continue
        Tm.append(float(r["T_mean_conv"])); Tr.append(float(r["T_ripple_conv"]))
        front.append(r["on_conv_front"] == "1")
    return np.array(Tm), np.array(Tr), np.array(front)


def load_pilot():
    ph, Tm, Tr = [], [], []
    for r in csv.DictReader(open("/tmp/pilotdata/pilot2_evals.csv")):
        ph.append(r["phase"]); Tm.append(float(r["T_mean"])); Tr.append(float(r["T_ripple"]))
    return np.array(ph), np.array(Tm), np.array(Tr)


def load_hifid():
    pts = []
    for ln in open("/tmp/pilotdata/reeval_pilot2.log"):
        m = re.search(r"(warmstart|bo)\s*\|.*\|\s*([\d.]+)/\s*([\d.]+)\s*\|", ln)
        if m:
            pts.append((m.group(1), float(m.group(2)), float(m.group(3))))
    return pts


def pareto(Tm, Tr):
    f = np.column_stack([Tm, -Tr]); nd = np.ones(len(f), bool)
    for i in range(len(f)):
        if nd[i]:
            nd[np.all(f <= f[i], axis=1) & np.any(f < f[i], axis=1)] = False
    return nd


bTm, bTr, bfront = load_bar()
ph, pTm, pTr = load_pilot()
hifid = load_hifid()

fig, ax = plt.subplots(figsize=(8, 6))
# bar cloud + front
ax.scatter(bTm, bTr, s=12, c="0.8", label="bar designs (Hackl, converged)", zorder=1)
o = np.argsort(bTm[bfront])
ax.plot(bTm[bfront][o], bTr[bfront][o], "-o", c="k", ms=7, lw=1.5,
        label=f"BAR front ({bfront.sum()} pts)", zorder=4)
# pilot2 decode designs
ws, bo = ph == "warmstart", ph == "bo"
ax.scatter(pTm[ws], pTr[ws], s=28, marker="s", facecolors="none", edgecolors="tab:blue",
           label="pilot2 warm-start (decode)", zorder=2)
ax.scatter(pTm[bo], pTr[bo], s=28, marker="^", c="tab:orange",
           label="pilot2 BO (decode)", zorder=3)
# high-fidelity bias-corrected points
for p, tm, tr in hifid:
    ax.scatter(tm, tr, s=170, marker="*", c=("tab:green" if p == "bo" else "darkgreen"),
               edgecolors="k", zorder=5)
ax.scatter([], [], s=170, marker="*", c="darkgreen", edgecolors="k",
           label="hi-fid bias-corrected (n_per=320)")

ax.set_xlabel("T_mean (N·m)  → better")
ax.set_ylabel("T_ripple (%)  ← better")
ax.set_title("Step-5 pilot2 vs bar — single operating point (near-saturated)\n"
             "BO adds no point that survives hi-fid re-eval; 'wins' are noise-level")
ax.set_ylim(0, 20)
ax.invert_yaxis()
ax.legend(loc="upper left", fontsize=8, framealpha=0.95)
ax.grid(alpha=0.3)
out = os.path.join(ROOT, "docs/figures/step5_pilot2_vs_bar.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.tight_layout(); fig.savefig(out, dpi=130)
print(f"saved {out}")
print(f"bar front {bfront.sum()} pts; pilot2 {ws.sum()} ws + {bo.sum()} bo; hi-fid valid {len(hifid)}")
