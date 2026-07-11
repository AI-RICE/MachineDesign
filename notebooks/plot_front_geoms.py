"""2x2 rotor-geometry grid for the Pareto ENDPOINTS at both bus voltages:
rows = {400 V, 500 V}, cols = {min-loss endpoint, min-ripple endpoint}. Finds each
endpoint from its pool (exact-peak voltage feasibility), builds the flux-barrier
polylines via the real generator, and renders one pole sector per cell. Writes
results/gen3_front_geoms.pdf. Run on bayes (needs h0h1_* / a design for the generator).
"""
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

import h0h1_par as P
import h0h1_study as H

RS, LEW, IMAX = 19.0, 2.4e-3, 1.3
LB = np.array([0.0, 0.0, -0.39, -0.39]); UB = np.array([1.3, 1.3, 0.39, 0.39])
DEM = [4.0, 8.0, 2.5]; SPD = [25.0, 16.0, 63.0]; OMEGA = [2 * math.pi * f for f in SPD]
TH = np.linspace(0.0, 2 * math.pi, 361, endpoint=False)
dq_of = lambda u: LB + u * (UB - LB)


def _peak(m1, a1, m3, a3):
    return float(np.max(np.abs(m1 * np.cos(TH + a1) + m3 * np.cos(3 * TH + a3))))


def ipk(dq):
    return _peak(math.hypot(dq[0], dq[1]), math.atan2(dq[1], dq[0]),
                 math.hypot(dq[2], dq[3]), math.atan2(dq[3], dq[2]))


def vpk(fl, dq, w):
    Vd1 = RS * dq[0] - w * (fl[1] + LEW * dq[1]); Vq1 = RS * dq[1] + w * (fl[0] + LEW * dq[0])
    Vd3 = RS * dq[2] - 3 * w * (fl[3] + LEW * dq[3]); Vq3 = RS * dq[3] + 3 * w * (fl[2] + LEW * dq[2])
    return _peak(math.hypot(Vd1, Vq1), math.atan2(Vq1, Vd1),
                 math.hypot(Vd3, Vq3), math.atan2(Vq3, Vd3))


def front_endpoints(out, vmax):
    z = np.load(f"{out}/gen3.npz", allow_pickle=True)
    Xg, Xi, T, R, FL = [np.array(z[k]) for k in ("Xg", "Xi", "T", "R", "FL")]
    key = {}
    for i in range(len(T)):
        key.setdefault(tuple(np.round(Xg[i], 6)), []).append(i)
    rows = []
    for g, ids in key.items():
        dq = np.array([dq_of(Xi[i]) for i in ids]); tt = np.array([T[i] for i in ids])
        rr = np.array([R[i] for i in ids]); fl = FL[ids]
        ll = np.sum(dq ** 2, 1); ip = np.array([ipk(dq[j]) for j in range(len(ids))]); ptp = rr * tt / 100
        cl, mp, ok, wr = 0.0, 0.0, True, 0.0
        for Tk, w in zip(DEM, OMEGA):
            vv = np.array([vpk(fl[j], dq[j], w) for j in range(len(ids))])
            fe = (ip <= IMAX + 1e-6) & (tt >= Tk) & (vv <= vmax)
            if not fe.any():
                ok = False; break
            j = int(np.where(fe)[0][np.argmin(ll[fe])]); cl += ll[j] / 3
            if ptp[j] > mp:
                mp, wr = ptp[j], rr[j]
        if ok:
            rows.append((cl, mp, wr, np.array(g)))
    nd = [r for r in rows if not any((o[0] <= r[0] and o[1] < r[1]) or (o[0] < r[0] and o[1] <= r[1]) for o in rows)]
    return min(nd, key=lambda r: r[0]), min(nd, key=lambda r: r[1])   # (min-loss, min-ripple)


gd = P.open_isolated_design("plotg", 95, "2024.2", slots=60, phases=5)
gen = P.make_generator(gd, False); lb, ub = H.geom_bounds_arrays(gen)
rmin, rmax = float(gd.rotor_r_min), float(gd.rotor_r_max)

cells = []  # (row, col, tag, role, cl, mp, wr, bars)
for row, (dirn, vmax, tag) in enumerate([("results/gen3_500w_v", 400.0, "400 V"),
                                         ("results/gen3_500w_v500", 500.0, "500 V")]):
    ml, mr = front_endpoints(dirn, vmax)
    for col, (role, r) in enumerate([("min-loss", ml), ("min-ripple", mr)]):
        cl, mp, wr, g = r
        bars = [np.asarray(b, float)[:, :2] for b in H.build_barriers(gen, g, lb, ub)]
        cells.append((row, col, tag, role, cl, mp, wr, bars))
gd.close_project()

allpts = np.vstack([b for c in cells for b in c[7]])
th = np.arctan2(allpts[:, 1], allpts[:, 0]); th0, th1 = th.min(), th.max()
fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.2), subplot_kw=dict(aspect="equal"))
for row, col, tag, role, cl, mp, wr, bars in cells:
    ax = axes[row, col]
    a = np.linspace(th0, th1, 120)
    for r in (rmin, rmax):
        ax.plot(r * np.cos(a), r * np.sin(a), color="0.55", lw=0.8)
    ax.plot([rmin * np.cos(th0), rmax * np.cos(th0)], [rmin * np.sin(th0), rmax * np.sin(th0)], color="0.55", lw=0.8)
    ax.plot([rmin * np.cos(th1), rmax * np.cos(th1)], [rmin * np.sin(th1), rmax * np.sin(th1)], color="0.55", lw=0.8)
    col_c = "tab:blue" if tag == "400 V" else "tab:red"
    for b in bars:
        ax.add_patch(Polygon(b, closed=True, facecolor=col_c, edgecolor="k", lw=0.5, alpha=0.55))
    ax.set_title(f"{tag}, {role}\ncycle-loss {cl:.2f}, ripple {wr:.0f}% ({mp:.2f} Nm)", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(allpts[:, 0].min() - 2, allpts[:, 0].max() + 2)
    ax.set_ylim(allpts[:, 1].min() - 2, allpts[:, 1].max() + 2)
fig.tight_layout()
fig.savefig("results/gen3_front_geoms.pdf", bbox_inches="tight")
print("wrote results/gen3_front_geoms.pdf")
for row, col, tag, role, cl, mp, wr, bars in cells:
    print(f"  {tag} {role}: cycle-loss={cl:.3f} worst-ripple={wr:.1f}% ({mp:.2f}Nm)")
