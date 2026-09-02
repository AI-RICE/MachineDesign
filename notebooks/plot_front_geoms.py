"""3x2 rotor-geometry grid for the Pareto ENDPOINTS of the three cases, ALL on the same
pooled geometries (400V+500V union), so the figure is consistent with the unified fronts:
  rows = {400 V/Nc113 baseline, 500 V/Nc113, 400 V free-turns},  cols = {min-loss, min-ripple}.
Finds each endpoint (exact-peak voltage feasibility; turns-free rows sweep a winding grid),
builds the flux-barrier polylines via the real generator, renders one pole sector per cell.
Writes results/gen3_front_geoms.pdf. Run on bayes.
"""
import math

import matplotlib

matplotlib.use("Agg")
import h0h1_par as P
import h0h1_study as H
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

POOLS = ["results/gen3_500w_gardner"]   # 220-geom pool: 208 union + 12 from the corrected constrained-EHVI run
R0, LEW0, IMAX, NC0 = 19.0, 2.4e-3, 1.3, 113.0
LB = np.array([0.0, 0.0, -0.39, -0.39]); UB = np.array([1.3, 1.3, 0.39, 0.39])
DEM = [4.0, 8.0, 2.5]; OMEGA = [2 * math.pi * f for f in (25.0, 16.0, 63.0)]
TH = np.linspace(0.0, 2 * math.pi, 361, endpoint=False)
NCGRID = np.arange(55.0, 161.0, 5.0)
dq_of = lambda u: LB + u * (UB - LB)


def _pk(m1, a1, m3, a3):
    return float(np.max(np.abs(m1 * np.cos(TH + a1) + m3 * np.cos(3 * TH + a3))))


def ipk(dq):
    return _pk(math.hypot(dq[0], dq[1]), math.atan2(dq[1], dq[0]), math.hypot(dq[2], dq[3]), math.atan2(dq[3], dq[2]))


def vpk(fl, dq, w):
    Vd1 = R0 * dq[0] - w * (fl[1] + LEW0 * dq[1]); Vq1 = R0 * dq[1] + w * (fl[0] + LEW0 * dq[0])
    Vd3 = R0 * dq[2] - 3 * w * (fl[3] + LEW0 * dq[3]); Vq3 = R0 * dq[3] + 3 * w * (fl[2] + LEW0 * dq[2])
    return _pk(math.hypot(Vd1, Vq1), math.atan2(Vq1, Vd1), math.hypot(Vd3, Vq3), math.atan2(Vq3, Vd3))


def load_union():
    Xg, Xi, T, R, FL = [], [], [], [], []
    for d in POOLS:
        z = np.load(f"{d}/gen3.npz", allow_pickle=True)
        Xg += list(z["Xg"]); Xi += list(z["Xi"]); T += list(z["T"]); R += list(z["R"]); FL += list(z["FL"])
    return list(map(np.array, (Xg, Xi, T, R, FL)))


def endpoints(pool, mode, vmax):
    """mode: 'fixed' (Nc=113 at vmax) or 'free' (best over turns). Returns (min-loss g, min-ripple g)."""
    Xg, Xi, T, R, FL = pool
    key = {}
    for i in range(len(T)):
        key.setdefault(tuple(np.round(Xg[i], 6)), []).append(i)
    rows = []
    for g, ids in key.items():
        dq = np.array([dq_of(Xi[i]) for i in ids]); tt = np.array([T[i] for i in ids])
        rr = np.array([R[i] for i in ids]); fl = FL[ids]; ll = np.sum(dq ** 2, 1)
        ip = np.array([ipk(dq[j]) for j in range(len(ids))]); ptp = rr * tt / 100
        vb = [np.array([vpk(fl[j], dq[j], w) for j in range(len(ids))]) for w in OMEGA]
        ncs = [113.0] if mode == "fixed" else NCGRID
        best = None
        for Nc in ncs:
            s = NC0 / Nc; cl, mp, ok = 0.0, 0.0, True
            for k, Tk in enumerate(DEM):
                fe = (ip * s <= IMAX + 1e-6) & (tt >= Tk) & (vb[k] / s <= vmax + 1e-6)
                if not fe.any():
                    ok = False; break
                j = int(np.where(fe)[0][np.argmin(ll[fe])]); cl += ll[j] / 3; mp = max(mp, ptp[j])
            if ok and (best is None or cl < best[0]):
                best = (cl, mp, Nc)
        if best is not None:
            rows.append((best[0], best[1], np.array(g)))
    nd = [r for r in rows if not any((o[0] <= r[0] and o[1] < r[1]) or (o[0] < r[0] and o[1] <= r[1]) for o in rows)]
    return min(nd, key=lambda r: r[0]), min(nd, key=lambda r: r[1])   # (min-loss, min-ripple)


pool = load_union()
gd = P.open_isolated_design("plotg", 95, "2024.2", slots=60, phases=5)
gen = P.make_generator(gd, False); lb, ub = H.geom_bounds_arrays(gen)
rmin, rmax = float(gd.rotor_r_min), float(gd.rotor_r_max)
CASES = [("400 V, Nc=113", "fixed", 400.0), ("500 V, Nc=113", "fixed", 500.0),
         ("400 V, free turns", "free", 400.0)]
cells = []
for row, (tag, mode, vmax) in enumerate(CASES):
    ml, mr = endpoints(pool, mode, vmax)
    for col, (role, r) in enumerate([("min-loss", ml), ("min-ripple", mr)]):
        cl, mp, g = r
        bars = [np.asarray(b, float)[:, :2] for b in H.build_barriers(gen, g, lb, ub)]
        cells.append((row, col, tag, role, cl, mp, bars))
gd.close_project()

allpts = np.vstack([b for c in cells for b in c[6]])
th = np.arctan2(allpts[:, 1], allpts[:, 0]); th0, th1 = th.min(), th.max()
fig, axes = plt.subplots(3, 2, figsize=(7.0, 9.6), subplot_kw=dict(aspect="equal"))
COL = {"400 V, Nc=113": "tab:blue", "500 V, Nc=113": "tab:red", "400 V, free turns": "tab:green"}
for row, col, tag, role, cl, mp, bars in cells:
    ax = axes[row, col]; a = np.linspace(th0, th1, 120)
    for r in (rmin, rmax):
        ax.plot(r * np.cos(a), r * np.sin(a), color="0.55", lw=0.8)
    for ang in (th0, th1):
        ax.plot([rmin * np.cos(ang), rmax * np.cos(ang)], [rmin * np.sin(ang), rmax * np.sin(ang)], color="0.55", lw=0.8)
    for b in bars:
        ax.add_patch(Polygon(b, closed=True, facecolor=COL[tag], edgecolor="k", lw=0.5, alpha=0.55))
    ax.set_title(f"{tag}, {role}\ncycle-loss {cl:.2f}, {mp:.2f} Nm p-p", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(allpts[:, 0].min() - 2, allpts[:, 0].max() + 2)
    ax.set_ylim(allpts[:, 1].min() - 2, allpts[:, 1].max() + 2)
fig.tight_layout()
fig.savefig("results/gen3_front_geoms.pdf", bbox_inches="tight")
print("wrote results/gen3_front_geoms.pdf")
for row, col, tag, role, cl, mp, bars in cells:
    print(f"  {tag:18s} {role:10s}: cycle-loss={cl:.3f} worst-ptp={mp:.2f}Nm")
