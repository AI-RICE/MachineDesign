"""Post-run voltage diagnostic on the gen3_500w_v pool (which has flux). For each demand's
electrical speed, find the MAX torque achievable under V_pk<=V_MAX across the FEA'd currents
per geometry -> shows whether each demand is voltage-feasible and at what torque. No FEA."""
import math
import sys

import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "results/gen3_500w_v"
z = np.load(f"{OUT}/gen3.npz", allow_pickle=True)
Xg, Xi, T, R, FL = [np.array(z[k]) for k in ("Xg", "Xi", "T", "R", "FL")]
LB = np.array([0, 0, -0.39, -0.39]); UB = np.array([1.3, 1.3, 0.39, 0.39])
dq_of = lambda u: LB + u * (UB - LB)
DEM = [4.0, 8.0, 6.0]; SPD = [25.0, 16.0, 63.0]; VMAX = 400.0; RS = 19.0; LEW = 2.4e-3
OMEGA = [2 * math.pi * f for f in SPD]


def vpk(fl, dq, w):
    Vd1 = RS * dq[0] - w * (fl[1] + LEW * dq[1]); Vq1 = RS * dq[1] + w * (fl[0] + LEW * dq[0])
    Vd3 = RS * dq[2] - 3 * w * (fl[3] + LEW * dq[3]); Vq3 = RS * dq[3] + 3 * w * (fl[2] + LEW * dq[2])
    return math.hypot(Vd1, Vq1) + math.hypot(Vd3, Vq3)


key = {}
for i in range(len(T)):
    key.setdefault(tuple(np.round(Xg[i], 6)), []).append(i)
print(f"{len(key)} geometries, {len(T)} FEA points; V_MAX={VMAX} V\n")
print("per demand: how many geometries can meet it under V_pk<=V_MAX, and best achievable T")
for Tk, f, w in zip(DEM, SPD, OMEGA):
    feas_geoms = 0; best_T_underV = 0.0; best_T_anygeom = []
    for _g, ids in key.items():
        tt = np.array([T[i] for i in ids]); fl = FL[ids]
        dq = np.array([dq_of(Xi[i]) for i in ids])
        vv = np.array([vpk(fl[j], dq[j], w) for j in range(len(ids))])
        undV = vv <= VMAX
        if undV.any():
            maxT = float(tt[undV].max()); best_T_anygeom.append(maxT)
            if maxT >= Tk:
                feas_geoms += 1
            best_T_underV = max(best_T_underV, maxT)
    med = np.median(best_T_anygeom) if best_T_anygeom else 0.0
    print(f"  P@{Tk:.1f}Nm @ {f:.0f}Hz: {feas_geoms}/{len(key)} geoms feasible | "
          f"max T under V across all = {best_T_underV:.2f} Nm | median per-geom max-T-under-V = {med:.2f}")
print("\nconstant-power check (500 W): T=P/w -> "
      + ", ".join(f"{f:.0f}Hz:{500.0/(2*math.pi*f/2):.2f}Nm" for f in SPD) + "  (mech, p=2)")
