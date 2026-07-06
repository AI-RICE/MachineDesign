"""Per-front-geometry per-demand optimal currents WITH the voltage constraint: shows the
current-bound (P2, ipk~Imax) vs voltage-bound (P3, V_pk~Vmax, field-weakened) regimes and the
dq3 content. No FEA. Usage: python analyze_front.py [run_dir]"""
import sys
import math
import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "results/gen3_500w_v"
z = np.load(f"{OUT}/gen3.npz", allow_pickle=True)
Xg, Xi, T, R, FL = [np.array(z[k]) for k in ("Xg", "Xi", "T", "R", "FL")]
LB = np.array([0, 0, -0.39, -0.39]); UB = np.array([1.3, 1.3, 0.39, 0.39])
dq_of = lambda u: LB + u * (UB - LB)
DEM = [4.0, 8.0, 2.5]; SPD = [25.0, 16.0, 63.0]; VMAX = 400.0; RS = 19.0; LEW = 2.4e-3; IMAX = 1.3
OMEGA = [2 * math.pi * f for f in SPD]
TH = np.asarray(z["theta"], float) if "theta" in z else np.linspace(0, 2 * math.pi, 361)


def ipk(dq):
    Im1, a1 = math.hypot(dq[0], dq[1]), math.atan2(dq[1], dq[0])
    Im3, a3 = math.hypot(dq[2], dq[3]), math.atan2(dq[3], dq[2])
    return float(np.max(np.abs(Im1 * np.cos(TH + a1) + Im3 * np.cos(3 * TH + a3))))


def vpk(fl, dq, w):
    # EXACT peak of the combined 1st+3rd voltage waveform over the theta grid (not the
    # conservative |V1|+|V3| triangle bound).
    Vd1 = RS * dq[0] - w * (fl[1] + LEW * dq[1]); Vq1 = RS * dq[1] + w * (fl[0] + LEW * dq[0])
    Vd3 = RS * dq[2] - 3 * w * (fl[3] + LEW * dq[3]); Vq3 = RS * dq[3] + 3 * w * (fl[2] + LEW * dq[2])
    Vm1, p1 = math.hypot(Vd1, Vq1), math.atan2(Vq1, Vd1)
    Vm3, p3 = math.hypot(Vd3, Vq3), math.atan2(Vq3, Vd3)
    return float(np.max(np.abs(Vm1 * np.cos(TH + p1) + Vm3 * np.cos(3 * TH + p3))))


key = {}
for i in range(len(T)):
    key.setdefault(tuple(np.round(Xg[i], 6)), []).append(i)
rows = []
for g, ids in key.items():
    dq = np.array([dq_of(Xi[i]) for i in ids]); tt = np.array([T[i] for i in ids])
    rr = np.array([R[i] for i in ids]); fl = FL[ids]; ll = np.sum(dq ** 2, 1)
    ip = np.array([ipk(dq[j]) for j in range(len(ids))]); ptp = rr * tt / 100
    per = []; ok = True; cl = 0; mp = 0
    for Tk, w in zip(DEM, OMEGA):
        vv = np.array([vpk(fl[j], dq[j], w) for j in range(len(ids))])
        fe = (ip <= IMAX + 1e-6) & (tt >= Tk) & (vv <= VMAX)
        if not fe.any():
            ok = False; break
        j = int(np.where(fe)[0][np.argmin(ll[fe])])
        per.append((Tk, dq[j], ip[j], vv[j], tt[j], rr[j])); cl += ll[j] / 3; mp = max(mp, ptp[j])
    if ok:
        rows.append((cl, mp, per))
rows.sort()
nd = [r for r in rows if not any((o[0] <= r[0] and o[1] < r[1]) or (o[0] < r[0] and o[1] <= r[1]) for o in rows)]
print(f"{len(rows)} feasible geoms, {len(nd)} on front  (demands {DEM} @ {SPD} Hz, V<= {VMAX} V)\n")
for cl, mp, per in nd:
    print(f"FRONT geom cycle_loss={cl:.3f} max_ptp={mp:.2f}Nm")
    for Tk, dq, ip, vv, tt, rr in per:
        r31 = math.hypot(dq[2], dq[3]) / max(math.hypot(dq[0], dq[1]), 1e-9)
        tag = "CURRENT-bound" if ip / IMAX > 0.9 else ("VOLT-bound" if vv / VMAX > 0.9 else "")
        print("  P@%3.1fNm: dq=[%+.2f,%+.2f,%+.2f,%+.2f] |i3/i1|=%2.0f%% ipk=%2.0f%%Im V=%3.0f(%2.0f%%Vm) T=%.2f rip=%.1f%%  %s"
              % (Tk, dq[0], dq[1], dq[2], dq[3], 100 * r31, 100 * ip / IMAX, vv, 100 * vv / VMAX, tt, rr, tag))
    print()
