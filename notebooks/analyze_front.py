"""Post-run analysis of gen3 front: per-demand min-loss optimal currents, whether P2 is
current-bound (ipk near I_MAX), and the dq3 (3rd-harmonic) currents + whether they add torque
or only shape the waveform. No FEA. Usage: python analyze_front.py [run_dir]"""
import sys
import numpy as np
import gen2

OUT = sys.argv[1] if len(sys.argv) > 1 else "results/gen3_500w"
z = np.load(f"{OUT}/gen3.npz", allow_pickle=True)
Xg, Xi, T, R = [np.array(z[k]) for k in ("Xg", "Xi", "T", "R")]
LB = np.array([0, 0, -0.39, -0.39]); UB = np.array([1.3, 1.3, 0.39, 0.39])
dq_of = lambda u: LB + u * (UB - LB)
DEM = [4.0, 8.0, 6.0]; IMAX = 1.3

key = {}
for i in range(len(T)):
    key.setdefault(tuple(np.round(Xg[i], 6)), []).append(i)

rows = []
for g, ids in key.items():
    dq = np.array([dq_of(Xi[i]) for i in ids]); tt = np.array([T[i] for i in ids]); rr = np.array([R[i] for i in ids])
    ll = np.sum(dq ** 2, 1); ip = np.array([gen2.ipk_of(dq[j]) for j in range(len(dq))]); ptp = rr * tt / 100
    base = ip <= IMAX + 1e-6; per = []; ok = True; cl = 0.0; mp = 0.0
    for Tk in DEM:
        fe = base & (tt >= Tk)
        if not fe.any():
            ok = False; break
        j = int(np.where(fe)[0][np.argmin(ll[fe])])
        per.append((Tk, dq[j], ip[j], tt[j], rr[j])); cl += ll[j] / 3; mp = max(mp, ptp[j])
    if ok:
        rows.append((cl, mp, g, per))
rows.sort()
nd = [r for r in rows if not any((o[0] <= r[0] and o[1] < r[1]) or (o[0] < r[0] and o[1] <= r[1]) for o in rows)]
print(f"{len(rows)} feasible geoms, {len(nd)} on front\n")
for cl, mp, g, per in nd:
    print(f"FRONT geom cycle_loss={cl:.3f} max_ptp={mp:.2f}Nm")
    for Tk, dq, ip, tt, rr in per:
        r31 = np.hypot(dq[2], dq[3]) / max(np.hypot(dq[0], dq[1]), 1e-9)
        print("   P@%2.0fNm: dq=[%.2f,%.2f,%+.2f,%+.2f]  |i3|/|i1|=%2.0f%%  ipk=%.2f (%2.0f%% Imax)  T=%.2f  rip=%.1f%%"
              % (Tk, dq[0], dq[1], dq[2], dq[3], 100 * r31, ip, 100 * ip / IMAX, tt, rr))
    print()
