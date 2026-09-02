"""Turns as a free design variable — re-evaluate the drive-profile optimization at an
arbitrary winding turn count Nc, from EXISTING FEA pools. No new FEA.

Physics (why this is analytic and honest, unlike a voltage re-filter): the FEA field is set
by the MMF = Nc*I. At a FIXED operating point (fixed geometry AND fixed MMF, i.e. fixed
torque/ripple/iron-saturation), changing turns is a pure V<->I transformer:
    I  -> I * (Nc0/Nc)          (Nc0 = 113, the pool's build)
    psi-> psi* (Nc/Nc0)         (flux linkage ~ N)
    Rs -> Rs * (Nc/Nc0)^2       (R ~ N^2)         Lew -> Lew*(Nc/Nc0)^2
    torque, ripple, copper loss (I^2 R)  ->  INVARIANT
So we do NOT need the optimizer to have sampled new currents: every stored (geometry,
current) point is re-scaled exactly. Turns changes only which points are FEASIBLE:
    peak current  ipk = ipk0 * (Nc0/Nc)   <= I_MAX     (tightens as Nc drops)
    peak voltage  Vpk(psi,I,w,R,Lew)       <= V_MAX     (relaxes as Nc drops, V ~ Nc)
=> there is an Nc that best matches the machine to a given (I_MAX, V_MAX) bus/inverter.

CAVEAT (pool-limited): this re-evaluates the GEOMETRIES the optimizer already sampled; it
does not re-optimize the rotor. It answers "for the designs we have, what turns count fits
the bus, and what front results", not "what is the global optimum at that Nc". We union all
pools given so the geometry set is as wide as possible.

Usage:
  python turns_eval.py --pools results/gen3_500w_v,results/gen3_500w_v500 --nc 113
  python turns_eval.py --pools ... --nc-sweep 55,140,5 --v-max 400 --i-max 1.3
"""
import argparse
import math

import numpy as np

NC0 = 113.0                       # turns the pools were built at
R0, LEW0 = 19.0, 2.4e-3           # phase resistance / end-winding leakage AT Nc0
LB = np.array([0.0, 0.0, -0.39, -0.39]); UB = np.array([1.3, 1.3, 0.39, 0.39])   # dq box at Nc0
TH = np.linspace(0.0, 2 * math.pi, 361, endpoint=False)


def _pk(m1, a1, m3, a3):
    return float(np.max(np.abs(m1 * np.cos(TH + a1) + m3 * np.cos(3 * TH + a3))))


def ipk(dq):
    return _pk(math.hypot(dq[0], dq[1]), math.atan2(dq[1], dq[0]),
               math.hypot(dq[2], dq[3]), math.atan2(dq[3], dq[2]))


def vpk(fl, dq, w, R, Lew):
    Vd1 = R * dq[0] - w * (fl[1] + Lew * dq[1]); Vq1 = R * dq[1] + w * (fl[0] + Lew * dq[0])
    Vd3 = R * dq[2] - 3 * w * (fl[3] + Lew * dq[3]); Vq3 = R * dq[3] + 3 * w * (fl[2] + Lew * dq[2])
    return _pk(math.hypot(Vd1, Vq1), math.atan2(Vq1, Vd1),
               math.hypot(Vd3, Vq3), math.atan2(Vq3, Vd3))


def load_pools(dirs):
    Xg, Xi, T, R, FL = [], [], [], [], []
    for d in dirs:
        z = np.load(f"{d}/gen3.npz", allow_pickle=True)
        Xg += list(z["Xg"]); Xi += list(z["Xi"]); T += list(z["T"]); R += list(z["R"]); FL += list(z["FL"])
    return np.array(Xg), np.array(Xi), np.array(T), np.array(R), np.array(FL)


def is_nd(O):
    keep = np.ones(len(O), bool)
    for i in range(len(O)):
        for j in range(len(O)):
            if j != i and O[j, 0] <= O[i, 0] and O[j, 1] <= O[i, 1] and (O[j] < O[i]).any():
                keep[i] = False; break
    return keep


def front_at(pool, Nc, vmax, imax, demands, omegas):
    """Confirmed (cycle_loss, worst_ptp) front + diagnostics at winding count Nc."""
    Xg, Xi, T, R, FL = pool
    s = NC0 / Nc                      # current multiplier; psi/=s ; R,Lew /= s^2
    R_nc, Lew_nc = R0 / s**2, LEW0 / s**2
    key = {}
    for i in range(len(T)):
        key.setdefault(tuple(np.round(Xg[i], 6)), []).append(i)
    obj, pct, p2_ipk, p3_vfrac = [], [], [], []
    for _g, ids in key.items():
        dq = np.array([LB + Xi[i] * (UB - LB) for i in ids])          # dq at Nc0
        tt = np.array([T[i] for i in ids]); rr = np.array([R[i] for i in ids]); fl = FL[ids]
        loss = np.sum(dq ** 2, 1)                                     # MMF-referred (turns-invariant)
        ip = np.array([ipk(dq[j]) * s for j in range(len(ids))])     # peak current at Nc
        ptp = rr * tt / 100.0
        cl, mp, mpct, ok = 0.0, 0.0, 0.0, True
        binds = {}
        for Tk, w in zip(demands, omegas):
            vv = np.array([vpk(fl[j] / s, dq[j] * s, w, R_nc, Lew_nc) for j in range(len(ids))])
            fe = (ip <= imax + 1e-6) & (tt >= Tk) & (vv <= vmax + 1e-6)
            if not fe.any():
                ok = False; break
            j = int(np.where(fe)[0][np.argmin(loss[fe])])
            cl += loss[j] / len(demands)
            if ptp[j] > mp:
                mp, mpct = ptp[j], rr[j]
            binds[Tk] = (ip[j] / imax, vv[j] / vmax)
        if ok:
            obj.append((cl, mp)); pct.append(mpct)
            p2_ipk.append(binds[demands[1]][0]); p3_vfrac.append(binds[demands[2]][1])
    if not obj:
        return dict(Nc=Nc, n_feas=0, front=[], p2_ipk=None, p3_vfrac=None)
    O = np.array(obj); pct = np.array(pct); nd = is_nd(O)
    front = sorted([[float(O[i, 0]), float(O[i, 1]), float(pct[i])] for i in np.where(nd)[0]])
    return dict(Nc=Nc, n_feas=len(obj), front=front,
                p2_ipk=float(np.max(p2_ipk)), p3_vfrac=float(np.max(p3_vfrac)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", default="results/gen3_500w_v,results/gen3_500w_v500")
    ap.add_argument("--nc", type=float, default=None)
    ap.add_argument("--nc-sweep", default=None, help="min,max,step")
    ap.add_argument("--v-max", type=float, default=400.0)
    ap.add_argument("--i-max", type=float, default=1.3)
    ap.add_argument("--demands", default="4,8,2.5")
    ap.add_argument("--speeds", default="25,16,63")
    args = ap.parse_args()
    pool = load_pools(args.pools.split(","))
    demands = [float(x) for x in args.demands.split(",")]
    omegas = [2 * math.pi * float(f) for f in args.speeds.split(",")]
    ng = len({tuple(np.round(g, 6)) for g in pool[0]})
    print(f"[turns_eval] {len(pool[2])} FEA points, {ng} geometries from {args.pools}")
    print(f"             V_MAX={args.v_max} V, I_MAX={args.i_max} A, demands={demands} @ {args.speeds} Hz\n")

    if args.nc_sweep:
        lo, hi, st = (float(x) for x in args.nc_sweep.split(","))
        ncs = np.arange(lo, hi + 1e-9, st)
        print(f"{'Nc':>6} {'feas':>5} {'|front|':>7} {'min_loss':>9} {'min_ripNm':>9} "
              f"{'P2 ipk/Im':>10} {'P3 V/Vm':>9}")
        for Nc in ncs:
            r = front_at(pool, float(Nc), args.v_max, args.i_max, demands, omegas)
            if r["n_feas"] == 0:
                print(f"{Nc:6.0f} {0:5d} {0:7d}       (no feasible design)"); continue
            ml = min(f[0] for f in r["front"]); mr = min(f[1] for f in r["front"])
            print(f"{Nc:6.0f} {r['n_feas']:5d} {len(r['front']):7d} {ml:9.3f} {mr:9.2f} "
                  f"{r['p2_ipk']:10.2f} {r['p3_vfrac']:9.2f}")
        print("\n(P2 ipk/Im -> 1 = current-bound;  P3 V/Vm -> 1 = voltage-bound. "
              "The Nc where both approach 1 matches the machine to the bus.)")
    else:
        Nc = args.nc if args.nc else NC0
        r = front_at(pool, Nc, args.v_max, args.i_max, demands, omegas)
        print(f"Nc={Nc:.0f}: {r['n_feas']} feasible geoms, {len(r['front'])} on front "
              f"(current scale x{NC0/Nc:.2f}, back-EMF x{Nc/NC0:.2f})")
        for cl, mp, pc in r["front"]:
            print(f"  cycle_loss={cl:.3f}  worst_ptp={mp:.2f} Nm ({pc:.1f}%)")
        if r["n_feas"]:
            print(f"  worst-case P2 peak current = {r['p2_ipk']:.2f} of I_MAX; "
                  f"P3 peak voltage = {r['p3_vfrac']:.2f} of V_MAX")


if __name__ == "__main__":
    main()
