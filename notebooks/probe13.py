"""Calibration probe: how much torque do the EXISTING rotor geometries make at the new
1.3 A peak-current rating? gen-2 ran at 10 A; reluctance torque ~ I^2 predicts ~(1.3/10)^2
= 1.7% of the old torque, but saturation makes that only a guess. This measures it with a
few FEA solves: for a handful of existing geometries, sweep the fundamental current ANGLE
at |I| = 1.3 A (fundamental only -> peak = |I|, so exactly on the limit) to find the MTPA
torque + ripple. Output sets the gen-3 torque demands. Usage:
  python notebooks/probe13.py --src results/gen2_v2 --n-geom 4
"""
import argparse
import json
import math
import os

import numpy as np

import gen2
import h0h1_par as P  # noqa: F401  (imported for parity/side-effect symmetry with gen2 worker)

I_PEAK = 1.3
ANGLES_DEG = [35.0, 40.0, 45.0, 50.0, 55.0]  # current angle from d-axis; brackets MTPA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="results/gen2_v2", help="gen2 npz dir for existing geometries")
    ap.add_argument("--out", default="results/gen3_500w")
    ap.add_argument("--n-geom", type=int, default=4)
    ap.add_argument("--fhz", type=float, default=50.0)
    ap.add_argument("--n-workers", type=int, default=16)
    ap.add_argument("--ncores", type=int, default=1)
    ap.add_argument("--slots", type=int, default=60)
    ap.add_argument("--phases", type=int, default=5)
    ap.add_argument("--aedt-version", default="2024.2")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    z = np.load(f"{args.src}/gen2.npz", allow_pickle=True)
    Xg, Xi, T = np.array(z["Xg"]), np.array(z["Xi"]), np.array(z["T"])
    # unique geometries with their best recorded (10 A) torque; pick a diverse few
    key = {}
    for i in range(len(T)):
        key.setdefault(tuple(np.round(Xg[i], 6)), []).append(i)
    geoms = [(np.array(g), max(T[j] for j in ids)) for g, ids in key.items()]
    geoms.sort(key=lambda gt: -gt[1])                       # by old max torque, desc
    n = args.n_geom
    pick = geoms[:max(1, n // 2)] + geoms[-(n - n // 2):]   # top + bottom (diverse)
    print(f"[probe13] {len(key)} unique geometries; probing {len(pick)} at |I|={I_PEAK} A, "
          f"{args.fhz} Hz. old-max-T (10 A): {[round(t, 1) for _, t in pick]}", flush=True)

    meta = dict(slots=args.slots, phases=args.phases, wide=False, fhz=args.fhz,
                ncores=args.ncores, aedt_version=args.aedt_version)
    # unique job id per (geometry, angle) so all solves spread across the workers in parallel
    # (rotor rebuilt per solve -- negligible vs a ~3.8 min solve, and it keeps the box busy).
    jobs, job_geom = [], {}
    for gidx, (g, _t) in enumerate(pick):
        for a in ANGLES_DEG:
            r = math.radians(a)
            dq = [I_PEAK * math.cos(r), I_PEAK * math.sin(r), 0.0, 0.0]
            jid = len(jobs); job_geom[jid] = gidx
            jobs.append((jid, list(map(float, g)), dq))
    res = gen2.eval_batch(jobs, meta, args.n_workers, "probe13", args.out)

    print("\n=== torque / ripple at |I|=1.3 A (fundamental only) ===")
    rows = []
    for jid in sorted(res):
        for row in res[jid]:
            dq, tm, rp = row[0], row[1], row[2]
            ok = row[4] if len(row) > 4 else 1
            ang = math.degrees(math.atan2(dq[1], dq[0]))
            rows.append((job_geom[jid], ang, tm, rp, ok))
    by_g = {}
    for gid, ang, tm, rp, ok in rows:
        if ok:
            by_g.setdefault(gid, []).append((ang, tm, rp))
    for gid in sorted(by_g):
        print(f"  geom {gid} (old10A max-T {pick[gid][1]:.1f} Nm):")
        best = max(by_g[gid], key=lambda x: x[1])
        for ang, tm, rp in sorted(by_g[gid]):
            star = "  <- MTPA" if (ang, tm, rp) == best else ""
            print(f"    ang {ang:5.1f} deg   T {tm:7.4f} Nm   ripple {rp:6.2f}%%{star}")
    tmax = max((tm for _g, _a, tm, _r, ok in rows if ok), default=float("nan"))
    print(f"\n[probe13] MAX torque across probed geometries at 1.3 A: {tmax:.4f} Nm", flush=True)
    if math.isfinite(tmax):
        print(f"[probe13] 500 W @ this torque -> {500.0 / tmax:.0f} rad/s "
              f"= {500.0 / tmax * 30.0 / math.pi:.0f} rpm rated speed", flush=True)
    json.dump(rows, open(f"{args.out}/probe13.json", "w"), indent=2)


if __name__ == "__main__":
    main()
