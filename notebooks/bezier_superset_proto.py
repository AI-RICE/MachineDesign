"""Step 2 (v2 plan): prototype the 2-D piecewise-cubic-Bézier superset and prove
it CONTAINS all three Hackl families ~exactly (geometry round-trip), where the
r(θ) spline failed at 11 mm.

Key idea — CORNER-AWARE fit: detect corners (bezier↔arc junctions, broken-line
vertices) as tangent-angle jumps, put segment breakpoints AT them (C⁰), and fit a
cubic Bézier per (sub)segment. Smooth re-entrant ends are handled because the
curve is 2-D parametric; corners are preserved because they are breakpoints.

  .venv/bin/python notebooks/bezier_superset_proto.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from machine_design.fea_emulator import load_fea_designs  # noqa: E402
from machine_design.generators import (  # noqa: E402
    HacklGenerator_3BrokenLines,
    HacklGenerator_OneLambda,
    HacklGenerator_SixLambdas,
)
from machine_design.lumped.geometry import REFERENCE_MACHINE  # noqa: E402

GENS = [("OneLambda", HacklGenerator_OneLambda), ("SixLambdas", HacklGenerator_SixLambdas),
        ("ThreeBrokenLines", HacklGenerator_3BrokenLines)]


def _dedup_closed(P):
    return P[:-1] if np.allclose(P[0], P[-1]) else P


def turning_deg(P):
    """Per-vertex turning angle (deg) for a closed polyline."""
    n = len(P)
    a = np.zeros(n)
    for i in range(n):
        v0 = P[i] - P[(i - 1) % n]
        v1 = P[(i + 1) % n] - P[i]
        n0, n1 = np.linalg.norm(v0), np.linalg.norm(v1)
        if n0 < 1e-12 or n1 < 1e-12:
            continue
        c = np.clip(np.dot(v0, v1) / (n0 * n1), -1, 1)
        a[i] = np.degrees(np.arccos(c))
    return a


def detect_corners(P, thresh_deg=12.0):
    ang = turning_deg(P)
    return np.where(ang > thresh_deg)[0]


def fit_cubic(seg):
    """Cubic Bézier with FIXED endpoints, least-squares interior control points."""
    B0, B3 = seg[0], seg[-1]
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(seg, axis=0), axis=1))]
    t = d / d[-1] if d[-1] > 0 else np.linspace(0, 1, len(seg))
    b0 = (1 - t) ** 3
    b1 = 3 * (1 - t) ** 2 * t
    b2 = 3 * (1 - t) * t ** 2
    b3 = t ** 3
    rhs = seg - np.outer(b0, B0) - np.outer(b3, B3)
    A = np.column_stack([b1, b2])
    sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)  # (2,2): rows = B1,B2
    return np.array([B0, sol[0], sol[1], B3])


def fit_chain(P, thresh_deg=12.0, target_seg_len=4.0):
    """Corner-aware closed Bézier chain: split at corners, subdivide long arcs,
    fit a cubic per sub-segment (C⁰). Returns list of (4,2) control-point arrays."""
    P = _dedup_closed(P)
    n = len(P)
    corners = detect_corners(P, thresh_deg)
    if len(corners) == 0:
        corners = np.array([0])
    # cyclic arcs between consecutive corners
    chain = []
    for k in range(len(corners)):
        i0 = corners[k]
        i1 = corners[(k + 1) % len(corners)]
        idx = [j % n for j in range(i0, i0 + ((i1 - i0) % n) + 1)]
        arc = P[idx]
        if len(arc) < 2:
            continue
        L = np.sum(np.linalg.norm(np.diff(arc, axis=0), axis=1))
        nsub = max(1, int(np.ceil(L / target_seg_len)))
        bnds = np.linspace(0, len(arc) - 1, nsub + 1).astype(int)
        for s in range(nsub):
            seg = arc[bnds[s]: bnds[s + 1] + 1]
            if len(seg) >= 2:
                chain.append(fit_cubic(seg))
    return chain


def eval_chain(chain, n_per=40):
    pts = []
    t = np.linspace(0, 1, n_per)[:, None]
    for c in chain:
        B = (1 - t) ** 3 * c[0] + 3 * (1 - t) ** 2 * t * c[1] + 3 * (1 - t) * t ** 2 * c[2] + t ** 3 * c[3]
        pts.append(B)
    return np.vstack(pts)


def roundtrip_maxerr(P, chain):
    fit = eval_chain(chain, 60)
    P = _dedup_closed(P)
    # max over original points of nearest distance to the fitted curve
    d = np.sqrt(((P[:, None, :] - fit[None, :, :]) ** 2).sum(-1)).min(1)
    return d.max(), np.sqrt((d ** 2).mean())


def good(short, cls, k=6):
    d = load_fea_designs(short, "../MachineDesign/results", None)
    idx = np.argsort(d.T_mean)[-k:]
    hk = cls(REFERENCE_MACHINE, r_stator_end=0.7, offset=0.35)
    out = []
    for j in idx:
        hk.set_parameters(hk.X_to_params(np.asarray(d.X[j], float)))
        out.append(hk.generate_barriers())
    return out


def main():
    print("2-D corner-aware Bézier-chain containment of Hackl barriers (round-trip, mm):")
    print(f"{'family':16s} | {'corners/barrier':>15} | {'max err':>8} | {'rms err':>8} | {'segs/barrier':>12} | {'DOF/barrier':>11}")
    for short, cls in GENS:
        designs = good(short, cls)
        mx, rms, ncorner, nseg = [], [], [], []
        for bars in designs:
            for b in bars:
                chain = fit_chain(b, thresh_deg=12.0, target_seg_len=4.0)
                m, r = roundtrip_maxerr(b, chain)
                mx.append(m); rms.append(r); nseg.append(len(chain))
                ncorner.append(len(detect_corners(_dedup_closed(b), 12.0)))
        dof = np.mean(nseg) * 6  # ~6 free coords per cubic in a C0 chain (approx)
        print(f"{short:16s} | {np.mean(ncorner):>15.1f} | {np.mean(mx):>8.3f} | {np.mean(rms):>8.4f} | "
              f"{np.mean(nseg):>12.1f} | {dof:>11.0f}")
    print("\n(r(θ) baseline: max ~11 mm at the re-entrant ends. Target here: ≪0.05 mm.)")


if __name__ == "__main__":
    main()
