"""Unified SynRM rotor parameterisation v2 — 2-D piecewise-cubic-Bézier superset.

Each barrier boundary is a CLOSED chain of M cubic Bézier segments in 2-D, with
free control points and C0 joints (corners emerge where tangents misalign). This
contains all three Hackl families ~exactly (Steps 2/3), handles re-entrant ends
and corners that the r(θ) family could not, and is feasibility-checked by a cheap
geometry validator (no FEA, no feasibility GP — used as a hard constraint inside
the acqf optimiser).

BO vector per barrier = 3M control points (M anchors + 2M interior), flattened
(x,y). N barriers -> D = 6*M*N. M chosen for D ≈ 100 (default M=6 -> D=108).

See docs/PARAMETERISATION_v2.md.
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import LineString

from .generators import BarrierGenerator


# ---- bezier primitives (corner-aware fit / eval) -----------------------------
def _dedup_closed(P):
    return P[:-1] if np.allclose(P[0], P[-1]) else P


def _turning_deg(P):
    n = len(P)
    a = np.zeros(n)
    for i in range(n):
        v0 = P[i] - P[(i - 1) % n]
        v1 = P[(i + 1) % n] - P[i]
        n0, n1 = np.linalg.norm(v0), np.linalg.norm(v1)
        if n0 > 1e-12 and n1 > 1e-12:
            a[i] = np.degrees(np.arccos(np.clip(np.dot(v0, v1) / (n0 * n1), -1, 1)))
    return a


def _fit_cubic(seg):
    """Cubic Bézier with fixed endpoints, LS interior control points -> (4,2)."""
    B0, B3 = seg[0], seg[-1]
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(seg, axis=0), axis=1))]
    t = d / d[-1] if d[-1] > 0 else np.linspace(0, 1, len(seg))
    A = np.column_stack([3 * (1 - t) ** 2 * t, 3 * (1 - t) * t ** 2])
    rhs = seg - np.outer((1 - t) ** 3, B0) - np.outer(t ** 3, B3)
    sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    return np.array([B0, sol[0], sol[1], B3])


def _eval_cubic(c, t):
    return ((1 - t) ** 3 * c[0] + 3 * (1 - t) ** 2 * t * c[1]
            + 3 * (1 - t) * t ** 2 * c[2] + t ** 3 * c[3])


class BezierSupersetGenerator(BarrierGenerator):
    def __init__(self, design, n_barriers=3, M=6, t_bridge=0.7, t_rib=0.5, t_shaft=0.5,
                 min_air=0.5, w_rod=0.5, n_per=40, corner_thresh=12.0, **kwargs):
        self.N = int(n_barriers)
        self.M = int(M)
        self.t_bridge, self.t_rib, self.t_shaft, self.min_air = t_bridge, t_rib, t_shaft, min_air
        self.w_rod = w_rod
        self.n_per = int(n_per)
        self.corner_thresh = corner_thresh
        super().__init__(design, r_stator_end=t_bridge, offset=w_rod / np.sqrt(2.0), **kwargs)
        self.n_ctrl = 3 * self.M            # control points per barrier (closed chain)
        self.block = self.n_ctrl * 2        # flattened DOF per barrier
        self._params: list | None = None

    # ---- decode ----------------------------------------------------------
    def _decode_one(self, ctrl):
        """ctrl: (3M,2) -> closed barrier polyline."""
        m, t = self.M, np.linspace(0, 1, self.n_per)[:, None]
        pts = []
        for i in range(m):
            c = np.array([ctrl[3 * i], ctrl[3 * i + 1], ctrl[3 * i + 2], ctrl[(3 * i + 3) % (3 * m)]])
            pts.append(_eval_cubic(c, t))
        loop = np.vstack(pts)
        # light radial clamp to [r_min, R] (keeps in-bounds + bridge>=t_bridge;
        # corrects sub-mm bezier overshoot at the surface; minimal distortion)
        r = np.linalg.norm(loop, axis=1)
        rc = np.clip(r, self.r_min, self.R)
        loop = loop * (rc / np.maximum(r, 1e-12))[:, None]
        return np.vstack([loop, loop[:1]])  # close

    def generate_barriers(self):
        if self._params is None:
            raise RuntimeError("call set_parameters() first")
        return [self._decode_one(c) for c in self._params]

    # ---- encode (warm-start): corner-aware fit -> fixed-M closed chain ----
    def _fit_one(self, barrier):
        P = _dedup_closed(np.asarray(barrier, float))
        n = len(P)
        corners = np.where(_turning_deg(P) > self.corner_thresh)[0]
        if len(corners) == 0:
            corners = np.array([0])
        corners = np.sort(corners)
        # arc lengths between consecutive (cyclic) corners
        arcs = []
        for k in range(len(corners)):
            i0, i1 = corners[k], corners[(k + 1) % len(corners)]
            idx = [j % n for j in range(i0, i0 + ((i1 - i0) % n) + 1)]
            arc = P[idx]
            L = float(np.sum(np.linalg.norm(np.diff(arc, axis=0), axis=1))) if len(arc) > 1 else 0.0
            arcs.append((arc, L))
        # allocate M segments across arcs (>=1 each), proportional to length
        Ltot = sum(L for _, L in arcs) or 1.0
        nseg = [max(1, int(round(self.M * L / Ltot))) for _, L in arcs]
        # adjust to sum exactly M
        while sum(nseg) > self.M and max(nseg) > 1:
            nseg[int(np.argmax(nseg))] -= 1
        while sum(nseg) < self.M:
            nseg[int(np.argmax([L for _, L in arcs]))] += 1
        # fit cubics; collect anchors+interiors (skip each cubic's P3 = next P0)
        ctrl = []
        for (arc, _), ns in zip(arcs, nseg):
            bnds = np.linspace(0, len(arc) - 1, ns + 1).astype(int)
            for s in range(ns):
                seg = arc[bnds[s]: bnds[s + 1] + 1]
                if len(seg) < 2:
                    seg = arc[bnds[s]: bnds[s] + 2] if bnds[s] + 2 <= len(arc) else arc[-2:]
                c = _fit_cubic(seg)
                ctrl.extend([c[0], c[1], c[2]])
        ctrl = np.array(ctrl[: 3 * self.M])
        if len(ctrl) < 3 * self.M:  # pad (rare)
            ctrl = np.vstack([ctrl, np.repeat(ctrl[-1:], 3 * self.M - len(ctrl), axis=0)])
        return ctrl

    def fit_barriers(self, barriers):
        polys = sorted((np.asarray(b, float) for b in barriers), key=lambda p: np.linalg.norm(p, axis=1).min())
        return np.concatenate([self._fit_one(p).reshape(-1) for p in polys[: self.N]])

    # ---- interface -------------------------------------------------------
    def X_to_params(self, X):
        X = np.asarray(X, float)
        return [X[b * self.block:(b + 1) * self.block].reshape(self.n_ctrl, 2) for b in range(self.N)]

    def set_parameters(self, params):
        if isinstance(params, np.ndarray) and params.ndim == 1:
            params = self.X_to_params(params)
        self._params = [np.asarray(p, float).reshape(self.n_ctrl, 2) for p in params]

    def random_parameters(self):
        raise NotImplementedError("use prior sampling around warm-start (Step 4b)")

    @property
    def bounds(self):
        lo = np.tile([0.0, 0.0], self.n_ctrl * self.N)
        hi = np.tile([self.r_max, self.r_max], self.n_ctrl * self.N)
        return lo, hi

    # ---- cheap geometry validator (feasibility; no FEA) ------------------
    def feasible_barriers(self, barriers, tol=1e-8):
        # in-bounds + simple + non-self-intersecting (inherited)
        if not super().feasible_barriers(barriers, tol):
            return False
        ls = [LineString(b) for b in sorted(barriers, key=lambda p: np.linalg.norm(p, axis=1).min())]
        # min iron: shaft, surface, inter-barrier
        rmins = [np.linalg.norm(b, axis=1).min() for b in barriers]
        rmaxs = [np.linalg.norm(b, axis=1).max() for b in barriers]
        if min(rmins) - self.r_min < self.t_shaft - tol:
            return False
        if self.r_max - max(rmaxs) < self.t_bridge - tol:
            return False
        for i in range(len(ls) - 1):
            if ls[i].distance(ls[i + 1]) < self.t_rib - tol:
                return False
        return True
