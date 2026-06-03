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
    """Cubic Bézier with fixed endpoints, LS interior control points -> (4,2).
    Uniform parameterisation: matches the generator's uniform-in-parameter
    sampling, so a piece that IS a cubic Bézier is recovered ~exactly (chord-
    length would mismatch the non-uniform Bézier speed and lose fidelity)."""
    B0, B3 = seg[0], seg[-1]
    t = np.linspace(0, 1, len(seg))
    A = np.column_stack([3 * (1 - t) ** 2 * t, 3 * (1 - t) * t ** 2])
    rhs = seg - np.outer((1 - t) ** 3, B0) - np.outer(t ** 3, B3)
    sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    return np.array([B0, sol[0], sol[1], B3])


def _straight_or_fit(arc):
    """If the arc is near-collinear (broken-line segment), return the exact
    straight cubic (control pts on the chord) — no spurious bow that could make
    adjacent segments cross. Otherwise LS-fit a cubic."""
    P0, P3 = arc[0], arc[-1]
    chord = P3 - P0
    L = np.linalg.norm(chord)
    if L > 1e-9:
        s = (arc - P0) @ chord / L ** 2
        dev = np.linalg.norm(arc - (P0 + np.outer(s, chord)), axis=1).max()
        if dev < 0.05:  # straight to <0.05 mm
            return np.array([P0, P0 + chord / 3, P0 + 2 * chord / 3, P3])
    return _fit_cubic(arc)


def _eval_cubic(c, t):
    return ((1 - t) ** 3 * c[0] + 3 * (1 - t) ** 2 * t * c[1]
            + 3 * (1 - t) * t ** 2 * c[2] + t ** 3 * c[3])


def _split_cubic(c, t=0.5):
    """de Casteljau split of cubic c into two cubics tracing the SAME curve."""
    P0, P1, P2, P3 = c
    A, B, C = P0 + (P1 - P0) * t, P1 + (P2 - P1) * t, P2 + (P3 - P2) * t
    D, E = A + (B - A) * t, B + (C - B) * t
    F = D + (E - D) * t
    return np.array([P0, A, D, F]), np.array([F, E, C, P3])


class BezierSupersetGenerator(BarrierGenerator):
    def __init__(self, design, n_barriers=3, M=6, t_bridge=0.7, t_rib=0.5, t_shaft=0.5,
                 min_air=0.5, w_rod=0.5, n_per=160, corner_thresh=12.0, **kwargs):
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

    # ---- encode (warm-start): adaptive natural-piece fit -> exactly M -----
    def _adaptive(self, arc, eps, out, depth=0, maxd=9):
        """Recursively split an arc at its max-fit-error point until each piece
        is a cubic to < eps. Auto-finds vertices (broken-line) AND smooth
        bezier<->arc junctions; the len/clip guards forbid degenerate pieces."""
        c = _straight_or_fit(arc)
        u = np.linspace(0, 1, len(arc))[:, None]
        err = np.linalg.norm(_eval_cubic(c, u) - arc, axis=1)
        if err.max() < eps or len(arc) < 6 or depth >= maxd:
            out.append(c)
            return
        s = int(np.clip(err.argmax(), 2, len(arc) - 3))
        self._adaptive(arc[: s + 1], eps, out, depth + 1, maxd)
        self._adaptive(arc[s:], eps, out, depth + 1, maxd)

    def _fit_one(self, barrier):
        """Sharp corners seed the segmentation; adaptive refinement makes each
        piece a near-exact cubic (catching gentle vertices/junctions the corner
        threshold misses); de Casteljau subdivision pads losslessly to exactly M
        cubics. Returns 3M control points (anchors+interiors), closed chain."""
        P = _dedup_closed(np.asarray(barrier, float))
        n = len(P)
        seeds = sorted(set([0] + list(np.where(_turning_deg(P) > 25.0)[0])))
        eps = 0.02
        while True:
            pieces = []
            for k in range(len(seeds)):
                i0, i1 = seeds[k], seeds[(k + 1) % len(seeds)]
                idx = [j % n for j in range(i0, i0 + ((i1 - i0) % n) + 1)]
                if len(idx) >= 2:
                    self._adaptive(P[idx], eps, pieces)
            if len(pieces) <= self.M or eps > 5.0:
                break
            eps *= 1.7  # design needs > M pieces at this tol: coarsen
        # drop degenerate (near-zero-length) pieces: their coincident anchors are
        # what let neighbouring segments cross at a sharp pinch (3BL d-axis).
        kept = [c for c in pieces if np.linalg.norm(c[3] - c[0]) > 1e-3]
        if kept:
            pieces = kept
        # pad to exactly M by splitting the longest segment (lossless)
        while len(pieces) < self.M:
            li = max(range(len(pieces)),
                     key=lambda i: np.linalg.norm(pieces[i][3] - pieces[i][0]))
            l, r = _split_cubic(pieces[li])
            pieces[li:li + 1] = [l, r]
        pieces = pieces[: self.M]
        return np.array([pt for c in pieces for pt in (c[0], c[1], c[2])])

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
