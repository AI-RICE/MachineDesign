import pickle
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.interpolate import BSpline, CubicSpline
from shapely.geometry import LineString

from .geometry import rotate

if TYPE_CHECKING:
    from .design import Design


def get_arc(R: float, start_deg: float, end_deg: float, n_points: int) -> np.ndarray:
    angles = np.linspace(np.radians(start_deg), np.radians(end_deg), n_points)

    x = R * np.cos(angles)
    y = R * np.sin(angles)

    return np.column_stack((x, y))


def signed_distance(P: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a * P[:, 0] + b * P[:, 1] + c


def check_barrier(barrier: np.ndarray) -> None:
    if not np.array_equal(barrier[0], barrier[-1]):
        raise ValueError("Barrier must start and end at the same point")


def save_params(params, file_name: str) -> None:
    with open(file_name, "wb") as handle:
        pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)


class BarrierGenerator(ABC):
    # TODO: rename offset
    def __init__(
        self, design: "Design", r_stator_end: float, offset: None | float = None, n_curve: int = 500, n_flat: int = 20
    ) -> None:
        self.r_max = design.rotor_r_max
        self.r_min = design.rotor_r_min
        self.R = self.r_max - r_stator_end
        self.offset = offset
        self.n_curve = n_curve
        self.n_flat = n_flat

    @abstractmethod
    def generate_barriers(self) -> list[np.ndarray]:
        pass

    @abstractmethod
    def random_parameters(self) -> Any:
        pass

    @abstractmethod
    def set_parameters(self, params) -> None:
        pass

    @abstractmethod
    def X_to_params(self, X: np.ndarray) -> tuple[Any, ...]:
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def feasible_barriers(self, barriers: list[np.ndarray], tol: float = 1e-8) -> bool:
        # Check whether the barriers are within bounds
        for barrier in barriers:
            norm_points = np.linalg.norm(barrier, axis=1)
            if any(norm_points > self.R + tol) or any(norm_points < self.r_min - tol):
                return False

        n = len(barriers)
        curves = [LineString(barrier) for barrier in barriers]

        # Check if curve intersects itself
        for curve in curves:
            if not curve.is_simple:
                return False

        # Check if two curves intersect themselves
        for i in range(n):
            for j in range(i + 1, n):
                if curves[i].intersects(curves[j]):
                    return False
        return True

    def split_barrier(self, barrier: np.ndarray) -> list[np.ndarray]:
        assert self.offset is not None
        check_barrier(barrier)

        lam = np.linspace(0, 1, self.n_flat)[:, None]
        barriers = []
        for a, b, c in zip([1.0, -1.0], [-1.0, 1.0], [-self.offset, -self.offset]):
            f = signed_distance(barrier, a, b, c)
            f_is_positive = f > 0
            mask_positive = np.where(f_is_positive)[0]

            barrier_new = [barrier[mask_positive[0]]]
            for i in range(len(mask_positive) - 1):
                i1 = mask_positive[i]
                i2 = mask_positive[i + 1]

                if i2 - i1 == 1:
                    barrier_new.append(barrier[i2])
                else:
                    interp = (1 - lam) * barrier[i1] + lam * barrier[i2]
                    barrier_new.extend(interp[1:])
            if not np.array_equal(barrier_new[-1], barrier_new[0]):
                interp = (1 - lam) * barrier_new[-1] + lam * barrier_new[0]
                barrier_new.extend(interp[1:])

            if len(barrier_new) > 1:
                barrier_new = np.array(barrier_new)
                check_barrier(barrier_new)
                barriers.append(barrier_new)
        return barriers

    def split_barriers(self, barriers: list[np.ndarray]) -> list[np.ndarray]:
        return [x for barrier in barriers for x in self.split_barrier(barrier)]


class FourStupid(BarrierGenerator):
    def __init__(self, design, r_stator_end, der1=1.0, der2=1.0, symmetric=True, **kwargs) -> None:
        self.der1 = der1
        self.der2 = der2
        self.w_mins_base = np.array([3, 2.5, 2.5, 2]) - 1.0
        self.symmetric = symmetric
        super().__init__(design, r_stator_end, **kwargs)

    def _create_barrier(
        self,
        y_min,
        w_min,
        y_mid,
        w_mid,
        theta,
        w_max,
    ):

        theta1 = (theta + 45) / 180 * np.pi
        x_max1 = self.R * np.cos(theta1)
        y_max1 = self.R * np.sin(theta1)

        x1 = [0, x_max1 / 2, x_max1]
        y1 = [y_min, y_mid, y_max1]
        f1 = CubicSpline(x1, y1, bc_type=((1, 0), (1, self.der1)))

        theta2 = theta1 + w_max / self.r_max
        x_max2 = self.R * np.cos(theta2)
        y_max2 = self.R * np.sin(theta2)

        s = w_mid / np.sqrt(1 + self.der1**2 / 4)
        x2 = [0, x1[1] - s * self.der1 / 2, x_max2]
        y2 = [y_min + w_min, y1[1] + s, y_max2]
        f2 = CubicSpline(x2, y2, bc_type=((1, 0), (1, self.der2)))

        x_interp1 = np.linspace(x1[0], x1[-1], self.n_curve)
        x_interp2 = np.linspace(x2[0], x2[-1], self.n_curve)
        # TODO: use more points for connecting?
        x_all = np.concatenate((x_interp1, x_interp2[::-1]))
        y_all = np.concatenate((f1(x_interp1), f2(x_interp2)[::-1]))
        if self.symmetric:
            x_all = np.concatenate((x_all, -x_all[::-1][1:]))
            y_all = np.concatenate((y_all, y_all[::-1][1:]))
        x_all, y_all = rotate(x_all, y_all, -45)
        return x_all, y_all

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lb = np.array([0, 0, 0, 0, 13, 2, 2, 2])
        ub = np.array([0.5, 0.5, 0.5, 0.5, 17, 5, 5, 4])
        return lb, ub

    def random_parameters(self):
        rand1 = 0.5 * np.random.random(4)
        rand2 = 13 + 4 * np.random.random()
        rand3 = 2 + 3 * np.random.random()
        rand4 = 2 + 3 * np.random.random()
        rand5 = 2 + 2 * np.random.random()
        return rand1, rand2, rand3, rand4, rand5

    def set_parameters(self, params) -> None:
        rand1, rand2, rand3, rand4, rand5 = params
        self.w_mins = self.w_mins_base + rand1
        y_min0 = rand2
        y_min1 = y_min0 + self.w_mins[0] + rand3
        y_min2 = y_min1 + self.w_mins[1] + rand4
        y_min3 = y_min2 + self.w_mins[2] + rand5
        self.w_mids = self.w_mins
        self.y_mins = np.array([y_min0, y_min1, y_min2, y_min3])
        self.y_mids = self.y_mins + np.array([2, 1.5, 1, 0.5])
        self.thetas = [1, 8, 15, 21]
        self.w_maxs = self.w_mins - np.array([0.5, 0.5, 0.5, 0])

    def X_to_params(self, X: np.ndarray):
        return X[:4], X[4], X[5], X[6], X[7]

    def generate_barriers(self) -> list[np.ndarray]:
        barriers = []
        for args in zip(self.y_mins, self.w_mins, self.y_mids, self.w_mids, self.thetas, self.w_maxs):
            x_all, y_all = self._create_barrier(*args)
            xy_all = np.vstack((x_all, y_all)).T
            barriers.append(xy_all)
        return barriers


class AbstractHacklGenerator(BarrierGenerator):
    def __init__(self, design, r_stator_end, **kwargs):
        self.phis_inner_min = np.asarray([4.0, 17.3, 30.6])
        self.phis_inner_max = np.asarray([10.2, 23.5, 36.8])
        self.phis_outer_min = np.asarray([10.6, 24.0, 37.3])
        self.phis_outer_max = np.asarray([16.8, 30.2, 44.0])
        self.n_barriers = len(self.phis_inner_min)
        super().__init__(design, r_stator_end, **kwargs)

    @abstractmethod
    def _bezier_point(self, x: float, y: float, is_inner: bool, order: int) -> tuple[float, float]:
        pass

    def random_parameters(self):
        phis_inner = self.phis_inner_min + (self.phis_inner_max - self.phis_inner_min) * np.random.rand(self.n_barriers)
        phis_outer = self.phis_outer_min + (self.phis_outer_max - self.phis_outer_min) * np.random.rand(self.n_barriers)
        return phis_inner, phis_outer

    def set_parameters(self, params) -> None:
        phis_inner, phis_outer = params
        self.phis_inner = phis_inner
        self.phis_outer = phis_outer

    def generate_barriers(self) -> list[np.ndarray]:
        barriers = []
        for i, (phi_inner, phi_outer) in enumerate(zip(self.phis_inner, self.phis_outer)):
            # Generate the longer part of the barrier
            pts_outer = self._get_bezier_curve(phi_outer, False, i)
            pts_inner = self._get_bezier_curve(phi_inner, True, i)

            # Generate the arcs at the end
            arc_top = get_arc(self.R, 90 - phi_outer, 90 - phi_inner, self.n_flat)
            arc_bottom = get_arc(self.R, phi_inner, phi_outer, self.n_flat)

            # Merge them together
            barrier = np.concatenate((pts_outer, arc_top[1:-1], pts_inner[::-1], arc_bottom[1:]))
            barriers.append(barrier)
        return barriers

    def _get_bezier_curve(self, phi_deg, is_inner, i):
        phi_rad = np.radians(phi_deg)
        x_end = self.R * np.cos(phi_rad)
        y_end = self.R * np.sin(phi_rad)
        x_bezier, y_bezier = self._bezier_point(x_end, y_end, is_inner, i)

        r0 = np.array([x_end, y_end])
        r1 = np.array([x_bezier, y_bezier])
        r2 = np.array([y_bezier, x_bezier])
        r3 = np.array([y_end, x_end])

        # Formula 33: Polynomial expansion to generate continuous points
        z_vals = np.linspace(0, 1, self.n_curve)[:, None]
        return (
            (1 - z_vals) ** 3 * r0
            + 3 * (1 - z_vals) ** 2 * z_vals * r1
            + 3 * (1 - z_vals) * z_vals**2 * r2
            + z_vals**3 * r3
        )


class HacklGenerator_OneLambda(AbstractHacklGenerator):
    def __init__(self, design, r_stator_end, **kwargs):
        self.lam_min = 0.25
        self.lam_max = 0.45
        super().__init__(design, r_stator_end, **kwargs)

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lb = np.concatenate((self.phis_inner_min, self.phis_outer_min, [self.lam_min]))
        ub = np.concatenate((self.phis_inner_max, self.phis_outer_max, [self.lam_max]))
        return lb, ub

    def random_parameters(self):
        pars = super().random_parameters()
        lam = self.lam_min + (self.lam_max - self.lam_min) * np.random.rand(1)[0]
        return *pars, lam

    def set_parameters(self, params) -> None:
        phis_inner, phis_outer, lam = params
        super().set_parameters((phis_inner, phis_outer))
        self.lam = lam

    def X_to_params(self, X: np.ndarray):
        return X[:3], X[3:6], X[6]

    def _bezier_point(self, x: float, y: float, is_inner: bool, order: int) -> tuple[float, float]:
        return self.lam * x + (1 - self.lam) * y, y


class HacklGenerator_TwoLambdas(AbstractHacklGenerator):
    def __init__(self, design, r_stator_end, **kwargs):
        self.lam_inner_min = 0.2
        self.lam_inner_max = 0.5
        self.lam_outer_min = 0.2
        self.lam_outer_max = 0.5
        super().__init__(design, r_stator_end, **kwargs)

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lb = np.concatenate((self.phis_inner_min, self.phis_outer_min, [self.lam_inner_min, self.lam_outer_min]))
        ub = np.concatenate((self.phis_inner_max, self.phis_outer_max, [self.lam_inner_max, self.lam_outer_max]))
        return lb, ub

    def random_parameters(self):
        pars = super().random_parameters()
        lam_inner = self.lam_inner_min + (self.lam_inner_max - self.lam_inner_min) * np.random.rand(1)[0]
        lam_outer = self.lam_outer_min + (self.lam_outer_max - self.lam_outer_min) * np.random.rand(1)[0]
        return *pars, lam_inner, lam_outer

    def set_parameters(self, params) -> None:
        phis_inner, phis_outer, lam_inner, lam_outer = params
        super().set_parameters((phis_inner, phis_outer))
        self.lam_inner = lam_inner
        self.lam_outer = lam_outer

    def X_to_params(self, X: np.ndarray):
        return X[:3], X[3:6], X[6], X[7]

    def _bezier_point(self, x: float, y: float, is_inner: bool, order: int) -> tuple[float, float]:
        lam = self.lam_inner if is_inner else self.lam_outer
        return lam * x + (1 - lam) * y, y


# 6 angles + 6 lam for each barrier = 12 parameters
class HacklGenerator_SixLambdas(AbstractHacklGenerator):
    def __init__(self, design, r_stator_end, **kwargs):
        # Tiered bounds to prevent intersection
        self.lam_inner_min = np.array([0.20, 0.31, 0.41])
        self.lam_inner_max = np.array([0.25, 0.35, 0.45])

        self.lam_outer_min = np.array([0.26, 0.36, 0.46])
        self.lam_outer_max = np.array([0.30, 0.40, 0.50])

        super().__init__(design, r_stator_end, **kwargs)

    @property
    def bounds(self):
        lb = np.concatenate((self.phis_inner_min, self.phis_outer_min, self.lam_inner_min, self.lam_outer_min))
        ub = np.concatenate((self.phis_inner_max, self.phis_outer_max, self.lam_inner_max, self.lam_outer_max))
        return lb, ub

    def random_parameters(self):
        phis_inner, phis_outer = super().random_parameters()
        lam_inner = np.random.uniform(self.lam_inner_min, self.lam_inner_max)
        lam_outer = np.random.uniform(self.lam_outer_min, self.lam_outer_max)
        return phis_inner, phis_outer, lam_inner, lam_outer

    def set_parameters(self, params):
        phis_inner, phis_outer, lam_inner, lam_outer = params
        super().set_parameters((phis_inner, phis_outer))
        self.lam_inner = np.asarray(lam_inner)
        self.lam_outer = np.asarray(lam_outer)

    def X_to_params(self, X):
        # 12 params: phis_in(3), phis_out(3), lam_in(3), lam_out(3)
        return X[:3], X[3:6], X[6:9], X[9:12]

    def _bezier_point(self, x, y, is_inner, order):
        lam = self.lam_inner[order] if is_inner else self.lam_outer[order]
        return lam * x + (1 - lam) * y, y


class HacklGenerator_3BrokenLines(AbstractHacklGenerator):
    def __init__(self, design, r_stator_end, **kwargs):
        super().__init__(design, r_stator_end, **kwargs)

        # Inner Bezier weight (lam)
        self.lam_min, self.lam_max = 0.30, 0.45

        # Outer broken line controls: r (radial distance), L (straight length)
        self.r_ctrl_min = np.array([22.5, 30.5, 36.5])
        self.r_ctrl_max = np.array([24.5, 32.5, 38.0])

        self.L_min = np.array([2.0, 2.0, 0.0])
        self.L_max = np.array([18.0, 12.0, 1.5])

    @property
    def bounds(self):
        # 13 params: phis_in(3), phis_out(3), lam(1), r(3), L(3)
        lb = np.concatenate((self.phis_inner_min, self.phis_outer_min, [self.lam_min], self.r_ctrl_min, self.L_min))
        ub = np.concatenate((self.phis_inner_max, self.phis_outer_max, [self.lam_max], self.r_ctrl_max, self.L_max))
        return lb, ub

    def random_parameters(self):
        phi_inner, phi_outer = super().random_parameters()
        lam = np.random.uniform(self.lam_min, self.lam_max)
        r_vals = np.random.uniform(self.r_ctrl_min, self.r_ctrl_max)
        L_vals = np.random.uniform(self.L_min, self.L_max)
        return phi_inner, phi_outer, lam, r_vals, L_vals

    def set_parameters(self, params):
        phi_inner, phi_outer, self.lam, r_vals, L_vals = params
        super().set_parameters((phi_inner, phi_outer))
        self.r_vals = np.asarray(r_vals)
        self.L_vals = np.asarray(L_vals)

    def X_to_params(self, X):
        return X[:3], X[3:6], X[6], X[7:10], X[10:13]

    def _bezier_point(self, x, y, is_inner, order):
        return self.lam * x + (1 - self.lam) * y, y

    def _get_bezier_curve(self, phi_deg, is_inner, i):
        if is_inner:
            # Inner: Use parent's smooth Bezier logic
            return super()._get_bezier_curve(phi_deg, is_inner, i)

        # Outer: 3-segment broken line logic
        phi_rad = np.radians(phi_deg)
        p_start = np.array([self.R * np.cos(phi_rad), self.R * np.sin(phi_rad)])
        p_end = np.array([p_start[1], p_start[0]])
        # y=x symmetry

        # Compute vertices
        r, L = self.r_vals[i], self.L_vals[i]
        v1 = np.array([r + L / 2, r - L / 2]) / np.sqrt(2.0)
        v2 = np.array([r - L / 2, r + L / 2]) / np.sqrt(2.0)

        # Discretize into dense points for compatibility
        z = np.linspace(0, 1, self.n_curve // 3)[:, None]

        seg1 = (1 - z) * p_start + z * v1
        seg2 = (1 - z) * v1 + z * v2
        seg3 = (1 - z) * v2 + z * p_end

        # Stack segments and drop duplicate joint points
        return np.vstack((seg1, seg2[1:], seg3[1:]))


class RadialSplineGenerator(BarrierGenerator):
    """Unified high-D rotor-barrier parameterisation (Family A).

    `n_barriers` nested flux barriers, each bounded by two clamped cubic
    B-spline radial profiles `r(s)` in polar coordinates over the barrier's
    own angular span ``[theta_lo, theta_hi]`` (degrees). Non-symmetric about
    the d-axis. A **repair / projection** decoder guarantees manufacturable,
    nested geometry: shaft iron, inter-barrier iron, surface bridge and a
    minimum air width are enforced by construction, so every parameter vector
    decodes to a valid design (the property that makes latent-space BO
    efficient). The central d-axis rib is added downstream by the existing
    ``split_barriers`` (offset = w_rod / sqrt(2)).

    Per-barrier parameter block (length ``2 + 2K``):
        ``[theta_lo, theta_hi, c_out[0..K-1], c_in[0..K-1]]``
    where ``c_out`` is the surface-side (higher-r) boundary and ``c_in`` the
    shaft-side (lower-r) boundary. Full vector length ``D = n_barriers*(2+2K)``.

    See ``docs/PARAMETERISATION.md`` for the full design rationale and backups.
    """

    def __init__(
        self,
        design: "Design",
        n_barriers: int = 3,
        K: int = 18,
        t_bridge: float = 0.7,
        t_rib: float = 0.5,
        t_shaft: float = 0.5,
        min_air: float = 0.5,
        w_rod: float = 0.5,
        n_eval: int = 120,
        n_grid: int = 361,
        theta_qmargin: float = 2.0,
        **kwargs,
    ) -> None:
        self.N = int(n_barriers)
        self.K = int(K)
        self.spline_k = 3
        self.t_bridge = float(t_bridge)
        self.t_rib = float(t_rib)
        self.t_shaft = float(t_shaft)
        self.min_air = float(min_air)
        self.w_rod = float(w_rod)
        self.n_eval = int(n_eval)
        self.n_grid = int(n_grid)
        # rib strip half-width: split_barriers removes |x - y| <= offset, whose
        # perpendicular (geometric) width is sqrt(2)*offset == w_rod.
        offset = self.w_rod / np.sqrt(2.0)
        # r_stator_end = t_bridge so the inherited feasibility upper bound
        # self.R == r_max - t_bridge matches the surface-bridge cap.
        super().__init__(design, r_stator_end=self.t_bridge, offset=offset, **kwargs)

        self.block = 2 + 2 * self.K
        # radial bounds for control points
        self.r_lo = self.r_min + self.t_shaft
        self.r_hi = self.r_max - self.t_bridge  # == self.R
        # angular bounds: guarantee theta_lo < 45 < theta_hi so the d-axis rib
        # pierces every barrier (as in the existing Hackl designs).
        self.theta_lo_lo = float(theta_qmargin)
        self.theta_lo_hi = 40.0
        self.theta_hi_lo = 50.0
        self.theta_hi_hi = 90.0 - float(theta_qmargin)

        self.s_grid, self.knots, self.basis = self._make_basis(self.n_eval, self.K, self.spline_k)
        self._params: list | None = None

    # ---- B-spline basis -------------------------------------------------
    def _make_basis(self, n_eval: int, K: int, k: int):
        n_interior = K - k - 1
        if n_interior < 0:
            raise ValueError(f"K={K} too small for cubic B-spline (need K >= {k + 1})")
        interior = np.linspace(0.0, 1.0, n_interior + 2)[1:-1] if n_interior > 0 else np.array([])
        t = np.concatenate((np.zeros(k + 1), interior, np.ones(k + 1)))
        s = np.linspace(0.0, 1.0, n_eval)
        return s, t, self._basis_at(s, t, k)

    def _basis_at(self, s, t, k: int) -> np.ndarray:
        s = np.clip(np.asarray(s, dtype=float), t[k], t[-k - 1] - 1e-12)
        return BSpline.design_matrix(s, t, k).toarray()

    # ---- interface ------------------------------------------------------
    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        blk_lo = np.concatenate(
            ([self.theta_lo_lo, self.theta_hi_lo], np.full(self.K, self.r_lo), np.full(self.K, self.r_lo))
        )
        blk_hi = np.concatenate(
            ([self.theta_lo_hi, self.theta_hi_hi], np.full(self.K, self.r_hi), np.full(self.K, self.r_hi))
        )
        return np.tile(blk_lo, self.N), np.tile(blk_hi, self.N)

    def X_to_params(self, X: np.ndarray) -> list:
        X = np.asarray(X, dtype=float)
        params = []
        for b in range(self.N):
            seg = X[b * self.block : (b + 1) * self.block]
            params.append(
                (float(seg[0]), float(seg[1]), seg[2 : 2 + self.K].copy(), seg[2 + self.K : 2 + 2 * self.K].copy())
            )
        return params

    def params_to_X(self, params) -> np.ndarray:
        return np.concatenate([np.concatenate(([tlo, thi], co, ci)) for (tlo, thi, co, ci) in params])

    def set_parameters(self, params) -> None:
        # accept either a flat X vector or a list of per-barrier tuples
        if isinstance(params, np.ndarray) and params.ndim == 1:
            params = self.X_to_params(params)
        self._params = list(params)

    def random_parameters(self):
        rng = np.random.default_rng()
        return self.X_to_params(self.random_X(rng))

    def random_X(self, rng: np.random.Generator) -> np.ndarray:
        """Prior sample biased toward realistic SynRM barriers, with heavy tails
        for diversity (decision P8, policy 2). Each barrier is an arc whose
        q-axis ends sit near the rotor surface and which dips toward the shaft at
        the d-axis to a per-barrier depth (inner barrier deepest). ~25% of draws
        use a wide deviation amplitude so the manifold keeps exploratory tails."""
        edges = np.linspace(self.r_lo, self.r_hi, self.N + 1)  # per-barrier d-axis depth bands
        cp_s = np.linspace(0.0, 1.0, self.K)
        dip_shape = np.sin(np.pi * cp_s)  # 0 at ends, 1 at d-axis
        amp = 2.5 if rng.random() < 0.25 else 0.6  # heavy-tail switch
        blocks = []
        for b in range(self.N):
            lo_b, hi_b = edges[b], edges[b + 1]
            # wide spans biased to the extremes (small theta_lo, large theta_hi)
            theta_lo = self.theta_lo_lo + (self.theta_lo_hi - self.theta_lo_lo) * rng.beta(1.3, 4.0)
            theta_hi = self.theta_hi_hi - (self.theta_hi_hi - self.theta_hi_lo) * rng.beta(1.3, 4.0)
            depth = rng.uniform(lo_b, hi_b)  # min radius at the d-axis
            end_r = self.r_hi - rng.uniform(0.0, 8.0)  # radius at q-axis ends (near surface, with tail)
            end_r = max(end_r, depth + 1.0)
            midline = end_r - (end_r - depth) * dip_shape
            air = rng.uniform(self.min_air, max(self.min_air * 2.0, (hi_b - lo_b) * 0.6))
            c_out = np.clip(midline + air / 2 + self._smooth_dev(rng, cp_s, amp), self.r_lo, self.r_hi)
            c_in = np.clip(midline - air / 2 + self._smooth_dev(rng, cp_s, amp), self.r_lo, self.r_hi)
            blocks.append(np.concatenate(([theta_lo, theta_hi], c_out, c_in)))
        return np.concatenate(blocks)

    def _smooth_dev(self, rng: np.random.Generator, cp_s: np.ndarray, amp: float = 1.5) -> np.ndarray:
        dev = np.zeros_like(cp_s)
        for m in range(1, 4):
            dev += rng.normal(0.0, amp / m) * np.sin(m * np.pi * cp_s + rng.uniform(0.0, np.pi))
        return dev

    # ---- decoder (repair) ----------------------------------------------
    def generate_barriers(self) -> list[np.ndarray]:
        if self._params is None:
            raise RuntimeError("call set_parameters() before generate_barriers()")
        # 1. raw boundaries on each barrier's own theta grid.
        #    Enforce theta-nesting (inner barrier spans widest, outer narrowest)
        #    so barriers are radially stacked at every shared angle and their
        #    blunt end-caps cannot cross.
        raw = []
        prev_lo = prev_hi = None
        for (theta_lo, theta_hi, c_out, c_in) in self._params:
            theta_lo = float(np.clip(theta_lo, self.theta_lo_lo, self.theta_lo_hi))
            theta_hi = float(np.clip(theta_hi, self.theta_hi_lo, self.theta_hi_hi))
            if prev_lo is not None:
                theta_lo = max(theta_lo, prev_lo)
                theta_hi = min(theta_hi, prev_hi)
            theta_hi = max(theta_hi, theta_lo + 1.0)  # keep a finite span
            prev_lo, prev_hi = theta_lo, theta_hi
            theta = theta_lo + self.s_grid * (theta_hi - theta_lo)
            raw.append([theta, self.basis @ np.asarray(c_in, float), self.basis @ np.asarray(c_out, float)])

        # 2. resample each barrier onto a global angle grid (NaN where absent).
        #    theta-nesting => the present set at every column is a prefix {0..m}.
        G = np.linspace(0.0, 90.0, self.n_grid)
        rin_G = np.full((self.N, self.n_grid), np.nan)
        rout_G = np.full((self.N, self.n_grid), np.nan)
        spans = []
        for b, (theta, r_in, r_out) in enumerate(raw):
            mask = (G >= theta[0]) & (G <= theta[-1])
            spans.append((theta[0], theta[-1]))
            rin_G[b, mask] = np.interp(G[mask], theta, r_in)
            rout_G[b, mask] = np.interp(G[mask], theta, r_out)

        # 3. two-sided radial projection per column: floor from the shaft up,
        #    then ceiling from the surface bridge down. Guarantees shaft iron,
        #    inter-barrier ribs, min air, surface bridge and nesting.
        floor0 = self.r_min + self.t_shaft
        cap = self.r_hi
        present = ~np.isnan(rin_G)
        for g in range(self.n_grid):
            bs = np.where(present[:, g])[0]
            floor = floor0
            for b in bs:
                rin_G[b, g] = max(rin_G[b, g], floor)
                rout_G[b, g] = max(rout_G[b, g], rin_G[b, g] + self.min_air)
                floor = rout_G[b, g] + self.t_rib
            ceil = cap
            for b in bs[::-1]:
                rout_G[b, g] = min(rout_G[b, g], ceil)
                rin_G[b, g] = min(rin_G[b, g], rout_G[b, g] - self.min_air)
                ceil = rin_G[b, g] - self.t_rib

        # 4. closed polygons (outer asc + inner desc + close); blunt radial caps
        barriers = []
        for b in range(self.N):
            m = present[b]
            th = np.radians(G[m])
            ro, ri = rout_G[b, m], rin_G[b, m]
            xy_out = np.column_stack((ro * np.cos(th), ro * np.sin(th)))
            xy_in = np.column_stack((ri * np.cos(th), ri * np.sin(th)))
            loop = np.vstack((xy_out, xy_in[::-1], xy_out[:1]))
            barriers.append(loop)
        return barriers

    # ---- encoder (warm-start) ------------------------------------------
    def fit_barriers(self, barriers: list[np.ndarray]) -> np.ndarray:
        """Least-squares fit this parameterisation to existing closed barrier
        polylines (e.g. from a Hackl generator). Returns a bounds-clipped X."""
        polys = sorted((np.asarray(b, float) for b in barriers), key=lambda p: np.linalg.norm(p, axis=1).min())
        blocks = [self._fit_one(p) for p in polys[: self.N]]
        while len(blocks) < self.N:
            blocks.append(blocks[-1].copy())
        X = np.concatenate(blocks)
        lo, hi = self.bounds
        return np.clip(X, lo, hi)

    def _fit_one(self, poly: np.ndarray) -> np.ndarray:
        r = np.linalg.norm(poly, axis=1)
        theta = np.degrees(np.arctan2(poly[:, 1], poly[:, 0]))
        theta_lo, theta_hi = float(theta.min()), float(theta.max())
        nb = self.n_eval
        edges = np.linspace(theta_lo, theta_hi, nb + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        idx = np.clip(np.digitize(theta, edges) - 1, 0, nb - 1)
        r_out = np.full(nb, np.nan)
        r_in = np.full(nb, np.nan)
        for bi in range(nb):
            rr = r[idx == bi]
            if rr.size:
                r_out[bi] = rr.max()
                r_in[bi] = rr.min()
        s_c = (centers - theta_lo) / max(theta_hi - theta_lo, 1e-9)
        valid = ~np.isnan(r_out)
        r_out = np.interp(s_c, s_c[valid], r_out[valid])
        r_in = np.interp(s_c, s_c[valid], r_in[valid])
        B = self._basis_at(s_c, self.knots, self.spline_k)
        c_out, *_ = np.linalg.lstsq(B, r_out, rcond=None)
        c_in, *_ = np.linalg.lstsq(B, r_in, rcond=None)
        return np.concatenate(([theta_lo, theta_hi], c_out, c_in))
