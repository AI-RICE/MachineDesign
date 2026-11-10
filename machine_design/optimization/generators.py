import pickle
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline
from shapely.geometry import LineString

from ..designs.design import Design
from .geometry import rotate

TEMPLATES: dict[str, dict[str, dict]] = {
    "IPMCore": {
        "defaults": {
            "PoleType": "1",
            "D1": "DiaShaft + 0.6*(DiaRotorGap - DiaShaft)",
            "O1": "2mm",
            "O2": "10mm",
            "B1": "2mm",
            "Rib": "1mm",
            "HRib": "2mm",
            "DminMag": "1mm",
            "ThickMag": "3mm",
            "WidthMag": "15mm",
        },
        "bounds": {
            "B1": (1.0, 4.0),
            "ThickMag": (2.0, 6.0),
            "WidthMag": (8.0, 20.0),
            "Rib": (0.5, 2.0),
        },
    },
    "PMCore": {
        "defaults": {
            "PoleType": "5",
            "ThickMag": "8mm",
            "WidthMag": "45mm",
        },
        "bounds": {
            "ThickMag": (8.0, 8.0),
            "WidthMag": (45.0, 45.0),
        }},
    "UPMCore": {
        "skew": False,
        "defaults": {
            "DiaLimit": "0.5*DiaRotorGap",
            "ThkBridge": "2mm",
            "ThkWeb": "5mm",
            "AngOffOut": "-10deg",
            "ThkDuctOut": "2mm",
            "ThkDuctIn": "5mm",
            "ThkPostOut": "1mm",
            "ThkPostIn": "-5mm",
            "ThkPostMid": "1mm",
            "WidMagOut": "13mm",
            "GapMagOut": "0.2mm",
            "WidMagIn": "30mm",
            "GapMagIn": "0.2mm",
        },
        "bounds": {
            "ThkBridge": (0.5, 2.0),
            "ThkWeb": (2.0, 6.0),
            "ThkDuctOut": (1.0, 4.0),
            "ThkDuctIn": (2.0, 6.0),
            "WidMagOut": (5.0, 15.0),
            "WidMagIn": (10.0, 30.0),
        },
    },
    "VPMCore": {
        "defaults": {
            "PoleType": "0",
            "ThkDuct": "4mm",
            "WidDuct": "30mm",
            "ThkBridge": "2mm",
            "AngPoleV": "140deg",
            "ArcPole": "155deg",
            "ThkPost": "1mm",
            "DminMag": "2mm",
            "WidMag": "30mm",
            "GapMag": "0.2mm",
        },
        "bounds": {
            "ThkDuct": (2.0, 6.0),
            "WidDuct": (15.0, 30.0),
            "ThkBridge": (0.5, 2.0),
            "AngPoleV": (100.0, 160.0, "deg"),
            "WidMag": (10.0, 30.0),
        },
    },
}
TEMPLATE_NAME = list(TEMPLATES)


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
    def __init__(self, design: Design, r_stator_end: float, offset: None | float = None, n_curve: int = 500, n_flat: int = 20) -> None:
        self.r_max = design.geometry.rotor_r_max
        self.r_min = design.geometry.rotor_r_min
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
        return (1 - z_vals) ** 3 * r0 + 3 * (1 - z_vals) ** 2 * z_vals * r1 + 3 * (1 - z_vals) * z_vals**2 * r2 + z_vals**3 * r3


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


class MagnetGenerator(BarrierGenerator):
    def __init__(
        self,
        design,
        r_stator_end,
        der1=1.0,
        der2=1.0,
        symmetric=True,
        min_bridge=0.0,
        shape="rect",
        bend_ratio=0.7,
        thickness_ratio=0.5,
        **kwargs,
    ):
        # fraction of the rotor height (r_max - r_min) occupied by all barriers together
        self.thickness_ratio = thickness_ratio
        self.n_designs = kwargs.get("n_designs", 1)
        self.der1 = der1
        self.der2 = der2
        self.symmetric = symmetric
        # "spline" = smooth curved barrier, "rect" = flat magnet pocket + straight leg
        self.shape = shape
        # x position of the bend as a fraction of y_min (< 1 keeps the bend inside the pole)
        self.bend_ratio = bend_ratio
        self.n_barriers = kwargs.get("n_barriers", 1)
        self.sector_angle = np.pi / design.geometry.PolePairs
        self.min_bridge = min_bridge
        super().__init__(design, r_stator_end, **kwargs)

    def feasible_barriers(self, barriers: list[np.ndarray], tol: float = 1e-8) -> bool:
        for barrier in barriers:
            dist_start = barrier[:, 1]
            dist_end = barrier[:, 0] * np.sin(self.sector_angle) - barrier[:, 1] * np.cos(self.sector_angle)
            if any(dist_start < self.min_bridge - tol) or any(dist_end < self.min_bridge - tol):
                return False
        return super().feasible_barriers(barriers, tol)

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

    def _polyline(self, points):
        # Discretize a polyline into dense points, dropping duplicate joints
        n_seg = max(self.n_curve // (len(points) - 1), 2)
        z = np.linspace(0, 1, n_seg)[:, None]
        segs = [(1 - z) * p + z * q for p, q in zip(points[:-1], points[1:])]
        return np.vstack([segs[0]] + [s[1:] for s in segs[1:]])

    def _create_rect_barrier(
        self,
        y_min,
        w_min,
        y_mid,
        w_mid,
        theta,
        w_max,
    ):
        # Lower edge: flat bottom (magnet pocket) up to the bend, then straight leg to the rim
        theta1 = (theta + 45) / 180 * np.pi
        p_rim1 = self.R * np.array([np.cos(theta1), np.sin(theta1)])
        b1 = np.array([self.bend_ratio * y_min, y_min])

        # Upper edge: flat part offset by w_min, leg parallel to the lower leg offset by w_mid
        d = (p_rim1 - b1) / np.linalg.norm(p_rim1 - b1)
        n = np.array([-d[1], d[0]])
        q = b1 + w_mid * n
        y_top = y_min + w_min
        b2 = q + (y_top - q[1]) / d[1] * d
        b2[0] = max(b2[0], 0.0)

        # Intersection of the upper leg line with the circle of radius R
        bd = np.dot(b2, d)
        s = -bd + np.sqrt(bd**2 - (np.dot(b2, b2) - self.R**2))
        p_rim2 = b2 + s * d

        lower = self._polyline([np.array([0.0, y_min]), b1, p_rim1])
        upper = self._polyline([np.array([0.0, y_top]), b2, p_rim2])

        x_all = np.concatenate((lower[:, 0], upper[::-1, 0]))
        y_all = np.concatenate((lower[:, 1], upper[::-1, 1]))
        if self.symmetric:
            x_all = np.concatenate((x_all, -x_all[::-1][1:]))
            y_all = np.concatenate((y_all, y_all[::-1][1:]))
        x_all, y_all = rotate(x_all, y_all, -45)
        return x_all, y_all

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        n = self.n_barriers

        lb = np.concatenate(
            [
                np.full(n, 0.7),  # rand1
                np.array([12]),  # rand2
                np.full(n - 1, 1),  # spacings
            ]
        )

        ub = np.concatenate([np.full(n, 1.0), np.array([20]), np.full(n - 1, 5)])

        return lb, ub

    def generate_w_mins_base(self, rotor_r_min, rotor_r_max):
        n_barriers = self.n_barriers
        available_height = rotor_r_max - rotor_r_min
        total_height_barrier = available_height * self.thickness_ratio

        if n_barriers == 1:
            weights = np.array([1.0])
        else:
            weights = np.array([0.99**i for i in range(n_barriers)])

        weights = weights / np.sum(weights)

        return total_height_barrier * weights

    def random_parameters(self):

        rand1 = np.random.uniform(0.7, 1.0, self.n_barriers)
        rand2 = np.random.uniform(12, 20)
        spacings = np.random.dirichlet(np.ones(self.n_barriers - 1))

        return rand1, rand2, spacings

    def set_parameters(self, params) -> None:
        rand1, rand2, spacings = params

        w_mins_base = self.generate_w_mins_base(self.r_min, self.r_max)

        w_mins = w_mins_base * rand1
        w_mids = w_mins.copy()

        total_height = np.sum(w_mins)
        total_spacing = np.sum(spacings)

        available_height = self.r_max - rand2

        eps = 1e-8
        denom = total_height + total_spacing
        scale = (available_height) / max(denom, eps)

        if scale < 1.0:
            w_mins *= scale
            spacings *= scale

        y_mins = np.zeros(self.n_barriers)
        y_mins[0] = rand2
        current_pos = y_mins[0]

        for i in range(1, self.n_barriers):
            current_pos += w_mins[i - 1] + spacings[i - 1]
            y_mins[i] = current_pos

        offsets = w_mins * 0.1
        y_mids = y_mins + offsets

        w_maxs = np.clip(w_mins * 0.7, 0.01, None)

        w_maxs = np.clip(w_mins * 0.7, 0.01, None)
        thetas = np.linspace(1, 20, self.n_barriers)
        thetas[0] = 1

        self.y_mins = y_mins
        self.w_mins = w_mins
        self.y_mids = y_mids
        self.w_mids = w_mids
        self.thetas = thetas
        self.w_maxs = w_maxs

        return y_mins, w_mins, y_mids, w_mids, thetas, w_maxs

    def X_to_params(self, X: np.ndarray, barrier=None):
        n = self.n_barriers if barrier is None else barrier

        if len(X) != 2 * n:
            raise ValueError(f"X length {len(X)} does not match expected 2*n={2 * n}")

        rand1 = X[:n]
        rand2 = X[n]
        spacings = X[n + 1 : n + 1 + (n - 1)]

        return rand1, rand2, spacings

    def generate_barriers(self) -> list[np.ndarray]:
        barriers = []
        create = self._create_rect_barrier if self.shape == "rect" else self._create_barrier
        for args in zip(self.y_mins, self.w_mins, self.y_mids, self.w_mids, self.thetas, self.w_maxs):
            x_all, y_all = create(*args)
            xy_all = np.vstack((x_all, y_all)).T
            barriers.append(xy_all)
        return barriers

    def generate_magnets(self, barriers: list[np.ndarray], margin: float = 0.0) -> list[np.ndarray]:
        magnets = []

        for barrier in barriers:
            x, y = barrier[:, 0], barrier[:, 1]
            mag_pts = np.column_stack((x, y))

            if margin <= 0:
                magnets.append([mag_pts])
                continue

            angles = np.degrees(np.arctan2(mag_pts[:, 1], mag_pts[:, 0]))

            center_angle = (angles.max() + angles.min()) / 2

            full_half_span = (angles.max() - angles.min()) / 2
            center_half_span = full_half_span * (1 - 2 * margin)

            center_mask = np.abs(angles - center_angle) < center_half_span
            magnets.append([mag_pts[center_mask]])

        return magnets

class MagnetGenerator_template(MagnetGenerator):
    def __init__(self, design, r_stator_end, template: str = "IPMCore", **kwargs):
        if template not in TEMPLATES:
            raise KeyError(f"Unknown template {template!r}, choose from {TEMPLATE_NAME}")
        if not TEMPLATES[template]["defaults"]:
            raise NotImplementedError(f"Template {template!r} has no parameters defined in TEMPLATES yet")
        super().__init__(design, r_stator_end, **kwargs)
        self.design = design
        self.template = template
        self.default_template = TEMPLATES[template]["defaults"]
        self.opt_bounds = TEMPLATES[template]["bounds"]
        self.opt_names = list(self.opt_bounds)

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lb = np.array([b[0] for b in self.opt_bounds.values()])
        ub = np.array([b[1] for b in self.opt_bounds.values()])
        return lb, ub

    def random_parameters(self):
        lb, ub = self.bounds
        return np.random.uniform(lb, ub)

    def X_to_params(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        if len(X) != len(self.opt_names):
            raise ValueError(f"X length {len(X)} does not match {len(self.opt_names)} parameters {self.opt_names}")
        return X

    def set_parameters(self, params) -> None:
        units = [b[2] if len(b) > 2 else "mm" for b in self.opt_bounds.values()]
        self.set_template(**{name: f"{value}{unit}" for name, value, unit in zip(self.opt_names, params, units)})

    @staticmethod
    def _var_name(udp_name: str) -> str:
        return udp_name.replace("/", "_")

    def generate_barriers(self) -> list[np.ndarray]:
        return []

    def _udp_params(self, info_core: str) -> list[list[str]]:
        has_skew = TEMPLATES[self.template].get("skew", True)
        return [
            ["DiaGap", "DiaRotorGap"],
            ["DiaYoke", "DiaShaft"],
            ["Length", "0mm"],
            *([["Skew", "0deg"]] if has_skew else []),
            ["Poles", "Poles"],
            *[[name, self._var_name(name)] for name in self.default_template],
            ["LenRegion", "0mm"],
            ["InfoCore", info_core], # 0 - core, 1 magnet
        ]

    def take_template(self, template_params: dict[str, str] | None = None):
        m2d = self.design.m2d
        geometry = self.design.geometry
        modeler = m2d.modeler

        params = {**self.default_template, **(template_params or {})}
        m2d["DiaRotorGap"] = "DiaStatorGap - 2*Airgap"
        for k, v in params.items():
            m2d[self._var_name(k)] = v

        rotor_id = modeler.create_udp(
            dll=f"RMxprt/{self.template}.dll",
            parameters=self._udp_params("0"),
            library="syslib",
            name="Rotor",
        )
        magnet_id = modeler.create_udp(
            dll=f"RMxprt/{self.template}.dll",
            parameters=self._udp_params("1"),
            library="syslib",
            name="Magnet",
        )

        modeler.set_working_coordinate_system("Section")
        modeler.split(assignment=[rotor_id, magnet_id], plane="ZX", sides="NegativeOnly")
        modeler.set_working_coordinate_system("Global")
        modeler.split(assignment=[rotor_id, magnet_id], plane="ZX", sides="PositiveOnly")

        rotor_id.material_name = geometry.Fe
        magnet_names = modeler.separate_bodies(assignment=magnet_id.name)
        magnet_ids = [modeler[name] for name in magnet_names]
        for mag in magnet_ids:
            if getattr(geometry, "magnet", None):
                mag.material_name = geometry.magnet
            mag.solve_inside = True
            mag.color = (255, 0, 0)

        geometry.rotor_id = rotor_id
        geometry.magnet_ids = magnet_ids
        geometry.magnet_cs_ids = []
        return rotor_id, magnet_ids

    def set_template(self, **shape: str) -> None:
        allowed = {self._var_name(name) for name in self.default_template}
        unknown = {k for k in shape if self._var_name(k) not in allowed}
        if unknown:
            raise KeyError(f"Unknown template parameters: {sorted(unknown)}")
        for k, v in shape.items():
            self.design.m2d[self._var_name(k)] = v