import pickle
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from sklearn import base
# from .design import Design
from machine_design.design import Design
# from .geometry import rotate
from machine_design.geometry import rotate
from scipy.interpolate import CubicSpline
from shapely.geometry import LineString


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
        self, design: Design, r_stator_end: float, offset: None | float = None, n_curve: int = 500, n_flat: int = 20
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


class HacklGenerator_OneLambdaTheta(HacklGenerator_OneLambda):
    def __init__(self, design, r_stator_end, **kwargs):
        self.theta_min = -20
        self.theta_max = 20
        super().__init__(design, r_stator_end, **kwargs)

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lb = np.concatenate((self.phis_inner_min, self.phis_outer_min, [self.lam_min, self.theta_min]))
        ub = np.concatenate((self.phis_inner_max, self.phis_outer_max, [self.lam_max, self.theta_max]))
        return lb, ub

    def random_parameters(self):
        pars = super().random_parameters()
        theta = self.theta_min + (self.theta_max - self.theta_min) * np.random.rand(1)[0]
        return *pars, theta

    def set_parameters(self, params) -> None:
        phis_inner, phis_outer, lam, theta = params
        super().set_parameters((phis_inner, phis_outer, lam))
        self.theta = theta

    def X_to_params(self, X: np.ndarray):
        return X[:3], X[3:6], X[6], X[7]

    def _bezier_point(self, x: float, y: float, is_inner: bool, order: int) -> tuple[float, float]:
        x, y = super()._bezier_point(x, y, is_inner, order)
        if order < 2:
            xx, yy = rotate(x, y, self.theta)
            return xx[0], yy[0]
        return x, y


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
    
    
class RandomBarrierGenerator(BarrierGenerator):
    def __init__(self, design, n_barriers, r_stator_end, der1=1.0, der2=1.0, symmetric=True, **kwargs):
        self.der1 = der1
        self.der2 = der2
        self.symmetric = symmetric
        self.n_barriers = n_barriers
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
        n = self.n_barriers

        lb = np.concatenate([
            np.full(n, 0.1),         # w_mins lower bound
            np.full(n, self.r_min),  # y_mins lower bound
            np.full(n, self.r_min),  # y_mids lower bound
            np.full(n, 0.1),         # w_mids lower bound
            np.full(n, 1.0),         # thetas lower bound
            np.full(n, 0.01),        # w_maxs lower bound
        ])
        ub = np.concatenate([
            np.full(n, 5.0),         # w_mins upper bound
            np.full(n, self.r_max),  # y_mins upper bound
            np.full(n, self.r_max),  # y_mids upper bound
            np.full(n, 5.0),         # w_mids upper bound
            np.full(n, 22.0),        # thetas upper bound
            np.full(n, 5.0),         # w_maxs upper bound
        ])
        return lb, ub

    def generate_w_mins_base(self, rotor_r_min, rotor_r_max):
        n_barriers = self.n_barriers
        available_height = rotor_r_max - rotor_r_min
        gap_ratio = 0.5
        total_dap = available_height*gap_ratio
        
        total_height_barrier = available_height - total_dap

        if n_barriers == 1:
            weights = np.array([1.0])
        else:
           weights = np.array([0.99**i for i in range(n_barriers)])
        
        weights = weights / np.sum(weights)

        return total_height_barrier * weights
    
    def random_parameters(self):
        w_mins_base = self.generate_w_mins_base(self.r_min, self.r_max)

        w_mins = w_mins_base * np.random.uniform(0.7, 1.0, self.n_barriers)
        w_mids = w_mins.copy()

        total_height = np.sum(w_mins)
        available_height = self.r_max - self.r_min
        remaining_space = available_height - total_height

        if total_height > available_height:
            w_mins *= available_height / total_height

        gaps = np.full(self.n_barriers + 1, remaining_space / (self.n_barriers + 1))

        y_min = np.zeros(self.n_barriers)
        current_pos = self.r_min + gaps[0]
        for i in range(self.n_barriers):
            y_min[i] = current_pos
            current_pos += w_mins[i] + gaps[i+1]

        offsets = w_mins * 0.1
        y_mids = y_min + offsets

        w_maxs = np.clip(w_mins * 0.7, 0.01, None)

        base = np.linspace(1, 20, self.n_barriers)
        noise = np.random.uniform(-2, 2, self.n_barriers)
        thetas = base + noise
        thetas[0] = 1

        return w_mins, y_min, y_mids, w_mids, thetas, w_maxs

    def set_parameters(self, params) -> None:
        w_mins, y_mins, y_mids, w_mids, thetas, w_maxs = params
        self.w_mins = w_mins
        self.y_mins = y_mins
        self.y_mids = y_mids
        self.w_mids = w_mids
        self.thetas = thetas
        self.w_maxs = w_maxs

    def X_to_params(self, X: np.ndarray):
        n = self.n_barriers
        return X[:n], X[n:2*n], X[2*n:3*n], X[3*n:4*n], X[4*n:5*n], X[5*n:6*n]

    def generate_barriers(self) -> list[np.ndarray]:
        barriers = []
        for args in zip(self.y_mins, self.w_mins, self.y_mids, self.w_mids, self.thetas, self.w_maxs):
            x_all, y_all = self._create_barrier(*args)
            xy_all = np.vstack((x_all, y_all)).T
            barriers.append(xy_all)
        return barriers