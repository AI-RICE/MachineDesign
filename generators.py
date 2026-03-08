from abc import ABC, abstractmethod
import numpy as np
from scipy.interpolate import CubicSpline
from design import Design
from geometry import rotate


def get_arc(R: float, start_deg: float, end_deg: float, n_points: int) -> np.ndarray:
    angles = np.linspace(np.radians(start_deg), np.radians(end_deg), n_points)

    x = R * np.cos(angles)
    y = R * np.sin(angles)

    return np.column_stack((x, y))

def signed_distance(P: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a * P[:,0] + b * P[:,1] + c

def check_barrier(barrier: np.ndarray) -> None:
    if not np.array_equal(barrier[0], barrier[-1]):
        raise ValueError("Barrier must start and end at the same point")

def compute_r_max(design: Design, r_stator_end: float) -> float:
    dia_stator_gap = design.mm_to_str("geom_params", "DiaStatorGap")
    airgap = design.mm_to_str("geom_params", "Airgap")
    return (dia_stator_gap / 2.0) - airgap - r_stator_end


class BarrierGenerator(ABC):
    def __init__(self, offset: float, n_curve: int = 500, n_flat: int = 20) -> None:
        self.offset = offset
        self.n_curve = n_curve
        self.n_flat = n_flat
    
    @abstractmethod
    def generate_barriers(self) -> list[np.ndarray]:
        pass

    @abstractmethod
    def generate_parameters(self) -> None:
        pass

    @abstractmethod
    def save_barriers(self, file_name: str) -> None:
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def random_barriers(self) -> list[np.ndarray]:
        self.generate_parameters()
        return self.generate_barriers()

    def split_barrier(self, barrier: np.ndarray) -> list[np.ndarray]:
        assert self.offset is not None
        check_barrier(barrier)

        lam = np.linspace(0, 1, self.n_flat)[:, None]
        barriers = []
        for a, b, c in zip([1., -1.], [-1., 1.], [-self.offset, -self.offset]):
            f = signed_distance(barrier, a, b, c)
            f_is_positive = f > 0
            mask_positive = np.where(f_is_positive)[0]

            barrier_new = [barrier[mask_positive[0]]]
            for i in range(len(mask_positive) - 1):
                i1 = mask_positive[i]
                i2 = mask_positive[i+1]

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
    def __init__(self, design, r_stator_end, der1=1., der2=1., symmetric=True, **kwargs) -> None:
        self.R = compute_r_max(design, r_stator_end)
        self.design = design
        self.der1 = der1
        self.der2 = der2
        self.w_mins_base = np.array([3, 2.5, 2.5, 2]) - 1.0
        self.symmetric = symmetric
        super().__init__(**kwargs)

    def _create_barrier(
            self,
            y_min,
            w_min,
            y_mid,
            w_mid,
            theta,
            w_max,
            ):
        
        theta1 = (theta+45) / 180*np.pi
        x_max1 = self.R*np.cos(theta1)
        y_max1 = self.R*np.sin(theta1)

        x1 = [0, x_max1/2, x_max1]
        y1 = [y_min, y_mid, y_max1]
        f1 = CubicSpline(x1, y1, bc_type=((1, 0), (1,self.der1)))

        theta2 = theta1 + w_max / self.design.rotor_r_max
        x_max2 = self.R*np.cos(theta2)
        y_max2 = self.R*np.sin(theta2)

        s = w_mid / np.sqrt(1 + self.der1**2/4)
        x2 = [0, x1[1]-s*self.der1/2, x_max2]
        y2 = [y_min+w_min, y1[1]+s, y_max2]
        f2 = CubicSpline(x2, y2, bc_type=((1, 0), (1,self.der2)))

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

    def generate_parameters(self) -> None:
        # Parameterize the design
        self.w_mins = self.w_mins_base + 0.5*np.random.random(4)
        y_min0 = 13+4*np.random.random()
        y_min1 = y_min0+self.w_mins[0]+2+3*np.random.random()
        y_min2 = y_min1+self.w_mins[1]+2+3*np.random.random()
        y_min3 = y_min2+self.w_mins[2]+2+2*np.random.random()
        self.w_mids = self.w_mins
        self.y_mins = np.array([y_min0, y_min1, y_min2, y_min3])
        self.y_mids = self.y_mins + np.array([2, 1.5, 1, 0.5])
        self.thetas = [1, 8, 15, 21]
        self.w_maxs = self.w_mins - np.array([0.5, 0.5, 0.5, 0])

    def generate_barriers(self) -> list[np.ndarray]:
        barriers = []
        for args in zip(self.y_mins, self.w_mins, self.y_mids, self.w_mids, self.thetas, self.w_maxs):
            x_all, y_all = self._create_barrier(*args)
            xy_all = np.vstack((x_all, y_all)).T
            barriers.append(xy_all)
        return barriers

    def save_barriers(self, file_name: str) -> None:
        np.savez(file_name,
             y_mins=self.y_mins,
             w_mins=self.w_mins,
             y_mids=self.y_mids,
             w_mids=self.w_mids,
             thetas=self.thetas,
             w_maxs=self.w_maxs)
        

class AbstractHacklGenerator(BarrierGenerator):
    def __init__(self, design, r_stator_end, **kwargs):
        self.R = compute_r_max(design, r_stator_end)
        self.phis_inner_min = np.asarray([4., 17.3, 30.6])
        self.phis_inner_max = np.asarray([10.2, 23.5, 36.8])
        self.phis_outer_min = np.asarray([10.6, 24., 37.3])
        self.phis_outer_max = np.asarray([16.8, 30.2, 44.])
        self.n_barriers = len(self.phis_inner_min)
        super().__init__(**kwargs)

    @abstractmethod
    def _bezier_point(self, x: float, y: float, is_inner: bool, order: int) -> tuple[float, float]:
        pass

    def generate_parameters(self) -> None:
        self.phis_inner = self.phis_inner_min + (self.phis_inner_max-self.phis_inner_min)*np.random.rand(self.n_barriers)
        self.phis_outer = self.phis_outer_min + (self.phis_outer_max-self.phis_outer_min)*np.random.rand(self.n_barriers)

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
            (1 - z_vals)**3 * r0
            + 3 * (1 - z_vals)**2 * z_vals * r1
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
        lb = np.concatenate((self.phis_inner_min, [self.lam_min]))
        ub = np.concatenate((self.phis_inner_max, [self.lam_max]))
        return lb, ub
    
    def generate_parameters(self):
        self.lam = self.lam_min + (self.lam_max-self.lam_min)*np.random.rand(1)[0]
        super().generate_parameters()

    def _bezier_point(self, x: float, y: float, is_inner: bool, order: int) -> tuple[float, float]:
        return self.lam * x + (1 - self.lam) * y, y

    def save_barriers(self, file_name: str):
        np.savez(file_name,
            phis_inner=self.phis_inner,
            phis_outer=self.phis_outer,
            lam=self.lam)



class HacklGenerator_OneLambdaTheta(HacklGenerator_OneLambda):
    def __init__(self, design, r_stator_end, **kwargs):
        self.theta_min = -20
        self.theta_max = 20
        super().__init__(design, r_stator_end, **kwargs)

    def generate_parameters(self):
        self.theta = self.theta_min + (self.theta_max-self.theta_min)*np.random.rand(1)[0]
        super().generate_parameters()

    def _bezier_point(self, x: float, y: float, is_inner: bool, order: int) -> tuple[float, float]:
        x, y = super()._bezier_point(x, y, is_inner, order)
        if order < 2:
            xx, yy = rotate(x, y, self.theta)
            return xx[0], yy[0]
        return x, y

    def save_barriers(self, file_name: str):
        np.savez(file_name,
            phis_inner=self.phis_inner,
            phis_outer=self.phis_outer,
            lam=self.lam,
            theta=self.theta)


class HacklGenerator_TwoLambdas(AbstractHacklGenerator):
    def __init__(self, design, r_stator_end, **kwargs):
        self.lam_inner_min = 0.25
        self.lam_inner_max = 0.45
        self.lam_outer_min = 0.25
        self.lam_outer_max = 0.45
        super().__init__(design, r_stator_end, **kwargs)

    def generate_parameters(self):
        self.lam_inner = self.lam_inner_min + (self.lam_inner_max-self.lam_inner_min)*np.random.rand(1)[0]
        self.lam_outer = self.lam_outer_min + (self.lam_outer_max-self.lam_outer_min)*np.random.rand(1)[0]
        super().generate_parameters()

    def _bezier_point(self, x: float, y: float, is_inner: bool, order: int) -> tuple[float, float]:
        lam = self.lam_inner if is_inner else self.lam_outer
        return lam * x + (1 - lam) * y, y

    def save_barriers(self, file_name: str):
        np.savez(file_name,
            phis_inner=self.phis_inner,
            phis_outer=self.phis_outer,
            lam_inner=self.lam_inner,
            lam_outer=self.lam_outer)