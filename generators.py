from abc import ABC, abstractmethod
import numpy as np
from scipy.interpolate import CubicSpline
from geometry import rotate


def signed_distance(P, a, b, c):
    return a * P[:,0] + b * P[:,1] + c

def check_barrier(barrier: np.ndarray) -> None:
    if not np.array_equal(barrier[0], barrier[-1]):
        raise ValueError("Barrier must start and end at the same point")


class BarrierGenerator(ABC):
    def __init__(self, offset: float | None = None) -> None:
        self.offset = offset
    
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

    def split_barrier(self, barrier: np.ndarray, n_points: int = 20) -> list[np.ndarray]:
        assert self.offset is not None
        check_barrier(barrier)

        lam = np.linspace(0, 1, n_points)[:, None]
        barriers = []
        for a, b, c in zip([1, -1], [-1, 1], [-self.offset, -self.offset]):
            f = signed_distance(barrier, a, b, c)
            f_is_positive = f > 0
            # TODO: add a better interpolation to the boundary
            if f_is_positive[0]:
                mask_negative = np.where(~f_is_positive)[0]
                idx_min = np.min(mask_negative)
                idx_max = np.max(mask_negative)
                if len(np.unique(f_is_positive[idx_min:idx_max])) != 1:
                    raise ValueError("The function crosses the center line too many times.")
                barrier_part1 = barrier[:idx_min-1]
                barrier_part2 = (1-lam)*barrier[idx_min-1] + lam*barrier[idx_max+1]
                barrier_part3 = barrier[idx_max+2:]
                barrier_new = np.concatenate((barrier_part1, barrier_part2, barrier_part3))
            else:
                mask_positive = np.where(f_is_positive)[0]
                idx_min = np.min(mask_positive)
                idx_max = np.max(mask_positive)
                if len(np.unique(f_is_positive[idx_min:idx_max])) != 1:
                    raise ValueError("The function crosses the center line too many times.")
                barrier_part1 = barrier[idx_min:idx_max]
                barrier_part2 = (1-lam)*barrier[idx_max] + lam*barrier[idx_min]
                barrier_new = np.concatenate((barrier_part1, barrier_part2))
            check_barrier(barrier_new)
            barriers.append(barrier_new)
        return barriers

    def split_barriers(self, barriers: list[np.ndarray]) -> list[np.ndarray]:
        if self.offset is None:
            return barriers
        else:
            return [x for barrier in barriers for x in self.split_barrier(barrier)]

class FourStupid(BarrierGenerator):
    def __init__(self, design, n=300, c=0.98, der1=1., der2=1., symmetric=True, **kwargs) -> None:
        self.design = design
        self.n = n
        self.c = c
        self.der1 = der1
        self.der2 = der2
        self.w_mins_base = np.array([3, 2.5, 2.5, 2]) - 1.0
        self.symmetric = symmetric
        super().__init__(**kwargs)

    def create_barrier(
            self,
            y_min,
            w_min,
            y_mid,
            w_mid,
            theta,
            w_max,
            ):
        
        rotor_r_max = self.design.rotor_r_max

        theta1 = (theta+45) / 180*np.pi
        x_max1 = self.c*rotor_r_max*np.cos(theta1)
        y_max1 = self.c*rotor_r_max*np.sin(theta1)

        x1 = [0, x_max1/2, x_max1]
        y1 = [y_min, y_mid, y_max1]
        f1 = CubicSpline(x1, y1, bc_type=((1, 0), (1,self.der1)))

        theta2 = theta1 + w_max / (2*np.pi*rotor_r_max) * 2*np.pi
        x_max2 = self.c*rotor_r_max*np.cos(theta2)
        y_max2 = self.c*rotor_r_max*np.sin(theta2)

        s = w_mid / np.sqrt(1 + self.der1**2/4)
        x2 = [0, x1[1]-s*self.der1/2, x_max2]
        y2 = [y_min+w_min, y1[1]+s, y_max2]
        f2 = CubicSpline(x2, y2, bc_type=((1, 0), (1,self.der2)))

        x_interp1 = np.linspace(x1[0], x1[-1], self.n)
        x_interp2 = np.linspace(x2[0], x2[-1], self.n)
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
            x_all, y_all = self.create_barrier(*args)
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
        

class HacklGenerator(BarrierGenerator):
    def __init__(self, design, **kwargs):
        dia_stator_gap = design.mm_to_str("geom_params", "DiaStatorGap")
        airgap = design.mm_to_str("geom_params", "Airgap")
        self.R = (dia_stator_gap / 2.0) - airgap - 0.7
        self.phis_inner_min = np.asarray([4., 17.3, 30.6])
        self.phis_inner_max = np.asarray([10.2, 23.5, 36.8])
        self.phis_outer_min = np.asarray([10.6, 24., 37.3])
        self.phis_outer_max = np.asarray([16.8, 30.2, 34.])
        self.lam_min = 0.25
        self.lam_max = 0.45
        self.n_barriers = len(self.phis_inner_min)
        super().__init__(**kwargs)

    def generate_parameters(self):
        self.lam = self.lam_min + (self.lam_max-self.lam_min)*np.random.rand(1)[0]
        self.phis_inner = self.phis_inner_min + (self.phis_inner_max-self.phis_inner_min)*np.random.rand(self.n_barriers)
        self.phis_outer = self.phis_outer_min + (self.phis_outer_max-self.phis_outer_min)*np.random.rand(self.n_barriers)

    def generate_parameters_representative(self):
        self.lam = 0.25
        self.phis_inner = [4.0, 17.3, 30.6]
        self.phis_outer = [10.6, 24.0, 37.3]

    def generate_barriers(self) -> list[np.ndarray]:
        # Generate internal parameters
        self.generate_parameters()

        # Create the barriers        
        barriers = []
        for phi_inner, phi_outer in zip(self.phis_inner, self.phis_outer):
            # Generate the longer part of the barrier
            pts_outer = self.get_bezier_curve(phi_outer)
            pts_inner = self.get_bezier_curve(phi_inner)
            
            # Generate the arcs at the end
            arc_top = self.get_arc(90 - phi_outer, 90 - phi_inner)
            arc_bottom = self.get_arc(phi_inner, phi_outer)
            
            # Merge them together
            barrier = np.concatenate((pts_outer, arc_top[1:-1], pts_inner[::-1], arc_bottom[1:]))
            barriers.append(barrier)
        return barriers

    def get_arc(self, start_deg, end_deg, num_points=15):
        angles = np.linspace(np.radians(start_deg), np.radians(end_deg), num_points)

        x = self.R * np.cos(angles)
        y = self.R * np.sin(angles)

        return np.column_stack((x, y))

    def get_bezier_curve(self, phi_deg, num_points=300):
        phi_rad = np.radians(phi_deg)
        x_end = self.R * np.cos(phi_rad)
        y_end = self.R * np.sin(phi_rad)
        x_bezier = self.lam * x_end + (1 - self.lam) * y_end

        r0 = np.array([x_end, y_end])
        r1 = np.array([x_bezier, y_end])
        r2 = np.array([y_end, x_bezier])
        r3 = np.array([y_end, x_end])

        # Formula 33: Polynomial expansion to generate continuous points
        z_vals = np.linspace(0, 1, num_points)[:, None]
        return (
            (1 - z_vals)**3 * r0
            + 3 * (1 - z_vals)**2 * z_vals * r1
            + 3 * (1 - z_vals) * z_vals**2 * r2
            + z_vals**3 * r3
        )

    def save_barriers(self, file_name: str):
        np.savez(file_name,
            phis_inner=self.phis_inner,
            phis_outer=self.phis_outer,
            lam=self.lam)
