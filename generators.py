from abc import ABC, abstractmethod
import numpy as np
from scipy.interpolate import CubicSpline
from geometry import rotate


class BarrierGenerator(ABC):
    @abstractmethod
    def random_barriers(self) -> list[np.ndarray]:
        pass

    @abstractmethod
    def save_barriers(self, file_name: str) -> None:
        pass


class FourStupid(BarrierGenerator):
    def __init__(self, design, n=100, c=0.98, der1=1., der2=1., symmetric=True) -> None:
        self.design = design
        self.n = n
        self.c = c
        self.der1 = der1
        self.der2 = der2
        self.w_mins_base = np.array([3, 2.5, 2.5, 2]) - 1.0
        self.symmetric = symmetric

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

    def generate_parameters(self):
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

    def random_barriers(self) -> list[np.ndarray]:
        # Generate internal parameters
        self.generate_parameters()

        # Create the barriers
        barriers = []
        for args in zip(self.y_mins, self.w_mins, self.y_mids, self.w_mids, self.thetas, self.w_maxs):
            x_all, y_all = self.create_barrier(*args)
            xy_all = np.vstack((x_all, y_all)).T
            barriers.append(xy_all)
        return barriers

    def save_barriers(self, file_name: str):
        np.savez(file_name,
             y_mins=self.y_mins,
             w_mins=self.w_mins,
             y_mids=self.y_mids,
             w_mids=self.w_mids,
             thetas=self.thetas,
             w_maxs=self.w_maxs)
        

class HacklGenerator(BarrierGenerator):
    def __init__(self, design):
        dia_stator_gap = design.mm_to_str("geom_params", "DiaStatorGap")
        airgap = design.mm_to_str("geom_params", "Airgap")
        self.R = (dia_stator_gap / 2.0) - airgap - 0.7
        self.phis_inner_min = np.asarray([2., 16., 28.])
        self.phis_inner_max = np.asarray([6., 20., 32.])
        self.phis_outer_min = np.asarray([8., 22., 34.])
        self.phis_outer_max = np.asarray([12., 26., 38.])
        self.n_barriers = len(self.phis_inner_min)

    def generate_parameters(self):
        self.lam = 0.25
        self.phis_inner = self.phis_inner_min + (self.phis_inner_max-self.phis_inner_min)*np.random.rand(self.n_barriers)
        self.phis_outer = self.phis_outer_min + (self.phis_outer_max-self.phis_outer_min)*np.random.rand(self.n_barriers)

    def generate_parameters_representative(self):
        self.lam = 0.25
        self.phis_inner = [4.0, 17.3, 30.6]
        self.phis_outer = [10.6, 24.0, 37.3]

    def get_arc(self, start_deg, end_deg, num_points=15):
        angles = np.linspace(np.radians(start_deg), np.radians(end_deg), num_points)
        return [np.array([self.R * np.cos(a), self.R * np.sin(a)]) for a in angles]

    def get_bezier_curve(self, phi_deg, num_points=30):
        phi_rad = np.radians(phi_deg)
        x0 = self.R * np.cos(phi_rad)
        y0 = self.R * np.sin(phi_rad)
        r0 = np.array([x0, y0])
        
        # Formula: r_{k,1} = [[lam, 1-lam], [0, 1]] * r_{k,0}
        x1 = self.lam * x0 + (1 - self.lam) * y0
        y1 = y0
        r1 = np.array([x1, y1])
        
        # Formula: r_{k,3} and r_{k,2} (mirrored across the y=x diagonal)
        r3 = np.array([y0, x0])
        r2 = np.array([y1, x1])
        
        # Formula 33: Polynomial expansion to generate continuous points
        z_vals = np.linspace(0, 1, num_points)
        curve_pts = []
        for z in z_vals:
            pt = (1-z)**3 * r0 + 3*(1-z)**2 * z * r1 + 3*(1-z) * z**2 * r2 + z**3 * r3
            curve_pts.append(pt)
            
        return curve_pts

    def random_barriers(self) -> list[np.ndarray]:
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
            closed_loop_pts = pts_outer + arc_top[1:] + pts_inner[::-1][1:] + arc_bottom[1:-1]
            barriers.append(np.asarray(closed_loop_pts))
        return barriers

    def save_barriers(self, file_name: str):
        np.savez(file_name,
            phis_inner=self.phis_inner,
            phis_outer=self.phis_outer,
            lam=self.lam)
