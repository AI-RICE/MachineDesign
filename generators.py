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