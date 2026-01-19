import numpy as np
from scipy.interpolate import CubicSpline

def rotation_matrix(alpha, rad=False):
    if not rad:
        alpha /= 360/(2*np.pi)
    return np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])

def rotate(x, y, alpha, **kwargs):
    T = rotation_matrix(alpha, **kwargs)
    xy_rot = T @ np.vstack((x, y))
    return xy_rot[0], xy_rot[1]

def create_barrier(
        design,
        y_min,
        w_min,
        y_mid,
        w_mid,
        theta,
        w_max,
        n=100,
        c=0.98,
        symmetric=True
        ):
    
    theta1 = (theta+45) / 180*np.pi
    x_max1 = c*design.rotor_r_max*np.cos(theta1)
    y_max1 = c*design.rotor_r_max*np.sin(theta1)
    der1 = 1

    x1 = [0, x_max1/2, x_max1]
    y1 = [y_min, y_mid, y_max1]
    f1 = CubicSpline(x1, y1, bc_type=((1, 0), (1,der1)))

    theta2 = theta1 + w_max / (2*np.pi*design.rotor_r_max) * 2*np.pi
    x_max2 = c*design.rotor_r_max*np.cos(theta2)
    y_max2 = c*design.rotor_r_max*np.sin(theta2)
    der2 = 1

    s = w_mid / np.sqrt(1 + der1**2/4)
    x2 = [0, x1[1]-s*der1/2, x_max2]
    y2 = [y_min+w_min, y1[1]+s, y_max2]
    f2 = CubicSpline(x2, y2, bc_type=((1, 0), (1,der2)))

    x_interp1 = np.linspace(x1[0], x1[-1], n)
    x_interp2 = np.linspace(x2[0], x2[-1], n)
    x_all = np.concatenate((x_interp1, x_interp2[::-1]))
    y_all = np.concatenate((f1(x_interp1), f2(x_interp2)[::-1]))
    if symmetric:
        x_all = np.concatenate((x_all, -x_all[::-1][1:]))
        y_all = np.concatenate((y_all, y_all[::-1][1:]))
    x_all, y_all = rotate(x_all, y_all, -45)
    return x_all, y_all