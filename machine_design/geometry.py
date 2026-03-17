import matplotlib.pyplot as plt
import numpy as np


def rotation_matrix(alpha, rad=False):
    if not rad:
        alpha /= 360 / (2 * np.pi)
    return np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])


def rotate(x, y, alpha, **kwargs):
    T = rotation_matrix(alpha, **kwargs)
    xy_rot = T @ np.vstack((x, y))
    return xy_rot[0], xy_rot[1]


def plot_barriers(barriers, design, title=None, file_name=None):
    plt.figure()
    alphas = np.linspace(0 * np.pi, 2 * np.pi / 4, 100)
    for r in [design.rotor_r_min, design.rotor_r_max]:
        plt.plot(r * np.cos(alphas), r * np.sin(alphas))
    for barrier in barriers:
        plt.plot(barrier[:, 0], barrier[:, 1])
    plt.axis("equal")
    if title is not None:
        plt.title(title)
    if file_name is not None:
        plt.savefig(file_name)
        plt.close()


def analyze_results(Tor: np.ndarray) -> tuple[float, float, float]:
    TorAvg = np.mean(Tor[:-1])
    TorRmsAC = np.sqrt(np.mean(np.square(Tor[:-1] - TorAvg)))
    TorRippleRms = TorRmsAC / TorAvg * 100
    return TorAvg, TorRmsAC, TorRippleRms
