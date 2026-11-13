"""Diagnostic: with Nper=2, plot the raw Flux_d/Flux_q/V_d/V_q waveforms to
visually check whether period 1 differs from period 2.

Run explicitly with:

    python tests/periodic_plot.py
"""

import tempfile

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from machine_design.config import load_config
from machine_design.designs.design import Design as LiveDesign
from machine_design.optimization.generators import HacklGenerator_OneLambda
from motors.synrm_3f_36s import Computation, Geometry

AEDT_VERSION = load_config()["aedt_version"]
R_STATOR_END = 0.7
OFFSET = 0.35
SEED = 42
NUM_CORES = 1
f_ref = 50
Nper = "2"
current_setpoint = (1.5, 1.5)
OUT_PATH = "tests/periodic_plot.png"


def _generate_one_lambda_barriers(seed):
    geometry = Geometry()
    dummy = LiveDesign(m2d=None, geometry=geometry, computation=Computation(geometry))
    generator = HacklGenerator_OneLambda(dummy, R_STATOR_END, offset=OFFSET)

    np.random.seed(seed)
    while True:
        params = generator.random_parameters()
        generator.set_parameters(params)
        barriers = generator.generate_barriers()
        barriers = generator.split_barriers(barriers)
        if generator.feasible_barriers(barriers):
            return barriers


def main():
    barriers = _generate_one_lambda_barriers(SEED)
    geometry = Geometry()
    computation = Computation(geometry, f=f_ref, Nper=Nper)
    with tempfile.TemporaryDirectory() as tmp_dir:
        design = LiveDesign.create(
            "period_plot",
            "Design01",
            f"{tmp_dir}/period_plot.aedt",
            geometry,
            computation,
            version=AEDT_VERSION,
            non_graphical=True,
            new_desktop=True,
            close_on_exit=False,
        )
        design.add_rotor()
        for barrier in barriers:
            design.add_rotor_barrier(barrier)
        out = design.compute(*current_setpoint, NUM_CORES=NUM_CORES)
        design.delete_rotor()
        design.m2d.close_desktop()

    point_per = int(computation.oper_params["PointPer"])
    theta = out["theta_el"]

    fields = ("Flux_d", "Flux_q", "V_d", "V_q")
    fig, axes = plt.subplots(len(fields), 2, figsize=(11, 9))

    for row, name in enumerate(fields):
        y = out[name]
        # left column: full timeline across both periods
        axes[row, 0].plot(theta / np.pi, y)
        axes[row, 0].set_ylabel(name)
        if row == 0:
            axes[row, 0].set_title("full timeline")

        # right column: period 1 vs period 2 overlaid on the same angle axis
        period1 = y[:point_per]
        period2 = y[point_per - 1 : 2 * point_per - 1]
        theta_rel = (theta[:point_per] - theta[0]) / np.pi
        axes[row, 1].plot(theta_rel, period1, label="period 1")
        axes[row, 1].plot(theta_rel, period2, "--", label="period 2")
        if row == 0:
            axes[row, 1].set_title("period 1 vs period 2 (overlaid)")
            axes[row, 1].legend()

    for ax in list(axes[:, 0]) + list(axes[:, 1]):
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda val, pos: f"{val:g}$\\pi$"))

    axes[-1, 0].set_xlabel("theta_el/pi [rad]")
    axes[-1, 1].set_xlabel("theta_el (within period)/pi [rad]")
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"Saved plot to {OUT_PATH}")


if __name__ == "__main__":
    main()
