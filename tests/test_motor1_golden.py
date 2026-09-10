import os

import numpy as np
import pytest

from machine_design.config import load_config
from machine_design.designs import load_design
from machine_design.optimization.generators import HacklGenerator_OneLambda
from motors.synrm_3f_36s import Computation, Geometry

pytestmark = pytest.mark.ansys

aedt_version = load_config()["aedt_version"]
r_stator_end = 0.7
offset = 0.35
seed = 0
num_cores = 4

golden_tor = np.loadtxt(os.path.join(os.path.dirname(__file__), "golden", "motor1_baseline", "tor.csv"), delimiter=",", skiprows=1)


def test_torque_matches_golden_baseline(tmp_path):
    geometry = Geometry()
    computation = Computation(geometry)
    design = load_design(
        str(tmp_path / "motor1_golden_check.aedt"),
        "Motor1GoldenCheck",
        "Design01",
        aedt_version,
        geometry,
        computation,
        non_graphical=True,
        new_desktop=True,
        close_on_exit=False,
    )
    try:
        np.random.seed(seed)
        generator = HacklGenerator_OneLambda(design, r_stator_end, offset=offset)
        while True:
            params = generator.random_parameters()
            generator.set_parameters(params)
            barriers = generator.generate_barriers()
            barriers = generator.split_barriers(barriers)
            if generator.feasible_barriers(barriers):
                break

        design.add_rotor()
        for barrier in barriers:
            design.add_rotor_barrier(barrier)

        result = design.compute(NUM_CORES=num_cores)
        design.delete_rotor()

        assert result is not None
        Tor = np.asarray(result, dtype=float)
        np.testing.assert_allclose(Tor, golden_tor, rtol=0.001, err_msg="torque waveform drifted from the golden baseline")
    finally:
        design.close_project()
