"""Goal: test both synrm_3f_60s.py and synrm_5f_60s.py can accept the same rotor and solve.

Run explicitly with:

    pytest -m ansys tests/test_ansys_synrm_60s_solve.py

Requires a running Ansys Electronics Desktop session and a free license seat.
"""

import numpy as np
import pytest

from machine_design.config import load_config
from machine_design.designs.design import Design as LiveDesign
from machine_design.optimization.generators import HacklGenerator_OneLambda
from motors.synrm_3f_60s import Computation as Computation3f
from motors.synrm_3f_60s import Geometry as Geometry3f
from motors.synrm_5f_60s import Computation as Computation5f
from motors.synrm_5f_60s import Geometry as Geometry5f

pytestmark = pytest.mark.ansys

AEDT_VERSION = load_config()["aedt_version"]
R_STATOR_END = 0.7
OFFSET = 0.35
SEEDS = (42, 43)
NUM_CORES = 1
Machines=[
    ("synrm_3f_60s", Geometry3f, Computation3f, (1.5, 1.5)),
    ("synrm_5f_60s", Geometry5f, Computation5f, (1.5, 1.5, 0.0, 0.0)),
]


def _generate_one_lambda_barriers(seed):
    # rotor_r_min/rotor_r_max are pure-Python derived params, so a throwaway
    # instance (no Ansys) is enough to size the generator.
    geometry = Geometry3f()
    dummy = LiveDesign(m2d=None, geometry=geometry, computation=Computation3f(geometry))
    generator = HacklGenerator_OneLambda(dummy, R_STATOR_END, offset=OFFSET)

    np.random.seed(seed)
    while True:
        params = generator.random_parameters()
        generator.set_parameters(params)
        barriers = generator.generate_barriers()
        barriers = generator.split_barriers(barriers)
        if generator.feasible_barriers(barriers):
            return barriers


def test_synrm_3f_60s_and_5f_60s_solve_with_same_rotor(tmp_path):
    for seed in SEEDS:
        barriers=_generate_one_lambda_barriers(seed)

        for name, geometry_cls, computation_cls, current_setpoint in Machines:
            geometry=geometry_cls()
            computation=computation_cls(geometry)
            design=LiveDesign.create(
                f"Check60_{name}_{seed}",
                "Design01",
                str(tmp_path/f"{name}_{seed}.aedt"),
                geometry,
                computation,
                version=AEDT_VERSION,
                non_graphical=True,
                new_desktop=False,
                close_on_exit=False,
            )
            try:
                design.add_rotor()
                for barrier in barriers:
                        design.add_rotor_barrier(barrier)
                torque=design.compute(*current_setpoint, NUM_CORES=NUM_CORES)
                assert torque is not None, f"{name} seed {seed}: torque is None"
            finally:
                design.close_project()
