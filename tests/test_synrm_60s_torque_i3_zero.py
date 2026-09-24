"""Goal: test 5f_60s performs the same with 3f_60s when Id3=Iq3=0 and the same rotor.

Run explicitly with:

    pytest -m ansys tests/test_synrm_60s_torque_i3_zero.py

Requires a running Ansys Electronics Desktop session and a free license seat.
"""

import numpy as np
import pytest

from machine_design.config import load_config
from machine_design.designs.design import Design as LiveDesign
from machine_design.optimization.generators import HacklGenerator_OneLambda
from machine_design.optimization.geometry import analyze_results
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
current_setpoints = (3.0, 3.0)


class Geometry3fSameNc(Geometry3f):
    def set_winds_params(self):
        super().set_winds_params()
        self.wind_params["Nc"] = "113"


class Geometry5fSameNc(Geometry5f):
    def set_winds_params(self):
        super().set_winds_params()
        self.wind_params["Nc"] = "113"


Machines = [
    ("synrm_3f_60s", Geometry3fSameNc, Computation3f, current_setpoints),
    ("synrm_5f_60s", Geometry5fSameNc, Computation5f, (*current_setpoints, 0.0, 0.0)),
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


def test_synrm_5f_60s_i3_zero_performs_same_with_3f_60s(tmp_path):
    for seed in SEEDS:
        barriers = _generate_one_lambda_barriers(seed)

        results = {}
        for name, geometry_cls, computation_cls, current_setpoint in Machines:
            geometry = geometry_cls()
            computation = computation_cls(geometry)
            design = LiveDesign.create(
                f"CheckI3zero_{name}_{seed}",
                "Design01",
                str(tmp_path / f"{name}_{seed}.aedt"),
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
                # design.m2d.variable_manager["Nper"] = "1"
                # design.m2d.variable_manager["PointPer"] = "301"  #results: 13.03%, 13.97%
                torque = design.compute(*current_setpoint, NUM_CORES=NUM_CORES)
                assert torque is not None, f"{name} seed {seed}: torque is None"
                TorAvg, _, TorRippleRms = analyze_results(torque["Moving1.Torque"])
                results[name] = TorAvg
            finally:
                design.close_project()

        torque_3f = results["synrm_3f_60s"]
        torque_5f = results["synrm_5f_60s"]
        diff_pct = abs(torque_3f - torque_5f) / abs(torque_3f) * 100
        print(f"seed {seed}: 3f TorAvg={torque_3f:.4f} Nm, 5f(i3=0) TorAvg={torque_5f:.4f} Nm, diff={diff_pct:.2f}%")
        assert diff_pct < 20, f"seed {seed}: 3f and 5f(i3=0) torque differ by {diff_pct:.2f}%, expected roughly comparable"
