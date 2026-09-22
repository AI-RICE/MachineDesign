""" check torque varies with differnet currents """

import numpy as np

from machine_design.config import load_config
from machine_design.designs.design import Design as LiveDesign
from machine_design.optimization.generators import HacklGenerator_OneLambda
from motors.synrm_3f_60s import Computation as Computation3f
from motors.synrm_3f_60s import Geometry as Geometry3f
from motors.synrm_5f_60s import Computation as Computation5f
from motors.synrm_5f_60s import Geometry as Geometry5f

AEDT_VERSION = load_config()["aedt_version"]
R_STATOR_END = 0.7
OFFSET = 0.35
SEED=42
NUM_CORES = 1


def generate_barriers(geometry_cls, computation_cls, seed):
    geometry=geometry_cls()
    dummy=LiveDesign(m2d=None, geometry=geometry, computation=computation_cls(geometry))
    generator=HacklGenerator_OneLambda(dummy, R_STATOR_END, offset=OFFSET)

    np.random.seed(seed)
    while True:
        params = generator.random_parameters()
        generator.set_parameters(params)
        barriers = generator.generate_barriers()
        barriers = generator.split_barriers(barriers)
        if generator.feasible_barriers(barriers):
            return barriers

def check(name, geometry_cls, computation_cls, project_name, current_setpoint, barriers):
    geometry=geometry_cls()
    computation=computation_cls(geometry)
    design=LiveDesign.create(
        project_name,
        "Design01",
        f"data/{project_name}.aedt",
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
        out = design.compute(*current_setpoint, NUM_CORES=NUM_CORES)
        torque = out["Moving1.Torque"]
        return {"name":name, "current":current_setpoint, "mean_torque":np.mean(torque[:-1])}
    finally:
        design.close_project()


if __name__ == "_main__":
    barriers = generate_barriers(Geometry3f, Computation3f, SEED)

    results = [
        check("synrm_3f_60s", Geometry3f, Computation3f, "check_zero", (0.0, 0.0), barriers),
        check("synrm_3f_60s", Geometry3f, Computation3f, "check_all", (1.5, 1.5), barriers),
        check("synrm_3f_60s", Geometry3f, Computation3f, "check_minus", (1.5, -1.5), barriers),
        check("synrm_5f_60s", Geometry5f, Computation5f, "check_i1_only", (7.0711, 7.0711, 0.0, 0.0), barriers),
        check("synrm_5f_60s", Geometry5f, Computation5f, "check_i3_only", (0.0, 0.0, 7.0711, 7.0711), barriers),
        check("synrm_5f_60s", Geometry5f, Computation5f, "check_all", (7.0711, 7.0711, 7.0711, 7.0711), barriers),
    ]

    print(f"{'name':<16}{'current<30'}{'mean_torque':>12}")
    for r in results:
        print(f"{r['name']:<16}{str(r['current']):<30}{r['mean_torque']:>12.4f}")