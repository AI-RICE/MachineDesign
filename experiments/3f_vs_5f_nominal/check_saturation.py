import os

import numpy as np
from machine_design.config import load_config
from machine_design.designs.design import Design as LiveDesign
from machine_design.optimization import HacklGenerator_SixLambdas

from motors.synrm_3f_60s import Geometry as Geometry3f, Computation as Computation3f
from motors.synrm_5f_60s import Geometry as Geometry5f, Computation as Computation5f

AEDT_VERSION = load_config()["aedt_version"]
r_stator_end = 0.7
offset = 0.7 / 2

os.makedirs("data", exist_ok=True)


class LinearIronGeometry3f(Geometry3f):
    def set_iron(self):
        self.Fe = "LinearIron"

    def build_stator(self, m2d):
        self._create_material(m2d, "LinearIron", permeability=3000, conductivity=2e6, density=7650)
        super().build_stator(m2d)


class LinearIronGeometry5f(Geometry5f):
    def set_iron(self):
        self.Fe = "LinearIron"

    def build_stator(self, m2d):
        self._create_material(m2d, "LinearIron", permeability=3000, conductivity=2e6, density=7650)
        super().build_stator(m2d)


CASES = {
    "case1_3f": dict(
        geometry_cls=Geometry3f, linear_geometry_cls=LinearIronGeometry3f, computation_cls=Computation3f,
        barrier_X=np.array([7.177, 21.179, 33.034, 13.549, 26.499, 38.354, 0.227, 0.347, 0.449, 0.273, 0.373, 0.478]),
        current=(-0.6611, -1.0383),
    ),
    "case2_5f": dict(
        geometry_cls=Geometry5f, linear_geometry_cls=LinearIronGeometry5f, computation_cls=Computation5f,
        barrier_X=np.array([6.696, 21.349, 32.554, 15.408, 25.639, 38.468, 0.216, 0.329, 0.442, 0.296, 0.387, 0.491]),
        current=(-0.4279, -0.5442, 0.0, 0.0),
    ),
}


def run_one(name, geometry_cls, computation_cls, barrier_X, current, tag):
    geometry = geometry_cls()
    computation = computation_cls(geometry)
    file_name = os.path.join(os.getcwd(), "data", f"{name}_{tag}.aedt")
    design = LiveDesign.create(
        f"CheckSaturation_{name}_{tag}", "Design01", file_name,
        geometry, computation, version=AEDT_VERSION, non_graphical=True, new_desktop=True, close_on_exit=False,
    )
    generator = HacklGenerator_SixLambdas(design, r_stator_end, offset=offset)
    params = generator.X_to_params(barrier_X)
    generator.set_parameters(params)
    barriers = generator.generate_barriers()
    barriers = generator.split_barriers(barriers)

    design.add_rotor()
    for barrier in barriers:
        design.add_rotor_barrier(barrier)

    out = design.compute(*current, NUM_CORES=1)
    design.delete_rotor()
    design.close_project()

    L_d_key = "L_d" if "L_d" in out else "Ld1"
    L_q_key = "L_q" if "L_q" in out else "Lq1"
    print(f"{name} [{tag}]: L_d={np.mean(out[L_d_key]):.6f} L_q={np.mean(out[L_q_key]):.6f} torque={np.mean(out['Moving1.Torque']):.3f}Nm")


for name, cfg in CASES.items():
    run_one(name, cfg["geometry_cls"], cfg["computation_cls"], cfg["barrier_X"], cfg["current"], tag="saturated")

