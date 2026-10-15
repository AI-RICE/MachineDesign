"""Compare old winding assignment (winding.py) vs. the new one (winding_layout_JL.py).

Run explicitly with:

    pytest -m ansys tests/test_ansys_new_winding_regression_synrm_3f_36s.py

Requires a running Ansys Electronics Desktop session and a free license seat.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from machine_design.config import load_config
from machine_design.designs.design import Design as LiveDesign
from machine_design.optimization.generators import HacklGenerator_OneLambda
from motors.synrm_3f_36s import Computation, Geometry

pytestmark = pytest.mark.ansys

AEDT_VERSION = load_config()["aedt_version"]
R_STATOR_END = 0.7
OFFSET = 0.35
SEEDS = (42, 43)
NUM_CORES = 1
current_setpoint = (1.5, 1.5)
check_expressions = [
    "Flux_d",
    "Flux_q",
    "Ui_d",
    "Ui_q",
    "I_d",
    "I_q",
    "L_d",
    "L_q",
    "Irms",
]
# we should verify all the above 9 vars + Moving1.Torque (a total of 10 vars with different seeds)


class ComputationNewWinding(Computation):
    def assign_stator_coils(self, m2d):
        m = self.geometry.n_phases
        phase_currents = [f"Im * cos(w*Time-{360 * k / m}deg+epsI)" for k in range(m)]
        self.assign_phase_windings_new(m2d, phase_currents, coil_pitch=9)


def _load_legacy_design_class():
    legacy_path = Path(__file__).parent / "legacy" / "design.py"
    spec = importlib.util.spec_from_file_location("legacy_design", legacy_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Design


def _generate_one_lambda_barriers(seed):
    # rotor_r_min/rotor_r_max are pure-Python derived params, so a throwaway
    # instance (no Ansys) is enough to size the generator.
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


def _add_rotor(design, barriers):
    design.add_rotor()
    for barrier in barriers:
        design.add_rotor_barrier(barrier)


def _create_design(design_cls, project_name, file_name, extra_args=()):
    return design_cls.create(
        project_name,
        "Design01",
        file_name,
        *extra_args,
        version=AEDT_VERSION,
        non_graphical=True,
        new_desktop=False,
        close_on_exit=False,
    )


def _compute_torque(design, barriers, add_rotor, setpoint=current_setpoint):
    add_rotor(design, barriers)
    torque = design.compute(*setpoint, NUM_CORES=NUM_CORES)
    design.delete_rotor()
    return torque


def test_legacy_live_and_new_winding_design_match_across_rotor_rebuilds(tmp_path):
    barrier_sets = [_generate_one_lambda_barriers(seed) for seed in SEEDS]
    LegacyDesign = _load_legacy_design_class()

    legacy_design, live_design, new_design = None, None, None
    try:
        legacy_design = _create_design(LegacyDesign, "RegressionTest_legacy", str(tmp_path / "legacy.aedt"))
        legacy_design.solution_expressions = ["Moving1.Torque", *check_expressions]

        def _legacy_extract_results(solutions):
            out = {}
            for expr in legacy_design.solution_expressions:
                out[expr] = np.array(solutions.data_real(expr, convert_to_SI=True))  # use SI units to compare
            return out

        legacy_design.extract_results = _legacy_extract_results

        live_geometry = Geometry()
        live_computation = Computation(live_geometry)
        live_design = _create_design(
            LiveDesign,
            "RegressionTest_live",
            str(tmp_path / "live.aedt"),
            extra_args=(live_geometry, live_computation),
        )

        new_geometry = Geometry()
        new_computation = ComputationNewWinding(new_geometry)
        new_design = _create_design(
            LiveDesign,
            "RegressionTest_new",
            str(tmp_path / "new.aedt"),
            extra_args=(new_geometry, new_computation),
        )

        for seed, barriers in zip(SEEDS, barrier_sets):
            torque_legacy = _compute_torque(legacy_design, barriers, _add_rotor)
            torque_live = _compute_torque(live_design, barriers, _add_rotor)
            torque_new = _compute_torque(new_design, barriers, _add_rotor)

            assert torque_legacy is not None, f"legacy torque is None for seed {seed}"
            assert torque_live is not None, f"live torque is None for seed {seed}"
            assert torque_new is not None, f"new torque is None for seed {seed}"
            np.testing.assert_allclose(torque_live["Moving1.Torque"], torque_legacy["Moving1.Torque"], rtol=1e-3, err_msg=f"live legacy mismatch for Moving1.Torque, seed {seed}")
            np.testing.assert_allclose(torque_new["Moving1.Torque"], torque_legacy["Moving1.Torque"], rtol=1e-3, err_msg=f"new legacy mismatch for Moving1.Torque, seed {seed}")
            np.testing.assert_allclose(torque_new["Moving1.Torque"], torque_live["Moving1.Torque"], rtol=1e-3, err_msg=f"new live mismatch for Moving1.Torque, seed {seed}")
            for expr in check_expressions:
                np.testing.assert_allclose(torque_live[expr], torque_legacy[expr], rtol=1e-3, err_msg=f"live legacy mismatch for {expr}, seed {seed}")
                np.testing.assert_allclose(torque_new[expr], torque_legacy[expr], rtol=1e-3, err_msg=f"new legacy mismatch for {expr}, seed {seed}")
                np.testing.assert_allclose(torque_new[expr], torque_live[expr], rtol=1e-3, err_msg=f"new live mismatch for {expr}, seed {seed}")

    finally:
        if live_design is not None:
            live_design.close_project()
        elif new_design is not None:
            new_design.close_project()
        elif legacy_design is not None:
            legacy_design.close_project()
