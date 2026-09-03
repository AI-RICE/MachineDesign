"""Golden-master check: tests/legacy/design.py vs. the live machine_design/design.py.

Generates two feasible rotor designs (different seeds) with HacklGenerator_OneLambda and
runs each through add_rotor -> compute -> delete_rotor on the SAME Design instance, both
with the frozen reference implementation and the current one - mirroring how an
optimization loop reuses one Design across many candidate rotors. Asserts the resulting
torque waveforms match at each step. Run explicitly with:

    pytest -m ansys tests/test_ansys_regression.py

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


def _compute_torque(design, barriers, add_rotor):
    add_rotor(design, barriers)
    torque = design.compute(NUM_CORES=NUM_CORES)
    design.delete_rotor()
    return torque


def test_legacy_and_live_design_match_across_rotor_rebuilds(tmp_path):
    barrier_sets = [_generate_one_lambda_barriers(seed) for seed in SEEDS]
    LegacyDesign = _load_legacy_design_class()

    legacy_design, live_design = None, None
    try:
        legacy_design = _create_design(LegacyDesign, "RegressionTest_legacy", str(tmp_path / "legacy.aedt"))

        live_geometry = Geometry()
        live_computation = Computation(live_geometry)
        live_design = _create_design(
            LiveDesign,
            "RegressionTest_live",
            str(tmp_path / "live.aedt"),
            extra_args=(live_geometry, live_computation),
        )

        for seed, barriers in zip(SEEDS, barrier_sets):
            torque_legacy = _compute_torque(legacy_design, barriers, _add_rotor)
            torque_live = _compute_torque(live_design, barriers, _add_rotor)

            assert torque_legacy is not None, f"legacy torque is None for seed {seed}"
            assert torque_live is not None, f"live torque is None for seed {seed}"
            np.testing.assert_allclose(torque_live, torque_legacy, rtol=1e-6, err_msg=f"mismatch for seed {seed}")
    finally:
        if live_design is not None:
            live_design.close_project()
        elif legacy_design is not None:
            legacy_design.close_project()
