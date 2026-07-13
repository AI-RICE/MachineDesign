"""Golden-master check: tests/legacy/design.py vs. the live machine_design/design.py.

Generates one feasible rotor design with HacklGenerator_OneLambda, builds it with both
the frozen reference implementation and the current implementation, runs both through
Ansys, and asserts the resulting torque waveforms match. Run explicitly with:

    pytest -m ansys tests/test_ansys_regression.py

Requires a running Ansys Electronics Desktop session and a free license seat.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from machine_design.design import Design as LiveDesign
from machine_design.design_computation import Computation
from machine_design.design_geometry import Geometry
from machine_design.generators import HacklGenerator_OneLambda

pytestmark = pytest.mark.ansys

AEDT_VERSION = "2024.1"
R_STATOR_END = 0.7
OFFSET = 0.35
SEED = 42
NUM_CORES = 1


def _load_legacy_design_class():
    legacy_path = Path(__file__).parent / "legacy" / "design.py"
    spec = importlib.util.spec_from_file_location("legacy_design", legacy_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Design


def _generate_one_lambda_barriers():
    # rotor_r_min/rotor_r_max are pure-Python derived params, so a throwaway
    # instance (no Ansys) is enough to size the generator.
    geometry = Geometry()
    dummy = LiveDesign(m2d=None, geometry=geometry, computation=Computation(geometry))
    generator = HacklGenerator_OneLambda(dummy, R_STATOR_END, offset=OFFSET)

    np.random.seed(SEED)
    while True:
        params = generator.random_parameters()
        generator.set_parameters(params)
        barriers = generator.generate_barriers()
        barriers = generator.split_barriers(barriers)
        if generator.feasible_barriers(barriers):
            return barriers


def _build_and_compute(design_cls, project_name, file_name, barriers, extra_args=()):
    design = design_cls.create(
        project_name,
        "Design01",
        file_name,
        *extra_args,
        version=AEDT_VERSION,
        non_graphical=True,
        new_desktop=False,
        close_on_exit=False,
    )
    design.add_rotor()
    for barrier in barriers:
        design.add_rotor_barrier(barrier)
    torque = design.compute(NUM_CORES=NUM_CORES)
    design.delete_rotor()
    return design, torque


def test_legacy_and_live_design_produce_same_torque(tmp_path):
    barriers = _generate_one_lambda_barriers()
    LegacyDesign = _load_legacy_design_class()

    legacy_design, live_design = None, None
    try:
        legacy_design, torque_legacy = _build_and_compute(
            LegacyDesign, "RegressionTest_legacy", str(tmp_path / "legacy.aedt"), barriers
        )

        live_geometry = Geometry()
        live_computation = Computation(live_geometry)
        live_design, torque_live = _build_and_compute(
            LiveDesign,
            "RegressionTest_live",
            str(tmp_path / "live.aedt"),
            barriers,
            extra_args=(live_geometry, live_computation),
        )

        assert torque_legacy is not None
        assert torque_live is not None
        np.testing.assert_allclose(torque_live, torque_legacy, rtol=1e-6)
    finally:
        if live_design is not None:
            live_design.close_project()
        elif legacy_design is not None:
            legacy_design.close_project()
