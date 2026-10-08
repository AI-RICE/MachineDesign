import numpy as np
import pytest

from machine_design.config import load_config
from machine_design.designs.design import Design
from machine_design.optimization.generators import HacklGenerator_OneLambda
from motors.synrm_5f_40s import Computation
from motors.synrm_5f_60s import Geometry

pytestmark = pytest.mark.ansys

AEDT_VERSION = load_config()["aedt_version"]
R_STATOR_END = 0.7
OFFSET = 0.35
SEED = 42
NUM_CORES = 1
current_setpoint = (0.8, 0.9, 0.2, -0.15)
Rtol = 0.03


def _feasible_barriers(geometry):
    dummy = Design(m2d=None, geometry=geometry, computation=Computation(geometry))
    generator = HacklGenerator_OneLambda(dummy, R_STATOR_END, offset=OFFSET)

    np.random.seed(SEED)
    while True:
        generator.set_parameters(generator.random_parameters())
        barriers = generator.split_barriers(generator.generate_barriers())
        if generator.feasible_barriers(barriers):
            return barriers


def test_torque_and_voltage_identities(tmp_path):
    geometry = Geometry()
    computation = Computation(geometry)
    barriers = _feasible_barriers(geometry)

    design = None
    try:
        design = Design.create(
            "DqIdentityCheck60s",
            "Design01",
            str(tmp_path / "dq_identity_60s.aedt"),
            geometry,
            computation,
            version=AEDT_VERSION,
            non_graphical=True,
            new_desktop=False,
            close_on_exit=False,
        )
        design.add_rotor()
        for barrier in barriers:
            design.add_rotor_barrier(barrier)
        out = design.compute(*current_setpoint, NUM_CORES=NUM_CORES)
        assert out is not None, "solve returned no results"

        np.testing.assert_allclose(np.mean(out["Torque_dq"]), np.mean(out["Moving1.Torque"]), rtol=Rtol, err_msg="torque identity mismatch")

        for h in (1, 3):
            w = h * computation.w
            d, q = f"d{h}", f"q{h}"
            vl_q = out[f"V_{q}"] - computation.Rstat * out[f"I_{q}"]
            vl_d = out[f"V_{d}"] - computation.Rstat * out[f"I_{d}"]
            print(f"h={h}: vl_d mean={np.mean(vl_d):.4f}, -w*Flux_q mean={np.mean(-w * out[f'Flux_{q}']):.4f}, ratio={np.mean(vl_d)/np.mean(-w*out[f'Flux_{q}']):.4f}")
            np.testing.assert_allclose(np.mean(vl_q), np.mean(w * out[f"Flux_{d}"]), rtol=Rtol, err_msg=f"voltage identity mismatch, q-axis, harmonic {h}")
            np.testing.assert_allclose(np.mean(vl_d), np.mean(-w * out[f"Flux_{q}"]), rtol=Rtol, err_msg=f"voltage identity mismatch, d-axis, harmonic {h}")
    finally:
        if design is not None:
            design.close_project()
