"""Issue 77: verify dq voltage is speed independent and only need to be computed with one real Ansys simulation.
Hypothesis: flux linkage psi_h(theta_el,i) depends only on rotor position and dq currents, not on electrical speed
w. If true, a single reference solve predicts voltage at different speed via
V_dh=Rs*Idh+w*(dpsi_dh/dtheta_el-h*ps_qh)
V_qh=Rs*Iqh+w*(dpsi_qh/dtheta_el+h*ps_dh)

Run explicitly with:

    pytest -m ansys tests/test_dq_voltage_speed_independence.py

Requires a running Ansys Electronics Desktop session and a free license seat.
"""

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
SEED = 42
NUM_CORES = 1
f_ref = 50
Nper = "1"
frequencies = (25.0, 50.0, 75.0, 100.0)
current_setpoint = (1.5, 1.5)


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
        new_desktop=True,
        close_on_exit=False,
    )


def _fft_derivative(y, theta):
    # theta spans a full period with theta[-1] == theta[0] + 2*pi (duplicate endpoint)
    y = np.asarray(y)[:-1]
    n = len(y)
    harmonic = np.fft.fftfreq(n, d=1.0 / n)
    dy = np.fft.ifft(1j * harmonic * np.fft.fft(y)).real
    return np.r_[dy, dy[0]]



def _compute_at(f, barriers, tmp_path):
    geometry = Geometry()
    computation = Computation(geometry, f=f, Nper=Nper)
    design = _create_design(LiveDesign, f"verifySpeedIndependence_f{f:g}", str(tmp_path / f"f{f:g}.aedt"), extra_args=(geometry, computation))
    _add_rotor(design, barriers)
    out = design.compute(*current_setpoint, NUM_CORES=NUM_CORES)
    design.delete_rotor()
    return design, out


@pytest.fixture(scope="module")
def results_by_frequency(tmp_path_factory):
    barriers = _generate_one_lambda_barriers(SEED)
    tmp_path = tmp_path_factory.mktemp("speed_independence")
    designs, results = [], {}
    try:
        for f in frequencies:
            design, out = _compute_at(f, barriers, tmp_path)
            designs.append(design)
            results[f] = out
        yield results
    finally:
        if designs:
            designs[0].m2d.close_desktop()


def test_fluxlinkage_independence(results_by_frequency):
    ref = results_by_frequency[f_ref]
    for f, out in results_by_frequency.items():
        if f == f_ref:
            continue
        np.testing.assert_allclose(out["Flux_d"], ref["Flux_d"], rtol=1e-3, atol=1e-5, err_msg=f"Flux_d differs at {f}Hz vs {f_ref}Hz")
        np.testing.assert_allclose(out["Flux_q"], ref["Flux_q"], rtol=1e-3, atol=1e-5, err_msg=f"Flux_q differs at {f}Hz vs {f_ref}Hz")


def test_predicted_voltage_solve(results_by_frequency):
    ref = results_by_frequency[f_ref]
    theta = ref["theta_el"]
    Rstat = Computation(Geometry()).Rstat
    dFlux_d=np.gradient(ref["Flux_d"],theta)
    dFlux_q=np.gradient(ref["Flux_q"],theta)

    def _report(label, w, out):
        V_d_pred = Rstat * current_setpoint[0] + w * (dFlux_d - ref["Flux_q"])
        V_q_pred = Rstat * current_setpoint[1] + w * (dFlux_q + ref["Flux_d"])
        err_d = np.abs(V_d_pred - out["V_d"])
        err_q = np.abs(V_q_pred - out["V_q"])
        rel_d = np.linalg.norm(err_d) / np.linalg.norm(out["V_d"])
        rel_q = np.linalg.norm(err_q) / np.linalg.norm(out["V_q"])
        print(f"{label}: V_d rel err={rel_d:.4%}, V_q rel err={rel_q:.4%}")
        assert rel_d<0.2, f"{label}: V_d rel err {rel_d: 0.2%} exceeds 20% threshold"
        assert rel_q<0.05, f"{label}: V_q rel err {rel_q: 0.2%} exceeds 20% threshold"

    _report(f"self-check at {f_ref}Hz", 2 * np.pi * f_ref, ref)
    for f, out in results_by_frequency.items():
        if f == f_ref:
            continue
        _report(f"{f}Hz", 2 * np.pi * f, out)

