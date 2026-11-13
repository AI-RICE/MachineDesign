"""Issue 77: verify dq voltage is speed independent and only need to be computed with one real Ansys simulation.
Hypothesis: flux linkage psi_h(theta_el,i) depends only on rotor position and dq currents, not on electrical speed
w. If true, a single reference solve predicts voltage at different speed via
V_dh=Rs*Idh+w*(dpsi_dh/dtheta_el-h*ps_qh)
V_qh=Rs*Iqh+w*(dpsi_qh/dtheta_el+h*ps_dh)

Run explicitly with:

    pytest -m ansys tests/test_dq_voltage_computation_5f_40s.py

Requires a running Ansys Electronics Desktop session and a free license seat.
"""

import numpy as np
import pytest

from machine_design.config import load_config
from machine_design.designs.design import Design as LiveDesign
from machine_design.optimization.generators import HacklGenerator_OneLambda
from motors.synrm_5f_40s import Computation, Geometry

pytestmark = pytest.mark.ansys

AEDT_VERSION = load_config()["aedt_version"]
R_STATOR_END = 0.7
OFFSET = 0.35
SEED = 42
NUM_CORES = 1
f_ref = 50
Nper = 2
PointPer = "201"
frequencies = (25.0, 50.0, 75.0, 100.0)
current_setpoint = (1.5, 1.5, 0.5, 0.5)


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


def _last_period_terms(y, theta, point_per):
    # backward difference using real adjacent samples
    y = np.asarray(y)[-(point_per + 1) :]
    theta = np.asarray(theta)[-(point_per + 1) :]
    h = theta[1] - theta[0]
    dy = (y[1:] - y[:-1]) / h
    y_half = (y[1:] + y[:-1]) / 2
    return dy, y_half


def _compute_at(f, barriers, tmp_path):
    geometry = Geometry()
    computation = Computation(geometry, f=f, Nper=str(Nper))
    computation.oper_params["PointPer"] = PointPer
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
        for harmonic in (1, 3):
            np.testing.assert_allclose(out[f"Flux_d{harmonic}"], ref[f"Flux_d{harmonic}"], rtol=1e-3, atol=1e-5, err_msg=f"Flux_d{harmonic} differs at {f}Hz vs {f_ref}Hz")
            np.testing.assert_allclose(out[f"Flux_q{harmonic}"], ref[f"Flux_q{harmonic}"], rtol=1e-3, atol=1e-5, err_msg=f"Flux_q{harmonic} differs at {f}Hz vs {f_ref}Hz")


def test_predicted_voltage_solve(results_by_frequency):
    ref = results_by_frequency[f_ref]
    theta = ref["theta_el"]
    computation = Computation(Geometry())
    computation.oper_params["PointPer"] = PointPer
    Rstat = computation.Rstat
    point_per = int(computation.oper_params["PointPer"])
    assert len(theta) == Nper * (point_per - 1) + 1, f"unexpected array length {len(theta)}, check Ansys output convention"
    dFlux_d1, Flux_d1_half = _last_period_terms(ref["Flux_d1"], theta, point_per)
    dFlux_q1, Flux_q1_half = _last_period_terms(ref["Flux_q1"], theta, point_per)
    dFlux_d3, Flux_d3_half = _last_period_terms(ref["Flux_d3"], theta, point_per)
    dFlux_q3, Flux_q3_half = _last_period_terms(ref["Flux_q3"], theta, point_per)

    def _check(label, w, out):
        Id1, Iq1, Id3, Iq3 = current_setpoint
        V_d1_pred = Rstat * Id1 + w * (dFlux_d1 - Flux_q1_half)
        V_q1_pred = Rstat * Iq1 + w * (dFlux_q1 + Flux_d1_half)
        V_d3_pred = Rstat * Id3 + w * (dFlux_d3 - 3 * Flux_q3_half)
        V_q3_pred = Rstat * Iq3 + w * (dFlux_q3 + 3 * Flux_d3_half)

        for harmonic, V_d_pred, V_q_pred in ((1, V_d1_pred, V_q1_pred), (3, V_d3_pred, V_q3_pred)):
            V_d_actual = out[f"V_d{harmonic}"][-point_per:]
            V_q_actual = out[f"V_q{harmonic}"][-point_per:]
            abs_d = np.linalg.norm(V_d_pred - V_d_actual)
            abs_q = np.linalg.norm(V_q_pred - V_q_actual)
            rel_d = np.linalg.norm(V_d_pred - V_d_actual) / np.linalg.norm(V_d_actual)
            rel_q = np.linalg.norm(V_q_pred - V_q_actual) / np.linalg.norm(V_q_actual)
            print(f"{label} h={harmonic}: V_d abs err {abs_d:.4f}V (actual norm {np.linalg.norm(V_d_actual):.4f}V), V_q abs err {abs_q:.4f}V (actual norm {np.linalg.norm(V_q_actual):.4f}V)")
            assert rel_d < 1e-3, f"{label} h={harmonic}: V_d rel err {rel_d: 0.2%} exceeds 0.1% threshold"
            assert rel_q < 1e-3, f"{label} h={harmonic}: V_q rel err {rel_q: 0.2%} exceeds 0.1% threshold"

    _check(f"self-check at {f_ref}Hz", 2 * np.pi * f_ref, ref)
    for f, out in results_by_frequency.items():
        if f == f_ref:
            continue
        _check(f"{f}Hz", 2 * np.pi * f, out)
