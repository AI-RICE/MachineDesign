"""Issue 77: verify dq voltage is speed independent and only need to be computed with one real Ansys simulation.
Hypothesis: flux linkage psi_h(theta_el,i) depends only on rotor position and dq currents, not on electrical speed
w. If true, a single reference solve predicts voltage at different speed via
V_dh=Rs*Idh+w*(dpsi_dh/dtheta_el-h*ps_qh)-h*w*Lew*Iqh
V_qh=Rs*Iqh+w*(dpsi_qh/dtheta_el+h*ps_dh)+h*w*Lew*Idh
Also verify voltage_from_flux matches InducedVoltage method.

Run explicitly with:

    pytest -m ansys tests/test_ansys_dq_voltage_computation_5f_40s.py

Requires a running Ansys Electronics Desktop session and a free license seat.
"""

import numpy as np
import pytest

from machine_design.config import load_config
from machine_design.designs.design import Design as LiveDesign
from machine_design.generic.transforms import induced_voltage_angle, to_dq, voltage_from_flux
from machine_design.optimization.generators import HacklGenerator_OneLambda
from motors.synrm_5f_40s import Computation, Geometry

pytestmark = pytest.mark.ansys

AEDT_VERSION = load_config()["aedt_version"]
R_STATOR_END = 0.7
OFFSET = 0.35
SEED = 42
NUM_CORES = 1
f_ref = 50
PointPer = "201"
frequencies = (25.0, 50.0, 75.0)
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


def _compute_at(f, barriers, tmp_path):
    geometry = Geometry()
    computation = Computation(geometry, f=f)
    computation.oper_params["PointPer"] = PointPer
    design = _create_design(LiveDesign, f"verifySpeedIndependence_f{f:g}", str(tmp_path / f"f{f:g}.aedt"), extra_args=(geometry, computation))
    _add_rotor(design, barriers)
    out, _ = design.compute(*current_setpoint, NUM_CORES=NUM_CORES)
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
            rel_d = np.max(np.abs(out[f"Flux_d{harmonic}"] - ref[f"Flux_d{harmonic}"])) / np.max(np.abs(ref[f"Flux_d{harmonic}"]))
            rel_q = np.max(np.abs(out[f"Flux_q{harmonic}"] - ref[f"Flux_q{harmonic}"])) / np.max(np.abs(ref[f"Flux_q{harmonic}"]))
            print(f"{f}Hz vs {f_ref}Hz h={harmonic}: Flux_d max rel diff {rel_d:.4%}, Flux_q max rel diff {rel_q:.4%}")
            np.testing.assert_allclose(out[f"Flux_d{harmonic}"], ref[f"Flux_d{harmonic}"], rtol=1e-3, atol=1e-5, err_msg=f"Flux_d{harmonic} differs at {f}Hz vs {f_ref}Hz")
            np.testing.assert_allclose(out[f"Flux_q{harmonic}"], ref[f"Flux_q{harmonic}"], rtol=1e-3, atol=1e-5, err_msg=f"Flux_q{harmonic} differs at {f}Hz vs {f_ref}Hz")


def test_predicted_voltage_solve(results_by_frequency):
    ref = results_by_frequency[f_ref]
    theta = ref["theta_el"]
    computation = Computation(Geometry())
    computation.oper_params["PointPer"] = PointPer
    Rstat = computation.Rstat
    assert len(theta) >= 2, f"unexpected array length {len(theta)}, check Ansys output convention"

    def _check(label, w, out):
        Id1, Iq1, Id3, Iq3 = current_setpoint

        for harmonic, Idh, Iqh, Flux_dh, Flux_qh in ((1, Id1, Iq1, ref["Flux_d1"], ref["Flux_q1"]), (3, Id3, Iq3, ref["Flux_d3"], ref["Flux_q3"])):
            V_d_pred, V_q_pred = voltage_from_flux(Flux_dh, Flux_qh, theta, Idh, Iqh, Rstat, w, harmonic=harmonic, Lew=computation.Lew)
            V_d_actual = out[f"V_d{harmonic}"]
            V_q_actual = out[f"V_q{harmonic}"]
            rel_d = np.linalg.norm(V_d_pred - V_d_actual) / np.linalg.norm(V_d_actual)
            rel_q = np.linalg.norm(V_q_pred - V_q_actual) / np.linalg.norm(V_q_actual)
            print(f"{label} h={harmonic}: V_d rel err {rel_d:.4%}, V_q rel err {rel_q:.4%}")
            assert rel_d < 1e-3, f"{label} h={harmonic}: V_d rel err {rel_d: 0.2%} exceeds 0.1% threshold"
            assert rel_q < 1e-3, f"{label} h={harmonic}: V_q rel err {rel_q: 0.2%} exceeds 0.1% threshold"

    for f, out in results_by_frequency.items():
        if f == f_ref:
            continue
        _check(f"{f}Hz", 2 * np.pi * f, out)


def test_voltage_from_flux_matches_induced_voltage(tmp_path):
    """test voltage from flux equals induced voltage"""
    barriers = _generate_one_lambda_barriers(SEED)
    geometry = Geometry()
    computation = Computation(geometry, f=f_ref)
    computation.oper_params["PointPer"] = PointPer
    design = _create_design(LiveDesign, "verifyInducedVoltageMatch", str(tmp_path / "f_ref.aedt"), extra_args=(geometry, computation))
    try:
        _add_rotor(design, barriers)
        out, solutions = design.compute(*current_setpoint, NUM_CORES=NUM_CORES)
        design.delete_rotor()
    finally:
        design.m2d.close_desktop()

    theta_el = out["theta_el"]
    _, vind_phases, current_phases, _ = computation.extract_phase_results(solutions)
    theta_vind = induced_voltage_angle(theta_el)
    Vind_d1, Vind_q1 = to_dq(vind_phases, theta_vind, harmonic=1)
    Vind_d3, Vind_q3 = to_dq(vind_phases, theta_vind, harmonic=3)

    Im1 = np.sqrt(computation.Id1**2 + computation.Iq1**2)
    Im3 = np.sqrt(computation.Id3**2 + computation.Iq3**2)
    epsI1 = np.arctan2(computation.Iq1, computation.Id1)
    epsI3 = np.arctan2(computation.Iq3, computation.Id3)

    time = np.array(solutions.primary_sweep_values)
    dI_dt_phases = np.zeros((len(time), geometry.n_phases))
    for k in range(geometry.n_phases):
        phase_offset = -2 * np.pi * k / geometry.n_phases
        dI_dt_phases[:, k] = -Im1 * computation.w * np.sin(computation.w * time + phase_offset + epsI1 - np.pi) - Im3 * 3 * computation.w * np.sin(3 * (computation.w * time + phase_offset) + epsI3 - np.pi)

    V_rest_phases = computation.Rstat * current_phases + computation.Lew * dI_dt_phases

    V_rest_d1, V_rest_q1 = to_dq(V_rest_phases, theta_el, harmonic=1)
    V_rest_d3, V_rest_q3 = to_dq(V_rest_phases, theta_el, harmonic=3)

    induced = {
        "V_d1": Vind_d1 + V_rest_d1,
        "V_q1": Vind_q1 + V_rest_q1,
        "V_d3": Vind_d3 + V_rest_d3,
        "V_q3": Vind_q3 + V_rest_q3,
    }

    for expr, V_induced in induced.items():
        rel = np.linalg.norm(out[expr] - V_induced[1:]) / np.linalg.norm(V_induced[1:])
        print(f"{expr}: rel err {rel:.4%}")
        assert rel < 1e-3, f"{expr} rel err {rel: 0.2%} exceeds 0.1% threshold to induced voltage"
