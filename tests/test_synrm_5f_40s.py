from types import SimpleNamespace

import numpy as np
import pytest

from machine_design.transforms import electrical_angle, to_dq
from motors.synrm_3f_36s import Geometry as BaseGeometry
from motors.synrm_5f_40s import Computation, Geometry


@pytest.fixture
def base_geometry():
    return BaseGeometry()


@pytest.fixture
def geometry2():
    return Geometry()


@pytest.fixture
def computation2(geometry2):
    return Computation(geometry2)


def test_geom_params_overrides_slot_number_only(base_geometry, geometry2):
    expected = dict(base_geometry.geom_params)
    expected["SlotNumber"] = "40"
    assert geometry2.geom_params == expected


def test_slot_params_overrides_bs1_bs2_setangle_only(base_geometry, geometry2):
    expected = dict(base_geometry.slot_params)
    expected["Bs1"] = "3.0mm"
    expected["Bs2"] = "4.3mm"
    expected["SetAngle"] = "9deg"
    assert geometry2.slot_params == expected


def test_wind_params_overrides_nc_only(base_geometry, geometry2):
    expected = dict(base_geometry.wind_params)
    expected["Nc"] = "113"
    assert geometry2.wind_params == expected


def test_oper_params_fully_replaced(computation2):
    assert computation2.oper_params == {
        "Id1": "0.0A",
        "Iq1": "0.0A",
        "Id3": "0.0A",
        "Iq3": "0.0A",
        "epsI1": "atan2(Iq1,Id1)",
        "epsI3": "atan2(Iq3,Id3)",
        "Im1": "sqrt(Id1^2+Iq1^2)",
        "Im3": "sqrt(Id3^2+Iq3^2)",
        "InitPos": "-45deg",
        "w": f"{2 * np.pi * 50}Hz",
        "RotSpeed": "1500.0rpm",
        "Nper": "1/10",
        "PointPer": "101",
    }


def test_derived_params_are_set(geometry2):
    assert hasattr(geometry2, "rotor_r_min")
    assert hasattr(geometry2, "rotor_r_max")


def test_solution_expressions_is_a_list_ending_in_torque(computation2):
    assert isinstance(computation2.solution_expressions, list)
    assert computation2.solution_expressions[0] == "Moving1.Position"
    assert computation2.solution_expressions[1] == "Moving1.Torque"
    assert computation2.solution_expressions[-1] == "L(PhaseE,PhaseE)"
    assert len(computation2.solution_expressions) == 42


def test_post_params_plot_names_unique(computation2):
    plot_names = list(computation2.post_params.values())
    assert len(plot_names) == len(set(plot_names))


def test_output_vars_is_empty(computation2):
    assert computation2.output_vars == {}


def test_Rstat_Lew_values(computation2):
    assert computation2.Rstat == 19.0
    assert computation2.Lew == 0.0


def test_extract_results_converts_units(computation2):
    computation2.set_variables(SimpleNamespace(variable_manager={}), Id1=1.0, Iq1=2.0, Id3=3.0, Iq3=4.0)

    def data_real(expr):
        letters = ["ABCDE".index(part[0]) + 1 for part in expr.split("Phase")[1:]]
        value = float(np.prod(letters) if letters else 7.0)
        return np.full(2, value * 1e9 if expr.startswith("L(") else value)

    solutions = SimpleNamespace(data_real=data_real, primary_sweep_values=[0.0, 1.0])
    out = computation2.extract_results(solutions)

    theta_el = electrical_angle(np.full(2, 7.0), computation2.InitPos, computation2.geometry.PolePairs, computation2.RotSign, degrees=True)
    current_phases = np.stack([np.full(2, i + 1.0) for i in range(5)], axis=-1)
    expected_I_d1, _ = to_dq(current_phases, theta_el, harmonic=1)

    assert out["I_d1"] == pytest.approx(expected_I_d1 / 1e3)
    assert out["Moving1.Torque"] == pytest.approx(np.full(2, 7.0))
