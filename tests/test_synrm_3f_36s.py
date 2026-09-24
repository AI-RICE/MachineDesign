from types import SimpleNamespace

import numpy as np
import pytest

from machine_design.generic.transforms import to_dq
from motors.synrm_3f_36s import Computation, Geometry


@pytest.fixture
def geometry():
    return Geometry()


@pytest.fixture
def computation(geometry):
    return Computation(geometry)


def test_setup_name(computation):
    assert computation.setup_name == "Setup1"


def test_iron(geometry):
    assert geometry.Fe == "Cogent Power - M350-50A, B-H at 50Hz"


def test_geom_params(geometry):
    assert geometry.geom_params == {
        "DiaStatorGap": "79mm",
        "DiaStatorYoke": "125mm",
        "Airgap": "0.225mm",
        "SlotNumber": "36",
        "SlotType": "3",
        "DiaShaft": "25mm",
        "StackLength": "85mm",
    }


def test_slot_params(geometry):
    assert geometry.slot_params == {
        "Hs0": "0.95mm",
        "Hs1": "0.31mm",
        "Hs2": "8.24mm",
        "Bs0": "2.2mm",
        "Bs1": "3.37mm",
        "Bs2": "4.8mm",
        "Rs": "1.5mm",
        "SetAngle": "10deg",
    }


def test_wind_params(geometry):
    assert geometry.wind_params == {
        "Layers": "1",
        "ParallelPaths": "1",
        "CoilPitch": "9",
        "SlotLiner": "0.3mm",
        "SpaceLayers": "0.2mm",
        "Nc": "68",
    }


def test_mod_params(geometry):
    assert geometry.PolePairs == 2
    assert geometry.mod_params == {
        "Poles": "2*2",
        "ModelLength": "85mm",
        "SymmetryFactor": "Poles",
        "StatorSkewAngle": "0deg",
    }


def test_oper_params(computation):
    assert computation.oper_params == {
        "Id": "0.0A",
        "Iq": "0.0A",
        "epsI": "atan2(Iq,Id)",
        "Im": "sqrt(Id^2+Iq^2)",
        "InitPos": "-30deg",
        "w": f"{2 * np.pi * 50}Hz",
        "RotSpeed": "1500.0rpm",
        "Nper": "1/6",
        "PointPer": "101",
    }


def test_rot_points(geometry):
    assert len(geometry.rot_points) == 6
    assert all(len(point) == 3 for point in geometry.rot_points)


def test_derived_params(geometry):
    assert geometry.rotor_r_min == pytest.approx(12.5)
    assert geometry.rotor_r_max == pytest.approx(39.5 - 0.225)


def test_solution_expressions_is_a_list_ending_in_torque(computation):
    assert isinstance(computation.solution_expressions, list)
    assert computation.solution_expressions[0] == "Moving1.Position"
    assert computation.solution_expressions[1] == "Moving1.Torque"
    assert computation.solution_expressions[-1] == "L(PhaseC,PhaseC)"
    assert len(computation.solution_expressions) == 20


def test_udp_par_list_stator(geometry):
    keys = [item[0] for item in geometry.udp_par_list_stator]
    assert keys == [
        "DiaGap",
        "DiaYoke",
        "Length",
        "Skew",
        "Slots",
        "SlotType",
        "Hs0",
        "Hs01",
        "Hs1",
        "Hs2",
        "Bs0",
        "Bs1",
        "Bs2",
        "Rs",
        "FilletType",
        "HalfSlot",
        "SegAngle",
        "LenRegion",
        "InfoCore",
    ]


def test_output_vars_is_empty(computation):
    assert computation.output_vars == {}


def test_post_params_plot_names_unique(computation):
    plot_names = list(computation.post_params.values())
    assert len(plot_names) == len(set(plot_names))


def test_mm_to_str(geometry):
    assert geometry.mm_to_str("geom_params", "DiaShaft") == pytest.approx(25.0)


def test_mm_to_str_raises_without_mm_suffix(geometry):
    with pytest.raises(Exception):
        geometry.mm_to_str("geom_params", "SlotNumber")


def test_extract_results_converts_units(computation):
    computation.set_variables(SimpleNamespace(variable_manager={}), Id=1.0, Iq=2.0)

    def data_real(expr):
        letters = ["ABC".index(part[0]) + 1 for part in expr.split("Phase")[1:]]
        value = float(np.prod(letters) if letters else 7.0)
        return np.full(2, value * 1e9 if expr.startswith("L(") else value)

    solutions = SimpleNamespace(data_real=data_real, primary_sweep_values=[0.0, 1.0])
    out = computation.extract_results(solutions)

    theta_el = np.deg2rad(np.full(2, 7.0)-computation.InitPos)*computation.geometry.PolePairs
    current_phases = np.stack([np.full(2, i + 1.0) for i in range(3)], axis=-1)
    expected_I_d, _ = to_dq(current_phases, theta_el, harmonic=1)

    assert out["I_d"] == pytest.approx(expected_I_d / 1e3)
    assert out["Moving1.Torque"] == pytest.approx(np.full(2, 7.0))
