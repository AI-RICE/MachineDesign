import re

import pytest

from motors.motor1 import Computation, Geometry


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
        "Im": "1.5*sqrt(2)A",
        "epsI": "pi/4",
        "InitPos": "-30deg",
        "f": "50Hz",
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


def test_solution_expressions(computation):
    assert computation.solution_expressions == "Moving1.Torque"


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


def test_output_vars_only_reference_earlier_keys(computation):
    # Ansys evaluates output variables in insertion order, so a formula referencing
    # another output variable must not reference one that is defined later.
    defined = set()
    all_keys = set(computation.output_vars.keys())
    for key, formula in computation.output_vars.items():
        later_keys = all_keys - defined - {key}
        referenced_later = [k for k in later_keys if re.search(rf"\b{re.escape(k)}\b", formula)]
        assert not referenced_later, f"'{key}' formula references not-yet-defined {referenced_later}"
        defined.add(key)


def test_output_vars_known_formula(computation):
    assert computation.output_vars["Irms"] == "sqrt(I_d^2+I_q^2)/sqrt(2)"


def test_post_params_plot_names_unique(computation):
    plot_names = list(computation.post_params.values())
    assert len(plot_names) == len(set(plot_names))


def test_mm_to_str(geometry):
    assert geometry.mm_to_str("geom_params", "DiaShaft") == pytest.approx(25.0)


def test_mm_to_str_raises_without_mm_suffix(geometry):
    with pytest.raises(Exception):
        geometry.mm_to_str("geom_params", "SlotNumber")
