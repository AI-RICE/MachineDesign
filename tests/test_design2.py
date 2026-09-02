import pytest

from motors.motor1 import Geometry
from motors.motor2 import Computation2, Geometry2


@pytest.fixture
def base_geometry():
    return Geometry()


@pytest.fixture
def geometry2():
    return Geometry2()


@pytest.fixture
def computation2(geometry2):
    return Computation2(geometry2)


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
        "f": "50Hz",
        "RotSpeed": "1500.0rpm",
        "Nper": "1/10",
        "PointPer": "101",
    }


def test_derived_params_are_not_set(geometry2):
    # Geometry2.set_derived_params() is a no-op, unlike Geometry's.
    assert not hasattr(geometry2, "rotor_r_min")
    assert not hasattr(geometry2, "rotor_r_max")


def test_solution_expressions_is_a_list_ending_in_torque(computation2):
    assert isinstance(computation2.solution_expressions, list)
    assert computation2.solution_expressions[-1] == "Moving1.Torque"
    assert len(computation2.solution_expressions) == 23


def test_post_params_plot_names_unique(computation2):
    plot_names = list(computation2.post_params.values())
    assert len(plot_names) == len(set(plot_names))


def test_output_vars_known_values(computation2):
    assert computation2.output_vars["Rstat"] == "19"
    assert computation2.output_vars["Lew"] == "0"
