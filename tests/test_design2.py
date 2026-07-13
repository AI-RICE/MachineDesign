import pytest

from machine_design.design import Design
from machine_design.design_computation import Computation, Computation2
from machine_design.design_geometry import Geometry, Geometry2


@pytest.fixture
def base():
    geometry = Geometry()
    computation = Computation(geometry)
    return Design(m2d=None, geometry=geometry, computation=computation)


@pytest.fixture
def design2():
    geometry = Geometry2()
    computation = Computation2(geometry)
    return Design(m2d=None, geometry=geometry, computation=computation)


def test_geom_params_overrides_slot_number_only(base, design2):
    expected = dict(base.geom_params)
    expected["SlotNumber"] = "40"
    assert design2.geom_params == expected


def test_slot_params_overrides_bs1_bs2_setangle_only(base, design2):
    expected = dict(base.slot_params)
    expected["Bs1"] = "3.0mm"
    expected["Bs2"] = "4.3mm"
    expected["SetAngle"] = "9deg"
    assert design2.slot_params == expected


def test_wind_params_overrides_nc_only(base, design2):
    expected = dict(base.wind_params)
    expected["Nc"] = "113"
    assert design2.wind_params == expected


def test_oper_params_fully_replaced(design2):
    assert design2.oper_params == {
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


def test_derived_params_are_not_set(design2):
    # Geometry2.set_derived_params() is a no-op, unlike Geometry's.
    assert not hasattr(design2, "rotor_r_min")
    assert not hasattr(design2, "rotor_r_max")


def test_solution_expressions_is_a_list_ending_in_torque(design2):
    assert isinstance(design2.solution_expressions, list)
    assert design2.solution_expressions[-1] == "Moving1.Torque"
    assert len(design2.solution_expressions) == 23


def test_post_params_plot_names_unique(design2):
    plot_names = list(design2.post_params.values())
    assert len(plot_names) == len(set(plot_names))


def test_output_vars_known_values(design2):
    assert design2.output_vars["Rstat"] == "19"
    assert design2.output_vars["Lew"] == "0"
