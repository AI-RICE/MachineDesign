import numpy as np
import pytest

from machine_design.designs.design import Design
from machine_design.optimization.generators import (
    HacklGenerator_3BrokenLines,
    HacklGenerator_OneLambda,
    HacklGenerator_SixLambdas,
    check_barrier,
    get_arc,
    signed_distance,
)
from motors.synrm_3f_36s import Computation, Geometry

r_stator_end = 0.7
offset = 0.35

# feasible design
known_good_designs = {
    "OneLambda": (HacklGenerator_OneLambda, np.array([7.1, 20.4, 33.7, 13.7, 27.1, 40.65, 0.35])),
    "SixLambdas": (
        HacklGenerator_SixLambdas,
        np.array([8.95052421, 23.01115583, 33.03271713, 14.34330278, 27.92171436, 37.3, 0.21260669, 0.31, 0.41, 0.3, 0.4, 0.46]),
    ),
    "3BrokenLines": (
        HacklGenerator_3BrokenLines,
        np.array([6.0538801, 23.5, 35.74582285, 10.6, 28.5920024, 40.7191182, 0.36646249, 22.65142559, 30.7050295, 38.0, 9.49161503, 2.0, 1.5]),
    ),
}


def test_get_arc_endpoints():
    arc = get_arc(R=2.0, start_deg=0, end_deg=90, n_points=3)
    assert arc.shape == (3, 2)
    assert arc[0] == pytest.approx([2.0, 0.0], abs=1e-10)
    assert arc[-1] == pytest.approx([0.0, 2.0], abs=1e-10)


def test_signed_distance():
    P = np.array([[0.0, 0.0], [1.0, 1.0]])
    dist = signed_distance(P, a=1, b=1, c=-1)
    assert dist == pytest.approx([-1.0, 1.0])


def test_check_barrier_closed():
    # closed, should not raise
    barrier = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    check_barrier(barrier)


def test_check_barrier_open():
    # open, should raise
    barrier = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    with pytest.raises(ValueError):
        check_barrier(barrier)


@pytest.fixture
def six_lambdas_generator():
    geometry = Geometry()
    dummy_design = Design(m2d=None, geometry=geometry, computation=Computation(geometry))
    return HacklGenerator_SixLambdas(dummy_design, r_stator_end, offset=offset)


@pytest.fixture
def configured_generator(six_lambdas_generator):
    x_best = known_good_designs["SixLambdas"][1]
    params = six_lambdas_generator.X_to_params(x_best)
    six_lambdas_generator.set_parameters(params)
    return six_lambdas_generator


def test_x_to_params_splits_groups(six_lambdas_generator):
    # 4 groups of 3
    x_best = known_good_designs["SixLambdas"][1]
    phis_inner, phis_outer, lam_inner, lam_outer = six_lambdas_generator.X_to_params(x_best)
    assert list(phis_inner) == pytest.approx([8.95052421, 23.01115583, 33.03271713])
    assert list(phis_outer) == pytest.approx([14.34330278, 27.92171436, 37.3])
    assert list(lam_inner) == pytest.approx([0.21260669, 0.31, 0.41])
    assert list(lam_outer) == pytest.approx([0.3, 0.4, 0.46])


def test_set_parameters_stores_values(configured_generator):
    assert configured_generator.phis_inner == pytest.approx([8.95052421, 23.01115583, 33.03271713])
    assert configured_generator.lam_outer == pytest.approx([0.3, 0.4, 0.46])


@pytest.fixture(params=list(known_good_designs))
def any_configured_generator(request):
    generator_cls, x_best = known_good_designs[request.param]
    geometry = Geometry()
    generator = generator_cls(Design(m2d=None, geometry=geometry, computation=Computation(geometry)), r_stator_end, offset=offset)
    generator.set_parameters(generator.X_to_params(x_best))
    return generator


def test_barrier_count(any_configured_generator):
    # one barrier per pole pair
    barriers = any_configured_generator.generate_barriers()
    assert len(barriers) == 3


def test_split_barriers_closed(any_configured_generator):
    # each barrier should be closed
    barriers = any_configured_generator.generate_barriers()
    barriers = any_configured_generator.split_barriers(barriers)
    for barrier in barriers:
        assert barrier[0] == pytest.approx(barrier[-1])


def test_known_good_design_is_feasible(any_configured_generator):
    barriers = any_configured_generator.generate_barriers()
    barriers = any_configured_generator.split_barriers(barriers)
    assert any_configured_generator.feasible_barriers(barriers) is True
