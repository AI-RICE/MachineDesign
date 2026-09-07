import numpy as np
import pytest

from machine_design.designs.design import Design
from machine_design.optimization.generators import HacklGenerator_SixLambdas, check_barrier, get_arc, signed_distance
from motors.synrm_3f_36s import Computation, Geometry

r_stator_end = 0.7
offset = 0.35

# design at Pareto front
x_best_six_lambdas = np.array([8.95052421, 23.01115583, 33.03271713, 14.34330278, 27.92171436, 37.3, 0.21260669, 0.31, 0.41, 0.3, 0.4, 0.46])


def test_get_arc_endpoints():
    arc = get_arc(R=2.0, start_deg=0, end_deg=90, n_points=3)
    assert arc.shape == (3, 2)
    assert arc[0] == pytest.approx([2.0, 0.0], abs=1e-10)
    assert arc[-1] == pytest.approx([0.0, 2.0], abs=1e-10)


def test_signed_distance():
    P = np.array([[0.0, 0.0], [1.0, 1.0]])
    dist = signed_distance(P, a=1, b=1, c=-1)
    assert dist == pytest.approx([-1.0, 1.0])


def test_check_barrier_closed_does_not_raise():
    barrier = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    check_barrier(barrier)


def test_check_barrier_open_raises():
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
    params = six_lambdas_generator.X_to_params(x_best_six_lambdas)
    six_lambdas_generator.set_parameters(params)
    return six_lambdas_generator


@pytest.fixture
def split_barriers(configured_generator):
    barriers = configured_generator.generate_barriers()
    return configured_generator.split_barriers(barriers)


def test_X_to_params_splits_into_four_groups_of_three(six_lambdas_generator):
    phis_inner, phis_outer, lam_inner, lam_outer = six_lambdas_generator.X_to_params(x_best_six_lambdas)
    assert list(phis_inner) == pytest.approx([8.95052421, 23.01115583, 33.03271713])
    assert list(phis_outer) == pytest.approx([14.34330278, 27.92171436, 37.3])
    assert list(lam_inner) == pytest.approx([0.21260669, 0.31, 0.41])
    assert list(lam_outer) == pytest.approx([0.3, 0.4, 0.46])


def test_set_parameters_stores_values(configured_generator):
    assert configured_generator.phis_inner == pytest.approx([8.95052421, 23.01115583, 33.03271713])
    assert configured_generator.lam_outer == pytest.approx([0.3, 0.4, 0.46])


def test_generate_barriers_returns_one_barrier_per_pole(configured_generator):
    barriers = configured_generator.generate_barriers()
    assert len(barriers) == 3


def test_split_barriers_closes_each_barrier(split_barriers):
    for barrier in split_barriers:
        assert barrier[0] == pytest.approx(barrier[-1])


def test_feasible_barriers_true_for_known_good_design(configured_generator, split_barriers):
    assert configured_generator.feasible_barriers(split_barriers) is True