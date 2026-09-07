import numpy as np
import pytest

from machine_design.optimization.generators import check_barrier, get_arc, signed_distance


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
