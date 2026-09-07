import numpy as np
import pytest

from machine_design.optimization.geometry import analyze_results, rotate, rotation_matrix


def test_rotation_matrix_zero_degrees():
    R = rotation_matrix(0)
    assert R == pytest.approx(np.eye(2))


def test_rotation_matrix_90_degrees():
    R = rotation_matrix(90)
    assert R == pytest.approx(np.array([[0, -1], [1, 0]]), abs=1e-10)


def test_rotation_matrix_rad_flag_matches_deg():
    R_deg = rotation_matrix(180)
    R_rad = rotation_matrix(np.pi, rad=True)
    assert R_deg == pytest.approx(R_rad)


def test_rotate_90_degrees():
    x, y = rotate(1, 0, 90)
    assert x == pytest.approx(0, abs=1e-10)
    assert y == pytest.approx(1)


def test_analyze_results_drops_last_point():
    Tor = np.array([9.0, 10.0, 11.0, 12.0, 999.0])  # last point excluded from the average
    TorAvg, TorRmsAC, TorRippleRms = analyze_results(Tor)
    assert TorAvg == pytest.approx(10.5)
    assert TorRmsAC == pytest.approx(1.1180339887498949)
    assert TorRippleRms == pytest.approx(10.647942749998999)
