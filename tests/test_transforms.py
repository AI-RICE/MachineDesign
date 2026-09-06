"""Pins the five-phase dq1/dq3 convention used by `machine_design.transforms`.

These need no Ansys. They exist because the same transform is written out by hand
in `motors/synrm_5f_40s.py`'s output variables, and a silent disagreement between the
two would corrupt every voltage and flux number downstream without failing loudly.
The live cross-check against Ansys is `tests/test_ansys_transforms.py`.
"""

import numpy as np
import pytest

from machine_design.transforms import N_PHASES, electrical_angle, to_dq, to_phases

THETA = np.linspace(0.0, 2.0 * np.pi, 37)


def _phase_waveform(theta, harmonic, d, q):
    """Build a per-phase waveform carrying exactly (d, q) at `harmonic`."""
    return to_phases(d, q, theta, harmonic)


@pytest.mark.parametrize("harmonic", [1, 3])
@pytest.mark.parametrize(("d", "q"), [(1.0, 0.0), (0.0, 1.0), (0.7, -0.3), (-1.2, 2.5)])
def test_round_trip_recovers_dq(harmonic, d, q):
    x = _phase_waveform(THETA, harmonic, d, q)
    d_out, q_out = to_dq(x, THETA, harmonic)
    np.testing.assert_allclose(d_out, d, atol=1e-12)
    np.testing.assert_allclose(q_out, q, atol=1e-12)


@pytest.mark.parametrize(("h_signal", "h_read"), [(1, 3), (3, 1)])
def test_harmonics_are_orthogonal(h_signal, h_read):
    """A pure dq1 waveform must read as zero in dq3, and vice versa.

    This is what makes the two current subspaces independent design variables; if
    it broke, dq3 injection would leak into the dq1 measurement.
    """
    x = _phase_waveform(THETA, h_signal, 1.3, -0.8)
    d, q = to_dq(x, THETA, h_read)
    np.testing.assert_allclose(d, 0.0, atol=1e-12)
    np.testing.assert_allclose(q, 0.0, atol=1e-12)


def test_superposed_harmonics_separate():
    """dq1 and dq3 components ride on the same phase waveform without interfering."""
    x = _phase_waveform(THETA, 1, 2.0, 0.5) + _phase_waveform(THETA, 3, -0.4, 0.9)
    d1, q1 = to_dq(x, THETA, 1)
    d3, q3 = to_dq(x, THETA, 3)
    np.testing.assert_allclose([d1, q1], [np.full_like(THETA, 2.0), np.full_like(THETA, 0.5)], atol=1e-12)
    np.testing.assert_allclose([d3, q3], [np.full_like(THETA, -0.4), np.full_like(THETA, 0.9)], atol=1e-12)


def test_q_axis_sign_convention():
    """`q` is the NEGATIVE sine sum, matching sin(-h*(theta - 2*pi*k/5)) in Ansys.

    Built from explicit cos/sin phase quantities rather than via `to_phases`, so this
    fails if the sign in `to_dq` is ever "corrected" to the more usual convention.
    """
    theta = 0.37
    k = np.arange(N_PHASES)
    phi = theta - 2.0 * np.pi * k / N_PHASES

    # A pure +cos(phi) phase quantity is pure d-axis, unit amplitude.
    d, q = to_dq(np.cos(phi), theta, 1)
    assert d == pytest.approx(1.0, abs=1e-12)
    assert q == pytest.approx(0.0, abs=1e-12)

    # A pure +sin(phi) phase quantity gives q = -1 under this convention, not +1.
    d, q = to_dq(np.sin(phi), theta, 1)
    assert d == pytest.approx(0.0, abs=1e-12)
    assert q == pytest.approx(-1.0, abs=1e-12)


def test_amplitude_scaling_is_two_fifths():
    """The 2/5 factor: a unit-amplitude d-axis waveform gives d = 1, not 5/2 or 1/5."""
    d, _ = to_dq(_phase_waveform(THETA, 1, 1.0, 0.0), THETA, 1)
    np.testing.assert_allclose(d, 1.0, atol=1e-12)


def test_electrical_angle_matches_ansys_expression():
    """theta_el = RotSign*(Position - InitPos)*PolePairs - pi, in radians."""
    position = np.array([0.0, 0.25, 0.5])
    init_pos = np.deg2rad(-45.0)
    got = electrical_angle(position, init_pos, pole_pairs=2, rot_sign=1.0)
    expected = 1.0 * (position - init_pos) * 2 - np.pi
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_electrical_angle_accepts_degrees_like_ansys_reports_them():
    """Ansys reports `Moving1.Position` in degrees; `degrees=True` must match radians.

    Guards the trap this API invites: the natural input comes back from Ansys in
    degrees, and passing it as radians fails silently rather than loudly.
    """
    position_deg, init_deg = np.array([315.0, 316.8, 318.6]), -45.0
    from_deg = electrical_angle(position_deg, init_deg, pole_pairs=2, degrees=True)
    from_rad = electrical_angle(np.deg2rad(position_deg), np.deg2rad(init_deg), pole_pairs=2)
    np.testing.assert_allclose(from_deg, from_rad, atol=1e-12)

    # The observed Ansys value: (315 - -45) * 2 - 180 degrees = 540 deg = 3*pi rad.
    assert from_deg[0] == pytest.approx(3.0 * np.pi, abs=1e-12)


def test_rejects_wrong_phase_count():
    with pytest.raises(ValueError, match="5 phases"):
        to_dq(np.zeros((len(THETA), 3)), THETA, 1)
