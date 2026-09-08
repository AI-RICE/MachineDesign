"""Pins the m-phase dq convention used by `machine_design.transforms`.

These need no Ansys. They exist because the same transform is written out by hand in
`motors/synrm_5f_40s.py`'s output variables, and a silent disagreement between the two
would corrupt every voltage and flux number downstream without failing loudly. The live
cross-check against Ansys is `tests/test_ansys_transforms.py`.

Parametrised over phase count as well as harmonic, so the 3-phase anchor is covered by
the same transform as the 5-phase one.
"""

import numpy as np
import pytest

from machine_design.transforms import electrical_angle, to_dq, to_phases

THETA = np.linspace(0.0, 2.0 * np.pi, 37)

# (n_phases, harmonic). A harmonic that is a multiple of the phase count is
# zero-sequence and not invertible -- see test_triplen_harmonic_is_zero_sequence.
CASES = [(3, 1), (5, 1), (5, 3)]
CASE_IDS = [f"{m}ph-h{h}" for m, h in CASES]


@pytest.mark.parametrize(("n_phases", "harmonic"), CASES, ids=CASE_IDS)
@pytest.mark.parametrize(("d", "q"), [(1.0, 0.0), (0.0, 1.0), (0.7, -0.3), (-1.2, 2.5)])
def test_round_trip_recovers_dq(n_phases, harmonic, d, q):
    x = to_phases(d, q, THETA, harmonic, n_phases)
    assert x.shape == (len(THETA), n_phases)
    d_out, q_out = to_dq(x, THETA, harmonic)
    np.testing.assert_allclose(d_out, d, atol=1e-12)
    np.testing.assert_allclose(q_out, q, atol=1e-12)


@pytest.mark.parametrize(("h_signal", "h_read"), [(1, 3), (3, 1)])
def test_harmonics_are_orthogonal(h_signal, h_read):
    """A pure dq1 waveform must read as zero in dq3, and vice versa.

    This is what makes the two current subspaces independent design variables; if it
    broke, dq3 injection would leak into the dq1 measurement.
    """
    x = to_phases(1.3, -0.8, THETA, h_signal, 5)
    d, q = to_dq(x, THETA, h_read)
    np.testing.assert_allclose(d, 0.0, atol=1e-12)
    np.testing.assert_allclose(q, 0.0, atol=1e-12)


def test_superposed_harmonics_separate():
    """dq1 and dq3 components ride on the same phase waveform without interfering."""
    x = to_phases(2.0, 0.5, THETA, 1, 5) + to_phases(-0.4, 0.9, THETA, 3, 5)
    d1, q1 = to_dq(x, THETA, 1)
    d3, q3 = to_dq(x, THETA, 3)
    np.testing.assert_allclose(d1, 2.0, atol=1e-12)
    np.testing.assert_allclose(q1, 0.5, atol=1e-12)
    np.testing.assert_allclose(d3, -0.4, atol=1e-12)
    np.testing.assert_allclose(q3, 0.9, atol=1e-12)


def test_triplen_harmonic_is_zero_sequence_in_three_phase():
    """m = 3, h = 3 is deliberately NOT invertible, and that is physics.

    3*(2*pi*k/3) is a whole number of turns, so every phase sees the same angle: the
    triplen harmonic is zero-sequence. Pinned so nobody "fixes" the transform to make
    this round-trip.
    """
    x = to_phases(1.0, 0.0, THETA, 3, 3)
    np.testing.assert_allclose(x[:, 0], x[:, 1], atol=1e-12)
    np.testing.assert_allclose(x[:, 0], x[:, 2], atol=1e-12)

    d, _ = to_dq(x, THETA, 3)
    assert not np.allclose(d, 1.0, atol=1e-6), "expected the triplen round-trip to fail"


@pytest.mark.parametrize("n_phases", [3, 5])
def test_amplitude_scaling_is_two_over_m(n_phases):
    """The 2/m factor: a unit-amplitude d-axis waveform must give d = 1."""
    d, _ = to_dq(to_phases(1.0, 0.0, THETA, 1, n_phases), THETA, 1)
    np.testing.assert_allclose(d, 1.0, atol=1e-12)


@pytest.mark.parametrize("n_phases", [3, 5])
def test_q_axis_sign_convention(n_phases):
    """`q` is the NEGATIVE sine sum, matching sin(-h*(theta - 2*pi*k/m)) in Ansys.

    Built from explicit cos/sin phase quantities rather than via `to_phases`, so this
    fails if the sign in `to_dq` is ever "corrected" to the more usual convention.
    """
    theta = 0.37
    phi = theta - 2.0 * np.pi * np.arange(n_phases) / n_phases

    # A pure +cos(phi) phase quantity is pure d-axis, unit amplitude.
    d, q = to_dq(np.cos(phi), theta, 1)
    assert d == pytest.approx(1.0, abs=1e-12)
    assert q == pytest.approx(0.0, abs=1e-12)

    # A pure +sin(phi) phase quantity gives q = -1 under this convention, not +1.
    d, q = to_dq(np.sin(phi), theta, 1)
    assert d == pytest.approx(0.0, abs=1e-12)
    assert q == pytest.approx(-1.0, abs=1e-12)


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


def test_rejects_too_few_phases():
    with pytest.raises(ValueError, match="at least 3 phases"):
        to_dq(np.zeros((len(THETA), 2)), THETA, 1)
