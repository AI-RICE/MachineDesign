"""Five-phase Park/Clarke transforms (dq1 / dq3), in Python.

This is the *single* Python implementation of the transform that Ansys applies in
`motors/synrm_5f_40s.py`'s output variables, where the same three lines are written out
once for flux linkage, once for input current and once for terminal voltage. The
convention here matches those expressions exactly, so a quantity transformed here
and the matching `*_d1`/`*_q1`/`*_d3`/`*_q3` output variable agree to solver
tolerance. `tests/test_ansys_transforms.py` asserts that on a live solve.

Convention (from `motors/synrm_5f_40s.py`, `set_output_vars`)::

    theta_el = RotSign * (Moving1.Position - InitPos) * PolePairs - pi
    cos<k>_<h> = cos( h * (theta_el - 2*pi*k/5))
    sin<k>_<h> = sin(-h * (theta_el - 2*pi*k/5))          <-- note the minus
    X_d<h> = 2/5 * sum_k  X_k * cos<k>_<h>
    X_q<h> = 2/5 * sum_k  X_k * sin<k>_<h>

**Units.** ``theta_el`` here is in RADIANS. Ansys's ``theta_el`` output variable comes
back in DEGREES -- the ``- pi`` in its expression is evaluated as -180 in an angular
context -- so convert with ``np.deg2rad`` before passing it in. Verified on a live
2024.2 solve: with that conversion the transform below reproduces Ansys's own
``Flux_d1/q1/d3/q3`` to 2e-15 (`tests/test_ansys_transforms.py`).

The minus inside the sine means ``q`` is the *negative* of the usual sine sum.
That is a deliberate quirk of this model, not a typo; `tests/test_transforms.py`
pins it so it cannot drift.

Why this lives in Python rather than as more Ansys output variables: see
"What runs in Ansys, what runs in Python" in `CONTRIBUTING.md`. In short, the
transform needs no field solution -- it is algebra on quantities the solve has
already produced -- and keeping it here makes it testable and lets one solve be
re-evaluated at any electrical speed.
"""

import numpy as np

N_PHASES = 5


def electrical_angle(position, init_pos, pole_pairs, rot_sign=1.0, degrees=False):
    """theta_el in RADIANS, matching Ansys's ``theta_el`` output variable.

    `position` and `init_pos` are mechanical angles -- radians by default, or degrees
    with ``degrees=True``. Pass ``degrees=True`` for values read back from Ansys:
    ``Moving1.Position`` is reported in degrees, so feeding it in as radians silently
    produces a meaningless angle. The return value is always radians, which is what
    `to_dq` and `to_phases` expect.
    """
    position = np.asarray(position, dtype=float)
    if degrees:
        position, init_pos = np.deg2rad(position), np.deg2rad(init_pos)
    return rot_sign * (position - init_pos) * pole_pairs - np.pi


def _phase_angles(theta_el, harmonic):
    """h * (theta_el - 2*pi*k/5) for k = 0..4, shaped (..., 5)."""
    theta_el = np.asarray(theta_el, dtype=float)[..., None]
    k = np.arange(N_PHASES)
    return harmonic * (theta_el - 2.0 * np.pi * k / N_PHASES)


def to_dq(x_phases, theta_el, harmonic):
    """Per-phase quantity -> (d, q) for the given harmonic.

    `x_phases` has the five phases on the last axis, in order A..E; `theta_el` is
    broadcast against the leading axes. Returns two arrays shaped like `theta_el`.
    """
    x_phases = np.asarray(x_phases, dtype=float)
    if x_phases.shape[-1] != N_PHASES:
        raise ValueError(f"expected {N_PHASES} phases on the last axis, got {x_phases.shape[-1]}")
    phi = _phase_angles(theta_el, harmonic)
    scale = 2.0 / N_PHASES
    d = scale * np.sum(x_phases * np.cos(phi), axis=-1)
    q = scale * np.sum(x_phases * np.sin(-phi), axis=-1)
    return d, q


def to_phases(d, q, theta_el, harmonic):
    """Inverse of `to_dq` for a single harmonic: (d, q) -> five phase values.

    Exact inverse for a pure `harmonic` component; summing the results of several
    harmonics reconstructs a multi-harmonic waveform, because the five-phase
    transforms for h = 1 and h = 3 are orthogonal (see `tests/test_transforms.py`).
    """
    phi = _phase_angles(theta_el, harmonic)
    return np.asarray(d, dtype=float)[..., None] * np.cos(phi) - np.asarray(q, dtype=float)[..., None] * np.sin(phi)
