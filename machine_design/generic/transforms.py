"""m-phase Park/Clarke transforms, in Python.

This is the *single* Python implementation of the transform that Ansys applies in
`motors/synrm_5f_40s.py`'s output variables, where the same three lines are written out
once for flux linkage, once for input current and once for terminal voltage. The
convention here matches those expressions exactly, so a quantity transformed here
and the matching `*_d1`/`*_q1`/`*_d3`/`*_q3` output variable agree to solver
tolerance. `tests/test_ansys_transforms.py` asserts that on a live solve.

Phase count and harmonic are both free. `to_dq` reads the phase count off the last
axis of its input, so the same function serves the 3-phase anchor and the 5-phase
one; `to_phases` takes it explicitly, since it has to synthesise that many values.
Caveat for the triplen case: in a 3-phase machine the 3rd harmonic is zero-sequence
(all phases in phase), so a dq3 read of it is not a constant vector -- that is
physics, not a defect of the transform.

Convention (from `motors/synrm_5f_40s.py`, `set_output_vars`)::

    theta_el = RotSign * (Moving1.Position - InitPos) * PolePairs - pi
    cos<k>_<h> = cos( h * (theta_el - 2*pi*k/m))
    sin<k>_<h> = sin(-h * (theta_el - 2*pi*k/m))          <-- note the minus
    X_d<h> = 2/m * sum_k  X_k * cos<k>_<h>
    X_q<h> = 2/m * sum_k  X_k * sin<k>_<h>

(written out for m = 5 in that file; `m` here is the phase count.)

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


def _phase_angles(theta_el, harmonic, n_phases):
    """h * (theta_el - 2*pi*k/m) for k = 0..m-1, shaped (..., m)."""
    theta_el = np.asarray(theta_el, dtype=float)[..., None]
    k = np.arange(n_phases)
    return harmonic * (theta_el - 2.0 * np.pi * k / n_phases)


def to_dq(x_phases, theta_el, harmonic):
    """Per-phase quantity -> (d, q) for the given harmonic.

    `x_phases` carries the phases on the last axis, in winding order (A, B, C, ...);
    the phase count is taken from that axis, so this works for any m. `theta_el` is
    broadcast against the leading axes. Returns two arrays shaped like `theta_el`.
    """
    x_phases = np.asarray(x_phases, dtype=float)
    n_phases = x_phases.shape[-1]
    if n_phases < 3:
        raise ValueError(f"need at least 3 phases on the last axis, got {n_phases}")
    phi = _phase_angles(theta_el, harmonic, n_phases)
    scale = 2.0 / n_phases
    d = scale * np.sum(x_phases * np.cos(phi), axis=-1)
    q = scale * np.sum(x_phases * np.sin(-phi), axis=-1)
    return d, q


def to_phases(d, q, theta_el, harmonic, n_phases):
    """Inverse of `to_dq` for a single harmonic: (d, q) -> `n_phases` phase values.

    Exact inverse for a pure `harmonic` component; summing the results of several
    harmonics reconstructs a multi-harmonic waveform, because the transforms for
    distinct harmonics are orthogonal over m equally spaced phases (see
    `tests/test_transforms.py`). `n_phases` is explicit here because, unlike `to_dq`,
    there is no input array to read it from.
    """
    phi = _phase_angles(theta_el, harmonic, n_phases)
    return np.asarray(d, dtype=float)[..., None] * np.cos(phi) - np.asarray(q, dtype=float)[..., None] * np.sin(phi)
