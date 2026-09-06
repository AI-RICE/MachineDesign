"""Live check that the Python dq transform agrees with the Ansys output variables.

`machine_design/transforms.py` and `motors/synrm_5f_40s.py`'s `set_output_vars` implement the
same five-phase Park/Clarke transform twice -- once in Python, once as Ansys expressions.
This solves one rotor with both dq1 and dq3 excited, then asserts that transforming the
per-phase quantities in Python reproduces Ansys's own dq outputs.

Two subjects, deliberately:

* **flux linkage** -- `FluxLinkage(PhaseA..E)` vs `Flux_d1/q1/d3/q3`. Flux stays in Ansys
  (it needs the field solution), so this comparison is permanent: it is the standing guard
  on the boundary described in "What runs in Ansys, what runs in Python" (CONTRIBUTING.md).
* **terminal voltage** -- `V_A..V_E` vs `V_d1/q1/d3/q3`. This one is a *one-shot*
  equivalence proof. The voltage transforms are slated to move to Python, which deletes
  both sides of the comparison, so it can only be run while they still exist. Once the
  Ansys-side voltage variables go, drop this half and keep the flux half.

Marked `ansys`, so it is skipped by default. Run with:

    python -m pytest -m "" tests/test_ansys_transforms.py -v
"""

import numpy as np
import pytest

from machine_design.config import load_config
from machine_design.designs.design import Design
from machine_design.optimization.generators import HacklGenerator_OneLambda
from machine_design.transforms import to_dq
from motors.synrm_5f_40s import Computation, Geometry

pytestmark = pytest.mark.ansys

AEDT_VERSION = load_config()["aedt_version"]
R_STATOR_END = 0.7
OFFSET = 0.35
SEED = 42
NUM_CORES = 1

# Both subspaces excited, so the h = 3 comparison is not a check against zero.
# Order is (Id1, Iq1, Id3, Iq3) -- compute() forwards these to set_variables().
CURRENTS = (0.8, 0.9, 0.2, -0.15)

PHASES = ["A", "B", "C", "D", "E"]
PHASE_FLUX = [f"FluxLinkage(Phase{p})" for p in PHASES]
PHASE_VOLTAGE = [f"V_{p}" for p in PHASES]

# (label, per-phase expressions, ansys dq prefix)
SUBJECTS = [("flux", PHASE_FLUX, "Flux"), ("voltage", PHASE_VOLTAGE, "V")]


class ComputationWithPhaseQuantities(Computation):
    """`Computation`, additionally extracting the per-phase quantities and theta_el.

    The shipped class extracts only the dq quantities, since that is all the optimizers
    need. Subclassed rather than edited so the shared anchor is untouched.
    """

    def set_solution_expressions(self):
        super().set_solution_expressions()
        dq = [f"{prefix}_{axis}{h}" for prefix in ("Flux", "V") for h in (1, 3) for axis in "dq"]
        extra = [*PHASE_FLUX, *PHASE_VOLTAGE, "theta_el", *dq]
        self.solution_expressions = [*self.solution_expressions, *extra]


def _feasible_barriers(geometry):
    dummy = Design(m2d=None, geometry=geometry, computation=Computation(geometry))
    generator = HacklGenerator_OneLambda(dummy, R_STATOR_END, offset=OFFSET)

    np.random.seed(SEED)
    while True:
        generator.set_parameters(generator.random_parameters())
        barriers = generator.split_barriers(generator.generate_barriers())
        if generator.feasible_barriers(barriers):
            return barriers


@pytest.fixture(scope="module")
def solved(tmp_path_factory):
    """One solve, shared by both subjects -- an FEA solve is far too costly to repeat."""
    geometry = Geometry()
    computation = ComputationWithPhaseQuantities(geometry)
    barriers = _feasible_barriers(geometry)

    design = None
    try:
        design = Design.create(
            "TransformBoundary",
            "Design01",
            str(tmp_path_factory.mktemp("aedt") / "transform.aedt"),
            geometry,
            computation,
            version=AEDT_VERSION,
            non_graphical=True,
            new_desktop=False,
            close_on_exit=False,
        )
        design.add_rotor()
        for barrier in barriers:
            design.add_rotor_barrier(barrier)
        return design.compute(*CURRENTS, NUM_CORES=NUM_CORES)
    finally:
        if design is not None:
            design.close_project()


@pytest.mark.parametrize(("label", "phase_exprs", "prefix"), SUBJECTS, ids=[s[0] for s in SUBJECTS])
@pytest.mark.parametrize("harmonic", [1, 3])
def test_python_transform_matches_ansys(solved, label, phase_exprs, prefix, harmonic):
    assert solved is not None, "solve returned no results"
    needed = [*phase_exprs, "theta_el", f"{prefix}_d{harmonic}", f"{prefix}_q{harmonic}"]
    missing = [k for k in needed if k not in solved]
    assert not missing, f"missing from the extracted solution: {missing}"

    # (n_timesteps, 5), phases in order A..E -- the order transforms.to_dq expects.
    per_phase = np.column_stack([np.asarray(solved[k], dtype=float) for k in phase_exprs])
    # Ansys returns theta_el in DEGREES (its "- pi" evaluates as -180 in an angular
    # context); transforms.to_dq takes radians.
    theta_el = np.deg2rad(np.asarray(solved["theta_el"], dtype=float))

    d_py, q_py = to_dq(per_phase, theta_el, harmonic)
    d_aedt = np.asarray(solved[f"{prefix}_d{harmonic}"], dtype=float)
    q_aedt = np.asarray(solved[f"{prefix}_q{harmonic}"], dtype=float)

    # rtol handles solver/report round-off; atol keeps near-zero samples from failing on
    # relative error alone, scaled to the signal rather than to absolute units.
    atol = 1e-6 * max(np.max(np.abs(d_aedt)), np.max(np.abs(q_aedt)), 1.0)
    np.testing.assert_allclose(d_py, d_aedt, rtol=1e-6, atol=atol, err_msg=f"{label} d{harmonic} mismatch")
    np.testing.assert_allclose(q_py, q_aedt, rtol=1e-6, atol=atol, err_msg=f"{label} q{harmonic} mismatch")
