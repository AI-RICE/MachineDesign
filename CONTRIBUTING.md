# Contributing to MachineDesign

A short guide so several people can work on this repo without stepping on each other.

## Repository layout

| directory | contains |
|---|---|
| `machine_design/designs/` | the shared `Design`, `GeometryBase`, `ComputationBase` classes and `load_design()`, one FEA solve for a given machine |
| `machine_design/optimization/` | design agnostic optimizers and geometry generators |
| `machine_design/parallel_calculation/` | parallel runners for sweeping many solves at once |
| `motors/` | concrete machine definitions (`motor1.py`, `motor2.py`), each subclassing the base `Geometry`/`Computation` classes |
| `notebooks/` | entry point scripts (`run.py`, `run_optimization.py`) and analysis notebooks |
| `tests/` | the test suite |

Add a new machine by subclassing an existing `motors/motor*.py` file instead of editing an existing class in place, so old results stay reproducible.

## Pull requests

- Shared code changes go through a branch and a PR, not a direct push to `main`.
- Create the branch from the GitHub issue it belongs to ("Create a branch" in the issue's Development panel), so it is linked automatically.
- Commit, push, and open the PR. Add a short description of what changed and why, and include `Closes #<issue>`.
- Make sure CI (Ruff and pytest) is green.
- Do not merge your own PR. Wait for review.

## Running tests

Point at the Ansys install first, since the version and path differ per machine:

```bash
export ANSYSEM_ROOT241=/data/AnsysEM/v241/Linux64
```

Most tests do not need Ansys and run with:

```bash
python -m pytest
```

A few tests are marked `ansys` and need a running Ansys Electronics Desktop session with a free license seat, so they are skipped by default. Run them with:

```bash
python -m pytest -m "" -v
```

`-m ""` clears the default marker filter so the `ansys` tests are included too.

## Running larger jobs

All compute can also go through the SLURM queue on `bayes`, see [`bayes-queue/README.md`](bayes-queue/README.md).
