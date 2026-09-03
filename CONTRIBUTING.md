# Contributing to MachineDesign

A short guide so several people can work on this repo without stepping on each other.

## The one rule

> **Shared code lives on `main` and is changed only by a small PR to `main`.
> Your research branch adds and edits only `experiments/<your-project>/`.**

If you find yourself editing `machine_design/`on your own branch, stop — that's a change everyone needs, so it's a PR to `main`. Then rebase your branch on `main` and everyone gets the fix once.

## Repository layout

| directory | contains |
|---|---|
| `machine_design/designs/` | the shared `Design`, `GeometryBase`, `ComputationBase` classes and `load_design()`, one FEA solve for a given machine |
| `machine_design/optimization/` | design agnostic optimizers and geometry generators |
| `machine_design/parallel_calculation/` | parallel runners for sweeping many solves at once |
| `experiments/<name>/` | your run configs, the profile/objective you're studying, analysis notebooks, result artifacts, write-ups |
| `motors/` | concrete machine definitions (`motor1.py`, `motor2.py`), each subclassing the base `Geometry`/`Computation` classes |
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

## Running things

All compute goes through the SLURM queue on `bayes` — see [`bayes-queue/README.md`](bayes-queue/README.md). One FEA solve = 1 core + its own `--mem` + one license; submit sweeps as job arrays (`bayes-queue/sweep.sbatch`).

## Quick checklists

**I want to optimize an existing machine with an existing method** → in `experiments/<name>/`, write a config that picks an anchor + a method + your objective/constraints, and an sbatch to run it. Touch nothing shared.

**I want to study a new machine** → add an **motors** (PR to `main`); if it's a variant, derive it from an existing anchor. Then use it from your experiment.

**I want a new optimization method** → add it to `machine_design/optimization/` against the interface (PR to `main`). Now anyone can use it on any anchor.

**I found a bug in a shared design/method** → fix it on `main` via PR, then rebase your branch. Do not patch it only on your branch.
