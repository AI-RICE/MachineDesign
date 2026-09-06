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
| `motors/` | one module per **anchor** (a named machine), each subclassing the base `Geometry`/`Computation` classes |
| `tests/` | the test suite |

### Anchors — machines are named, not numbered

An **anchor** is a named machine definition: everything needed to instantiate and solve one physical machine — class, slots, phases, winding (turns, `Rs`), end-winding `Lew`, **rotor radii**, geometry generator + parameter bounds, FEA template.

The name carries only what the anchor fixes for good — machine class, phases, slot count. Values that are routinely swept (turns, currents, bus voltage) stay out of it, so a name never goes stale.

One anchor per module in `motors/`, named `<class>_<phases>f_<slots>s`:

| module | machine |
|---|---|
| `motors/synrm_3f_36s.py` | 3-phase SynRM, 36 stator slots |
| `motors/synrm_5f_40s.py` | 5-phase SynRM, 40 stator slots, dq1+dq3 excitation |

Each exports `Geometry` and `Computation`, so the module name carries the identity and the class names stay uniform — `from motors.synrm_5f_40s import Geometry, Computation` says which machine you are solving, where `Geometry2` did not.

- **Derive, don't fork.** To study a variant (add magnets, change turns), subclass an existing anchor and override only what differs — do **not** edit an anchor's class in place, so old results stay reproducible. A genuinely new machine is a new anchor module.
- **An anchor must declare all of the above**, rotor radii included. They are the radial window the barrier generators draw into; an anchor that does not set them cannot have a rotor built for it.

## What runs in Ansys, what runs in Python

Ansys computes only what needs the field solution. Everything downstream of it is Python.

| in Ansys (needs the solve) | in Python (algebra on solved quantities) |
|---|---|
| torque waveform, flux linkage, `InducedVoltage`, forces, core loss | Park/Clarke dq1/dq3 (`machine_design/transforms.py`), `Rs*I`, `Lew*dI/dt`, peak search over a waveform, terminal voltage at any speed |

Why:

- **One solve, many operating points.** Flux linkage at a given current is speed independent, so `V = Rs*I + j*w*psi` can be re-evaluated at any electrical speed in Python. An Ansys output variable is frozen at the solved frequency, so computing voltage there costs one solve *per* operating point -- three times the FEA for a three-point profile, N times for a drive cycle.
- **Post-processing scalars stay cheap.** `Lew`, winding turns, the bus voltage and `Rs` are post-processing quantities. Keeping them in Python means a stored pool of solves can be re-scored at a new value with no new FEA.
- **It can be tested.** Ansys expressions have no unit tests, parse differently across versions (#15), and mix units inside one argument (`2*pi*f*Time - 72deg`). The Python transforms are covered by `tests/test_transforms.py`.

**One owner per quantity.** Where both sides *could* compute something, exactly one does, and the code says which. `Lew` in `motors/motor2.py` is the live example: it stays `0` in the model, and the end-winding voltage term belongs to the Python consumer. Setting it in both places double-counts it.

`tests/test_ansys_transforms.py` (marked `ansys`) asserts that the Python transform and the Ansys output variables agree on a live solve, so this boundary is checked rather than merely described.

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

**I want to study a new machine** → add an **anchor** (PR to `main`); if it's a variant, derive it from an existing anchor. Then use it from your experiment.

**I want a new optimization method** → add it to `machine_design/optimization/` against the interface (PR to `main`). Now anyone can use it on any anchor.

**I found a bug in a shared design/method** → fix it on `main` via PR, then rebase your branch. Do not patch it only on your branch.
