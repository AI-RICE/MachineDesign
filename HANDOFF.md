# Handoff: new-parameterization track (`newparam` branch)

You (Claude) are in a git worktree at
`/Users/smidl/zcu/PFN4BOrevisited/applications/ReluctanceDrive/MachineDesign-newparam/`,
on branch `newparam`. The sibling worktree at `../MachineDesign/` continues the
PFN/Gibbs-prior work in parallel — **don't touch it from here**; do not
cross-pollinate experiments or share checkpoint files unless the user asks.

## Goal of this track

Develop a **new SynRM rotor parameterisation** that **replaces** the three
existing Hackl-style parameterisations (`OneLambda` D=7, `SixLambdas` D=12,
`ThreeBrokenLines` D=13). One unified geometric description rather than three.
The user has motivation for this but **hasn't briefed it yet** — your first job
is to elicit:

- *Why* replace the existing ones? (Limited expressiveness? Bad feasibility
  region? Awkward for FEA? Not enough barrier shapes? Inconsistent across the
  family of machines they care about?)
- *What target* should the new parameterisation cover that the current three
  don't?
- *What dimensionality* is acceptable (D=10? D=20?) — affects both BO sample
  complexity and lumped-solver tractability.
- *What invariants* must hold? (e.g. barrier-following channels per §4 of
  `../CLAUDE.md`? Saturation behaviour? Manufacturability?)

Ask before designing.

## Constraint: no PFN

The user explicitly chose to debug the new parameterisation using **standard
Gaussian-Process BO**, not the PFN. Reasons:

- GP-EI is the trusted baseline; if you can't fit a GP to your new
  parameterisation's FEA data, you don't have a parameterisation problem
  worth taking to a PFN.
- The §6.7 "GP capacity diagnostic" (see `../CLAUDE.md` §13) is the
  parameterisation sanity gate: does a SingleTaskGP+ARD on n=64 FEA samples
  reach reasonable RMSE? If yes, the parameterisation is viable; if no,
  rethink.

`machine_design/pfn/` is fully ignorable on this branch. So are
`notebooks/m4_pfn_*.py`, `notebooks/m5_bo_benchmark.py`'s PFN arm, and any
`checkpoints/*.pt`. They sit in the tree harmlessly.

## What you have to work with

**Reusable GP-BO infrastructure** (all on this branch already, from the `pfn`
merge):

- `notebooks/m5_bo_benchmark.py` — pool-based BO loop with GP-EI, analytical
  EI, Random baseline. Drop in the new generator's `(X_pool, y_pool)` and you
  get a complete GP-BO benchmark.
- `notebooks/m4_parametric_bottleneck.py` — strip the PFN arm; the
  `SingleTaskGP @ Matern + ARD, Type-II ML` baseline at varying `n_train` is
  exactly the parameterisation-viability test.
- `notebooks/m4_pfn_implicit_kernel.py` — `_fit_gp` + `_gp_hyperparams` are
  copy-pastable Type-II ML SingleTaskGP helpers with the correct
  `Normalize(d=D, bounds=bounds_t)` input transform and `float64` discipline
  (the *don't* float32 a candidate inside a feasibility check rule).
- `notebooks/run_optimization_smoke.py` — live-ANSYS GP-EHVI smoke variant.
- `machine_design/fea_emulator.py` — `load_fea_designs(name)` for existing
  parameterisations; you'll likely need to add an emulator-loading path for
  the new one once it produces enough FEA data.
- `machine_design/lumped/` — the lumped-reluctance solver. May or may not be
  reusable for the new parameterisation; depends on what the new geometry
  looks like.

**The interface your new generator must implement**: see
`machine_design/generators/HacklGenerator_OneLambda` for the canonical
contract. The downstream code (`m4_5_paired_sweep.py`, `m5_bo_benchmark.py`,
the lumped library builder, the FEA pipeline) expects:

- `gen.bounds` — `(lo, hi)` arrays of shape `(D,)`.
- `gen.X_to_params(c)` — array → named dict.
- `gen.set_parameters(params)` — apply.
- `gen.generate_barriers()` — produce barrier geometry.
- `gen.feasible_barriers(barriers)` → bool — used in rejection sampling.

## Read first

- `../CLAUDE.md` (the parent project doc) — particularly §3 (Sadda's
  read-only tree), §4 (barrier-following channel topology and the lumped
  solver design), §5 (PFN+BO loop), §6 (ANSYS protocol), §11 (data-hygiene
  rules — these apply on this branch too), and **§13 (lessons learned +
  revised plan; the empirical finding that broad-GP > matched-physics is the
  context for why we're rethinking the parameterisation now)**.
- `README.md` — what the repo is.
- This file — your scope.

## Hygiene reminders

- Sadda's tree `/home/sadda/Projects/MachineDesign/` on `bayes` is
  **read-only**. Same rule here.
- Live ANSYS runs follow `../CLAUDE.md` §6 (output to `~/Public/PFN/`,
  bayes SSH, `nohup` + `ppid=1`).
- Don't auto-commit; ask before destructive git operations
  (`git reset --hard`, `git push --force`, etc.).
- File naming for new artefacts: prefix new-parameterisation outputs with the
  parameterisation's name (e.g., `NewParam_n100000_s0.npz`, not
  `OneLambda_*`), so when the two branches eventually merge there are no
  collisions.

## Suggested first session

1. Read `../CLAUDE.md` §3, §4, §11, §13.
2. Ask the user the four questions in "Goal of this track" above.
3. Sketch the new parameterisation (in this doc or a separate
   `PARAMETERISATION.md`); get sign-off on the design before implementing.
4. Implement `machine_design/generators/<NewName>Generator.py` mirroring the
   `HacklGenerator_OneLambda` interface.
5. Generate a small lumped library (~10k tasks) with the new generator to
   sanity-check feasibility/coverage.
6. Run `m4_parametric_bottleneck.py` (GP arm only) on a small FEA sample to
   gate viability before committing to a full library.

## What the parallel `pfn` track is doing

For reference — don't act on this, just be aware:

- The `pfn` worktree is implementing a Gibbs-kernel non-stationary prior
  per `../MachineDesign/PLAN_gibbs_prior.md` (testing whether kernel
  non-stationarity beats Type-II ML GP on FEA-OOD BO).
- Currently uses parameterisations: `OneLambda` (primary), with
  `SixLambdas`/`ThreeBrokenLines` planned for cross-parameterisation
  validation.
- If `newparam` produces a viable new generator, the `pfn` track may
  eventually train a PFN on it — but that's not a near-term obligation.
