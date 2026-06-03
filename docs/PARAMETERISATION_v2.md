# Unified SynRM parameterisation — plan v2 (Bézier superset)

**Goal:** one parameterisation that is *better* than the three Hackl families
(OneLambda 7-D / SixLambdas 12-D / ThreeBrokenLines 13-D) — replacing all three.

**Supersedes** [PARAMETERISATION.md](PARAMETERISATION.md) (the `r(θ)` RadialSpline
spec), which is now a **documented negative result**: a single-valued `r(θ)`
spline cannot represent the **re-entrant barrier ends** near the surface (E9 /
`twod_vs_rtheta.png`), and torque is so feature-sensitive that geometry RMS is a
poor proxy. Living doc — updated on the fly.

## Three non-negotiable principles (hard lessons from v1)

1. **Validate against *converged-mesh* FEA, never geometry RMS.** 3 mm FEA
   overstates torque ~2.7% and carries a ~3% same-geometry artifact; the
   ~0.04 N·m inter-family differences live inside it (E9). Converged mesh ≈ 0.5 mm
   (confirm airgap/stator too); residual FEA-noise floor ~1% = the significance
   threshold for every comparison.
2. **The family must contain all three Hackl families *exactly*** → warm-start is
   lossless and the comparison carries no re-encode confound.
3. **Feasibility is a learned constraint, not a representation straitjacket**
   (decision below) — so the geometry can stay free/expressive.

## What "better" means + the bar

- **Objectives (primary):** maximise T_mean, minimise T_ripple (the Pareto front).
  ⚠️ **Escape hatch:** the single-operating-point front may be near-saturated
  (~4.4–4.45 N·m ceiling); if designs don't separate, move to a **richer
  evaluation** (multiple current angles / load points / added objectives). Keep
  this option open from the start.
- **The bar to beat:** the **union of the three families' Pareto fronts at
  converged mesh**. Since the new family *contains* all three, its warm-start *is*
  that union front; success = BO pushes designs that **dominate** it.
- **Success (any one is a win):** (a) unified front **dominates** the union
  (beyond ~1% noise); (b) **matches** it with **fewer FEA evals**; (c) **ties**
  but with **one** parameterisation instead of three (the practical motivation).

## The family: 2-D piecewise-cubic-Bézier superset

Each barrier = inner + outer **2-D piecewise-cubic-Bézier** boundaries with
**corner-capable (C⁰) joints** — Hackl's own construction generalised:
- smooth curves (1λ/6λ) → aligned tangents; **corners/straight lines** (3BL) →
  non-collinear/collinear control points; **re-entrant ends** → native (2-D
  parametric, the `r(θ)` killer).
- Contains OneLambda/SixLambdas/ThreeBrokenLines as **exact** parameter settings
  (beziers exact; arcs to ~1e-3 mm; broken lines exact).
- DOF ~100–200 → the **DSP-GP / SAASBO** optimisers we validated (E3/E5) apply.

## Feasibility = constrained BO (decision)

No by-construction straitjacket. Instead:
- a **cheap, geometry-only validator** (Shapely: simple, non-self-intersecting,
  nested, in-rotor, min-iron — **no FEA**);
- a **second GP modelling P(feasible)**; acquisition = **EI · P(feasible)**
  (constrained EHVI for the MO case), and/or the validator as a **hard filter
  inside the acqf optimiser** (reject infeasible candidates before FEA).
- This frees the control points to be unconstrained while spending FEA only on
  feasible designs. (Backup: by-construction centerline+width if the feasible
  region proves too thin for the constraint-GP to learn efficiently.)

## Optimiser

DSP-GP-BO (single-obj sanity) → **qLogEHVI / MORBO** (multi-obj), warm-started
from the pooled exact re-encodings, **all FEA at the converged mesh**.

## Step-by-step (gate-by-gate)

| # | Step | Gate / success |
|---|---|---|
| 0 | **Lock FEA fidelity** — converged mesh (rotor+airgap+stator), pipeline default, noise floor | torque reproducible <1% |
| 1 | **Set the bar** — pooled three-family Pareto front at converged mesh | bar recorded |
| 2 | **Design + prototype the Bézier superset**; prove **exact Hackl containment** (geometry) | each family round-trips to ≪ torque scale |
| 3 | **Converged-FEA fidelity gate** — re-encode each family → FEA at converged mesh → torque matches originals | <1% (the gate v1 failed) |
| 4 | **Generator + feasibility** — encode/decode/bounds; geometry validator; feasibility-GP; broad prior; pooled exact warm-start | feasible-by-acqf working; warm-start lossless |
| 5 | **BO at converged mesh** — DSP-GP sanity, then qLogEHVI/MORBO vs the bar | dominate / sample-efficiency / tie-unified |
| 6 | **Decision + write-up** | honest verdict (any outcome is publishable per D7) |

## Standing risks
- **Feasible region geometry** — if the constraint-GP can't learn it from cheap
  validator labels (thin/disconnected region), fall back to by-construction.
- **Saturated objective** — invoke the richer-evaluation escape hatch.
- **FEA cost at 0.5 mm** — ~minutes/eval; budget the sweeps accordingly.
