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
- **DOF ≈ 100** (richer than Hackl 7–13, workable) → fixed M segments/barrier
  (~33 DOF/barrier). Coarser than the lossless encoder, so the **converged-FEA
  fidelity must be re-checked at this D** (Step 3 was at ~0.04 mm; D≈100 gives
  ~0.1–0.2 mm — verify it stays <1–2% torque, bump D if not). DSP-GP/SAASBO apply.

## Feasibility = cheap checker as a hard constraint (no second GP)

A feasibility GP only pays off when feasibility is *expensive*. Ours is **cheap
(geometry-only, ~ms)** — Shapely: simple / non-self-intersecting / nested /
in-rotor / min-iron. So use it **directly inside the acquisition optimiser**:
generate candidates → **filter to feasible with the checker** → optimise the acqf
over the feasible set (validate before any FEA). No second model; zero FEA wasted
on infeasible designs; control points stay free. (Backup if the feasible region
is too thin: by-construction centerline+width.)

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

## Encoder design (exact / structure-aware) — the warm-start that matters

**Why (Step-4b finding).** A naive uniform-arclength *fit* encoder at the BO
resolution M=6 (D=108) loses **−4.6% torque** vs the original Hackl (`step4_fea_check.py`):
torque sensitivity is ≈ **20 × (geometry error in mm) %**, so 0.23 mm → 4.6%.
Reaching <1% by *fitting* would need ~0.05 mm → M≈16–20 → D≈300 — well above the
D≈100 target. The fit, not the representation, is the bottleneck (an M=6 chain
*can* hold a Hackl barrier exactly: 2 beziers + 2 arcs ≈ 4–6 cubics).

**Approach — encode the natural pieces, then subdivide losslessly to M:**
1. **Natural-piece segmentation.** Breakpoints = sharp **corners** (turning-angle
   jumps — the broken-line vertices) **+ adaptive splits**: recursively split each
   inter-corner arc at its max-fit-error point until per-piece error < ε
   (~0.01 mm). This auto-discovers the (smooth) bezier↔arc junctions without
   hand-coding them. Broken-line segments are **exact degenerate cubics**; the
   ~6° end arcs are cubic to ~1e-5 mm. → near-lossless natural chain.
   - ThreeBrokenLines: ~6 natural pieces (3 line-cubics + 2 arcs + 1 inner bezier).
     OneLambda/SixLambdas: ~4 (outer bez + 2 arcs + inner bez).
2. **Reach fixed M by exact de Casteljau subdivision.** A cubic splits into two
   cubics tracing the **identical** curve. For a family with < M natural pieces,
   split the longest segments until exactly M — **lossless** (geometry unchanged);
   the extra control points are **spare DOF for BO to deform that region**.
   "Where do the remaining segments go?" → here, by exact subdivision.
3. **Subdivision rule:** split longest segments / favour the smooth boundaries;
   **corners and bridge points stay pinned breakpoints** (they carry the torque),
   so similar geometries map to similar control-point layouts (smooth GP map).
4. **M floor = max family natural count** (≈6, ThreeBrokenLines) ⇒ default
   **M=6 → D=108**; M configurable for margin.

**Verification gate:** re-encode each family → (a) geometry round-trip ≪ torque
scale, (b) **FEA ΔT <1%** at M=6 (redo of Step-4b with the exact encoder).

## Progress

**Step 4 — encoder DONE** ([`../machine_design/bezier_generator.py`](../machine_design/bezier_generator.py);
test [`../../notebooks/bezier_generator_test.py`](../../notebooks/bezier_generator_test.py)).
Final `BezierSupersetGenerator(M=6 → D=108, n_per=160)`. Implementation lessons:
- **The −4.6% at M=6 was two artifacts, not a representation limit.** (a) `_fit_cubic`
  used **chord-length** parameterisation while Hackl samples **uniformly in the Bézier
  parameter** → fixed by uniform-t fitting (a natural piece is then recovered to
  **1e-14, exact**); (b) `_decode_one` sampled only **n_per=40/segment** → a 35 mm
  segment got ~0.9 mm point spacing, so the FEA polygon (and the nearest-neighbour
  metric) was coarse → fixed by **n_per=160** (~1000 pts, matching the original).
- **Encoder = adaptive recursive split to ε + lossless de Casteljau pad to M.**
  Corner-threshold detection missed gentle broken-line vertices; recursive
  splitting at the max-error point auto-finds every vertex/junction. Spare
  segments for simpler families come from exact de Casteljau subdivision.
- **Two robustness fixes for 3BL** (raised it 74% → **100%** feasible re-encode):
  straight-snap near-collinear pieces (no spurious bow), and **drop degenerate
  zero-length pieces** (their coincident anchors caused the surface-hugging
  d-axis-pinch self-cross).
- **Result:** all three families **~99–100% feasible** re-encode over 120 random
  designs; geometry ~0.06–0.08 mm (smooth) / ~0.12 mm (3BL) on the top-torque set
  (nearest-neighbour metric, decode-sampling-limited — pieces are exact). Feasibility
  is a hard filter (Shapely simple + min-iron/rib/bridge/shaft). Perturbation
  feasibility around a warm-start: σ=0.1 mm → 34%, σ=0.25 → 13% (thin feasible
  region — the acqf optimiser must filter; flagged as a standing risk).
**Step 4b — FEA fidelity gate PASS** (`../../notebooks/step4_fea_check.py --M 6`,
converged mesh 0.5/0.5; log `step4_m6_fixed.log` on bayes). M=6/D=108 re-encode
vs original Hackl: **worst |ΔT| = 0.94% (PASS <1%)**, all feasible —
OneLambda −0.55, SixLambdas −0.94, 3BL −0.71, SixLambdas-lowRip −0.24; ripple
within 0.4 pts. The two encoder fixes moved M=6 from **−4.6% → −0.94%**. ⇒
**D=108 confirmed** as the warm-start/BO setting; warm-start is lossless to <1%.
- *Caveat (recorded):* all ΔT are slightly **negative** (~2× the 0.4% noise floor)
  — residual decode-polygon under-torque. Consistent across Bézier space (no
  internal BO bias), but a ~0.9% handicap vs a bar on original-Hackl geometry →
  **re-evaluate final-front designs at higher `n_per`** before any dominate claim
  (as the ICEM GP-EHVI Pareto reeval did). Cheap insurance: raise `n_per` for the
  reported designs only.

- **Pending in Step 4:** broad prior sampler + pooled exact warm-start (skip the
  ~1% infeasible re-encodes), and wiring the feasibility-constrained acqf.


**Step 0 — DONE** (`../../notebooks/step0_fidelity.py`). Converged FEA setting =
**rotor mesh 0.5 mm + airgap (Band) mesh ~0.5 mm**. Airgap refinement moves mean
torque ≤0.4% (OneLambda) / 0% (SixLambdas); **noise floor (same-shape resample)
~0.06–0.4%** at this setting — vs ~3% at the old 3 mm default, so design
differences down to a few tenths of a % are now resolvable. `Design.compute` has
`mesh_length` + `airgap_mesh` knobs; **v2 pipeline uses (0.5, 0.5)**. Gotcha:
AEDT mesh ops persist across `compute()` calls (named ops overwrite; absence
doesn't clear) — the BO loop is unaffected (never assigns airgap; "rotor"
overwrites by name), but interleaving airgap settings in one session needs a
mesh-clear. Standing convention: **every v2 FEA call passes explicit
(mesh_length=0.5, airgap_mesh=0.5)**.

**Step 3 — PASS** (`../../notebooks/step3_fidelity_gate.py`). Bézier re-encode →
FEA at converged (0.5, 0.5) vs original Hackl: **worst |ΔT| = 0.51%** (OneLambda
−0.38, SixLambdas +0.09, 3BL −0.51, SixLambdas-lowRip +0.28; ripple within
0.4 pts), barely above the noise floor. The ~0.04 mm corner-aware re-encode is
**effectively lossless in torque** — the gate `r(θ)` failed (11 mm → 9–12% loss).
⇒ warm-start from the Hackl families will be lossless; representation + FEA
foundation fully validated.

**Step 2 — PASS** (`../../notebooks/bezier_superset_proto.py`; fig
[`figures/bezier_superset_containment.png`](figures/bezier_superset_containment.png)).
Corner-aware fit (detect tangent jumps → C⁰ breakpoints → cubic per sub-segment)
contains all three families to **~0.04 mm** max round-trip (rms 0.018), vs `r(θ)`'s
11 mm — ~280×. Corners (4/barrier smooth, 5/barrier 3BL) and re-entrant ends both
captured. Accuracy↔DOF tradeoff (OneLambda): 38 DOF/barrier → 0.19 mm, 50 → 0.11,
66 → 0.08, 122 → 0.04 (even at `r(θ)`-matched 38 DOF it's ~60× better). **Step-4
design note:** for a fixed-D BO vector, use a fixed segment count M per barrier
(free C⁰ tangents → corners emerge where tangents misalign); ~M=12 → ~0.08 mm.

## Standing risks
- **Feasible region geometry** — if the constraint-GP can't learn it from cheap
  validator labels (thin/disconnected region), fall back to by-construction.
- **Saturated objective** — invoke the richer-evaluation escape hatch.
- **FEA cost at 0.5 mm** — ~minutes/eval; budget the sweeps accordingly.
