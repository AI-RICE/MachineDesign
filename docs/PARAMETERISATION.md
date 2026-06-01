# New SynRM rotor parameterisation — design spec (for sign-off)

Branch: `newparam`. Scope: one **unified** rotor-barrier parameterisation to
replace the three Hackl-style ones (`OneLambda`/`SixLambdas`/`ThreeBrokenLines`),
designed as the substrate for **latent-space BO (LOL-BO)**, debugged with
standard GP-BO. No PFN on this branch. See `../HANDOFF.md` and `../../CLAUDE.md`.

**Status:** signed off; generator + geometry gate implemented and passing.
- `RadialSplineGenerator` in [machine_design/generators.py](../machine_design/generators.py)
  (D=114, repair decoder §6, B-spline encoder §7, θ-nested two-sided projection).
- Geometry gate [notebooks/radialspline_geometry_gate.py](../notebooks/radialspline_geometry_gate.py):
  **1000/1000 random feasible**, warm-start round-trips all three Hackl families
  at **~0.35 mm** radial RMSE (50/50 feasible each). Renders look like plausible
  SynRM rotors.
- Prior policy **2** (realistic-biased, heavy tails) per P8.
- **Next:** VAE + latent GP-fit viability gate (§11.2 / §13 test 6).

Working name: **`RadialSpline`** (files/artefacts prefixed `RadialSpline_*`).
Changeable — say the word.

---

## 1. Decisions (this track)

| # | Decision | Choice |
|---|---|---|
| P1 | Representation family | **A** — functional boundary, polar `r(θ)` per barrier. (B = polar SDF/level-set is the backup if fixed topology proves limiting.) |
| P2 | Symmetry | **Non-symmetric** allowed (barriers need not mirror about the d-axis). Conventional d-axis symmetry is the backup if FEA shows pathological even harmonics. |
| P3 | Barrier count | **Fixed N = 3** (matches current family). |
| P4 | Dimensionality | **K = 18** control points per boundary ⇒ **D = 114**. (Low end of 100–300, chosen for speed.) |
| P5 | Decoder | **Repair / projection** (always-valid decode). Reject-decode is the backup. |
| P6 | Lumped solver | Disposable on this branch (not required). |
| P7 | BO method | LOL-BO (Maus et al., NeurIPS 2022): TuRBO trust region in a jointly-retrained VAE latent. |
| P8 | Initial manifold | **Broad prior sampler** (diverse feasible geometries, not just Hackl perturbations) so the VAE starts wide — mirrors `../../CLAUDE.md` §13 *coverage beats specificity*. See §13. |

---

## 2. Geometry frame

- One pole, angular range `θ ∈ [0°, 90°]`, d-axis at 45°, q-axes at 0°/90°.
- Polar `r(θ)` centred at the rotor centre.
- Reference machine ([design.py](../machine_design/design.py)): `r_min = 12.5 mm`
  (shaft), `r_max = 39.275 mm` (rotor surface), radial build ≈ 26.8 mm, steel
  M350-50A (0.5 mm laminations), 4-pole / 36-slot, stack 85 mm.
- Rotor iron = annulus `[r_min, r_max]` minus the 3 air barriers minus the
  inserted central rib.

---

## 3. Parameter vector layout (D = 114)

Barrier index `b ∈ {0,1,2}`, `0` = innermost (near shaft), `2` = outermost
(near surface). Each barrier is an air pocket bounded by an **inner** boundary
`r_in_b(θ)` (shaft-side) and an **outer** boundary `r_out_b(θ)` (surface-side),
`r_in_b < r_out_b`.

Per barrier (38 values):

| Slot | Count | Meaning |
|---|---|---|
| `θ_lo_b`, `θ_hi_b` | 2 | angular extent (deg); constrained so `θ_lo_b < 45 < θ_hi_b` (every barrier crosses the d-axis, so the central rib pierces it — matches current design) |
| `c_out_b[0..17]` | 18 | B-spline control radii of the outer boundary |
| `c_in_b[0..17]` | 18 | B-spline control radii of the inner boundary |

Total `3 × 38 = 114`.

Boundary curve: `s ∈ [0,1]`, `θ(s) = θ_lo_b + s·(θ_hi_b − θ_lo_b)`,
`r_out_b(s) = ` clamped cubic B-spline on **fixed uniform knots** with control
values `c_out_b` (same for `r_in_b`). Fixed knots ⇒ the VAE always sees one
consistent basis.

> Backup bases (drop-in, same layout): Chebyshev/cosine coefficients (fewer DOF
> for very smooth curves) or raw radii at fixed `s` (simplest). B-spline chosen
> for local control + trivial least-squares warm-start.

---

## 4. Structural constraints (values confirmed)

All four feed the **repair** decoder (§6), so every design is manufacturable
by construction. Defaults are named, citable config (per `../../CLAUDE.md` §11):

| Constraint | Symbol | Default | Provenance |
|---|---|---|---|
| Surface iron (tangential bridge): `r_max − r_out_2(θ) ≥` | `t_bridge` | **0.7 mm** | `r_stator_end`, in every run script |
| Central d-axis rib full width | `w_rod` | **0.5 mm** (`offset = w_rod/√2 ≈ 0.35`) | `split_barriers`, in FEA path |
| Inter-barrier iron: `r_in_{b+1}(θ) − r_out_b(θ) ≥` | `t_rib` | **0.5 mm** | = 1 lamination (override later w/ datasheet) |
| Shaft iron: `r_in_0(θ) − r_min ≥` | `t_shaft` | **0.5 mm** | = 1 lamination (override later) |
| Min air width: `r_out_b(θ) − r_in_b(θ) ≥` | `min_air` | **0.5 mm** | keeps pockets non-degenerate |

Thickness is checked **radially** (fast); the **authoritative** check is
perpendicular distance via Shapely `LineString.distance` (rigorous; reuses the
repo's existing Shapely dependency).

---

## 5. Central rod (d-axis radial rib)

Reuse the existing mechanism exactly: after assembling air polygons, apply
`split_barriers` with `offset = w_rod/√2 ≈ 0.35`, which removes the air strip
`|x − y| ≤ offset` along the d-axis, leaving a `w_rod ≈ 0.5 mm` iron rib through
every barrier. Identical to the FEA baseline ⇒ no new rib code, no new modelling
risk.

---

## 6. Decoder (params → FEA geometry) — repair, always-valid

1. Clamp `θ_lo_b ∈ [θ_qmargin, 45−δ]`, `θ_hi_b ∈ [45+δ, 90−θ_qmargin]`
   (`θ_qmargin` keeps a q-axis iron web; `δ` keeps a finite d-axis span).
2. Evaluate `r_in_b`, `r_out_b` on a shared fine θ-grid (each barrier only where
   it exists, i.e. `θ ∈ [θ_lo_b, θ_hi_b]`).
3. **Sequential radial projection**, shaft → surface, per θ:
   - floor = `r_min + t_shaft`;
   - for `b = 0,1,2`: `r_in_b ← max(r_in_b, floor)`;
     `r_out_b ← max(r_out_b, r_in_b + min_air)`;
     floor ← `r_out_b + t_rib`;
   - finally clamp `r_out_2 ≤ r_max − t_bridge`, re-projecting inward if needed.
   This always yields valid **nested** geometry (isotonic-style projection).
4. Build each closed air polygon: outer boundary (θ ascending) + radial end-cap
   + inner boundary (θ descending) + radial end-cap; start == end
   (`check_barrier` contract).
5. Insert central rib via `split_barriers` (§5).
6. Return `list[np.ndarray]` in the exact format `add_rotor_barrier` consumes.

> End-cap choice: **blunt radial caps** (robust, always valid). Backup: taper to
> a point (`r_in = r_out` at ends) for sharper barrier tips.
> Decoder-strategy backup (P5): reject-decode + `objective_fallback` if repair
> folds the latent space badly.

---

## 7. Encoder / warm-start (existing → RadialSpline)

N = 3 matches the current family ⇒ **one-to-one, no padding**.

1. Take an existing design's `generate_barriers()` (dense closed polylines).
2. Per barrier: to polar, split into outer/inner halves by θ, set
   `θ_lo,θ_hi = min/max θ`, **least-squares fit** the 18 control radii per half.
3. Existing designs are d-axis-symmetric ⇒ a subset of the non-symmetric space;
   warm-start lands inside the new space exactly.

Round-trip test (encode→decode ≈ identity, geometry only, no FEA) is the first
acceptance gate.

---

## 8. Bounds (leak-free)

Per-barrier **tiered radial bands** + angular bounds, derived by **sampling the
three existing generators' geometry envelopes** (public bounds, **no FEA torques
— §11 clean**). This gives a box that (a) contains all warm-start designs and
(b) starts the repair near-feasible. Concretely: run the existing generators
over their public `bounds`, take the radial min/max per nesting level → tier
bands for `c_in_b`/`c_out_b`; take observed φ-ranges → `θ` bounds. Repair
handles any residual ordering violation.

---

## 9. VAE + LOL-BO

- Raw D = 114 → VAE latent ≈ 12–16 (MLP-VAE; backups: 1-D conv along the
  control-point axis, or shared per-barrier encoder + DeepSet).
- VAE training data = sampled feasible RadialSpline vectors (random smooth
  curves passed through the repair decoder) ∪ warm-start encodings of the
  existing designs ⇒ the latent manifold *is* the feasible-geometry manifold.
- LOL-BO: TuRBO trust region in latent space, periodic joint VAE retrain on
  accumulating (design, objective) data. Repair decoder ⇒ **every latent point
  is FEA-evaluable** (the property that makes latent TR-BO efficient).

---

## 10. Interface contract (mirrors `HacklGenerator_OneLambda`)

- `bounds` → `(lo, hi)`, shape `(114,)` (§8).
- `X_to_params(X)` → `(θ, c_out, c_in)` per barrier (reshape 114 → 3×38).
- `set_parameters(params)` → store.
- `generate_barriers()` → repair decode (§6) → `list[np.ndarray]`.
- `feasible_barriers(barriers)` → reuse existing (in-bounds + simple +
  non-intersecting); with repair it should ~always pass — kept as a tripwire.
- `random_parameters()` → sample box → repair (for VAE/prior data + rejection
  init).

---

## 11. Viability gate — adapted (important)

The HANDOFF "GP+ARD on n=64 raw FEA" gate **changes meaning at D=114**: plain
GP-BO doesn't work in 114-D — that's *why* LOL-BO exists. So the gate is
re-sequenced:

1. **Geometry gate (no FEA):** random feasible coverage, renders look like
   plausible SynRM rotors, warm-start round-trips existing designs.
2. **Latent gate (needs VAE, no new FEA):** re-encode the *existing* simple-param
   FEA designs into RadialSpline → VAE latent; does `SingleTaskGP+ARD` fit the
   already-available torques in latent space at small n? This is the real
   parameterisation-viability test.
3. **Live gate:** small LOL-BO run vs GP baseline (Phase 3, later).

> Open: do we accept this re-sequencing, or also want a low-D-projection GP
> sanity on raw space first?

---

## 12. Implementation plan

1. `machine_design/generators.py`: add `RadialSplineGenerator` (interface §10,
   decoder §6, encoder §7). Reuse `split_barriers`, `feasible_barriers`,
   `get_arc`.
2. Geometry gate (§11.1): random designs + renders + round-trip test.
3. Bounds derivation script (§8), output `RadialSpline_bounds.npz`.
4. VAE + warm-start dataset; latent gate (§11.2) using existing FEA.
5. LOL-BO loop (reuse `m5_bo_benchmark.py` GP infra for the baseline arm).

---

## 13. Pre-FEA VAE / manifold tests + expansion strategy

The VAE is trained on **geometry only** (no torques), so manifold quality is
testable before any FEA. Run on a **train/test split** of the feasible-vector
pool (sampled designs ∪ warm-start encodings).

**Label-free tests (fully §11-clean):**
1. **Reconstruction train/val split** — overfit signature = low train / high val
   recon error. The direct "manifold memorising vs generalising" test.
2. **Repair-magnitude on prior samples** — sample `z ~ N(0,I)`, decode, measure
   how far repair had to move things. Small = manifold aligned with the feasible
   set; large = VAE decodes to infeasible raw geometry (loose/misaligned).
3. **Latent interpolation smoothness** — interpolate two encoded designs, decode
   along the path, check geometry morphs smoothly (no jumps/self-intersections).
4. **Posterior-collapse / dim usage** — KL per latent dim; dead dims ⇒ effective
   latent (and manifold) narrower than nominal.
5. **Round-trip on existing designs** (= §11.1) — manifold *contains* the
   known-good region.

**Label-using test (§11 boundary):**
6. **Latent GP fit, train/test split** (= §11.2) — re-encode existing FEA
   designs, fit GP+ARD in latent on a train split, test RMSE/Spearman on
   held-out. Guards against the GP overfitting the latent.
   **Hygiene constraint (binding):** the existing FEA *torques* may be used to
   *measure* test 6, but **never to tune VAE hyperparameters** — the VAE is
   selected on reconstruction (geometry, leak-free) only. Tuning the VAE against
   test 6 would re-open the §11 leak.

### Manifold expansion in LOL-BO

LOL-BO **does** re-shape the manifold during optimisation — this is its key
advantage over a frozen-VAE latent BO:
- **Joint periodic retraining**: the VAE is retrained on accumulating data with
  a combined loss (reconstruction + surrogate/DKL alignment), so the manifold
  follows the search and absorbs newly acquired edge points.
- **Trust region (TuRBO-style)**: recenters on the incumbent, adapts size,
  restarts on collapse.

**Limit (honest):** expansion is **local/incremental** — BO proposes only by
decoding latents, so a great design in a never-seen geometry region is reached
only by *walking there* through acquired neighbours, not by teleporting. A
too-narrow **initial** manifold therefore slows or strands the search.

**Decision P8 — broad initial prior.** The primary cure is on our side and
mirrors `../../CLAUDE.md` §13 (*coverage beats specificity*): the prior sampler
must generate **diverse** feasible geometries (wide θ spans, varied curvatures),
not just perturbations of the three Hackl designs, so the VAE *starts* broad.
The overfitting tests above are how we detect starting too narrow.

**Widening knobs, by leverage:** (1) broaden the prior sampler [main lever];
(2) retraining frequency + recon-vs-objective weight; (3) acquisition-time
latent temperature (`z` slightly wider than `N(0,1)`); (4) trust-region
restarts into unexplored latent.

> Backup: if the manifold stays too narrow despite a broad prior, evaluate
> LOL-BO successors that regularise latent–objective alignment / coverage
> (e.g. CoBO-style Lipschitz alignment) — web-search to verify specifics before
> adopting.

---

## 14. Backup-choices summary (one place)

| Fork | Primary | Backup |
|---|---|---|
| Representation | A (`r(θ)` spline) | B (polar SDF/level-set) |
| Symmetry | non-symmetric | d-axis symmetric |
| Basis | clamped cubic B-spline | Chebyshev / raw radii |
| End caps | blunt radial | tapered to point |
| Decoder | repair | reject + fallback |
| VAE encoder | MLP | 1-D conv / DeepSet |
| Barrier count | fixed N=3 | presence-gated slots |
| Manifold width | broad prior + LOL-BO joint retrain | CoBO-style latent–objective alignment |
