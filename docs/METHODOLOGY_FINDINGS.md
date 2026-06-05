# Methodology findings — high-dimensional BO over parametric-curve geometries

**Scope.** This note distils the *methodological* lessons from the `newparam`
branch, abstracted from the synchronous-reluctance-motor (SynRM) testbed it was
run on. The application was a vehicle; the product is the methodology for
**Bayesian optimisation over free-form geometric parameterisations** (here:
piecewise-cubic Bézier flux-barrier boundaries, D≈100).

**Motivation / where this fits.** Long-term goal: *high-dimensional BO over
Bézier-like (and 3D-generalisable) curve/surface families that serve as generic
building blocks for machine design.* This branch's narrow question was **how to
evaluate one geometric parameterisation against another** — which forced us to
build, and debug, the whole comparison machinery. A sister project pursues the
PFN (amortised-prior) angle; several findings below rhyme with it and are flagged.

Status: the comparison machinery is built and validated end-to-end on live FEA;
the SynRM single-operating-point comparison returned a clean **negative result**
(below), which is itself a methodological contribution.

---

## 1. The "exact-superset" parameterisation pattern

To compare a rich parameterisation against hand-crafted baselines *without a
re-encoding confound*, make the rich family a **superset that contains each
baseline as an exact special case**.

- We used a closed chain of **M cubic Bézier segments per barrier, free control
  points, C⁰ joints** (corners emerge where tangents misalign). It contains all
  three baseline families (smooth-arc "OneLambda/SixLambdas" and piecewise-linear
  "ThreeBrokenLines") essentially exactly: straight segments are degenerate
  cubics; circular arcs are cubic to ~1e-3 mm; the baselines' own Béziers are
  recovered to ~machine precision.
- Payoff: the **warm-start is lossless** — every baseline design re-encodes into
  the rich space and back with no objective penalty, so any BO improvement is
  attributable to the search, not to representation change.
- **Generalises to 3D**: the same pattern is tensor-product Bézier patches /
  T-splines for surfaces. The encoder, fidelity and feasibility discipline below
  carry over unchanged; only the primitive changes.

**Transferable rule:** *if you claim a richer parameterisation helps, prove it
contains the baseline exactly first — otherwise the warm-start itself biases the
comparison.*

---

## 2. Encoder lessons (fitting baseline designs into the rich family)

The encoder (baseline geometry → rich-family control vector) is where most of the
subtle bugs lived. All of these are general to "fit a flexible curve family to
existing designs":

1. **Parameterise the fit to match the data's sampling.** Curves sampled
   uniformly in their native parameter must be fit with **uniform-t**, not
   chord-length. Chord-length fitting of a uniform-sampled cubic introduces
   error precisely because the Bézier speed is non-uniform. Fixing this took the
   per-piece recovery from ~0.2 mm to **1e-14 (exact)**.
2. **Decode resolution is a fidelity knob, not a detail.** The control vector is
   exact, but the *polygon* handed to the simulator has finite resolution
   (`n_per` points/segment). Too coarse → a **systematic objective bias** (here
   −0.9% at n_per=160; −0.5% at the fine proto). Match the baseline's vertex
   density.
3. **Recover natural pieces by adaptive recursive splitting** (split a piece at
   its max-fit-error point until < ε). This auto-discovers both hard corners
   *and* smooth curvature junctions without threshold tuning. Then reach a
   **fixed dimension by lossless de Casteljau subdivision** (a cubic splits into
   two cubics tracing the identical curve — spare DOF, zero error).
4. **Robustness guards matter:** snap near-collinear pieces to exact straight
   segments (kills spurious bows), and drop degenerate zero-length pieces (their
   coincident control points cause self-intersections at sharp pinches). These
   took the worst baseline family from 74% → **~100% feasible re-encode**.

---

## 3. Fidelity discipline — the backbone

The single most important methodological lesson, and the one most likely to
silently corrupt results:

- **Validate against converged-simulator output, never a geometry proxy.**
  Geometry RMS is a *poor* surrogate for the objective: here torque sensitivity
  was ≈ **20 × (geometry error in mm) %**, and a smoother fit with 70× better RMS
  was *worse* in torque because it rounded objective-critical features. Always
  close the loop through the real evaluator.
- **Converge the simulator's own discretisation before any comparison.** The
  coarse default mesh (3 mm) overstated the objective ~2.7% *and* reordered the
  baseline Pareto front — only **7 of 15** apparent front designs survived at the
  converged mesh; the rest were mesh artifacts. A coarse evaluator manufactures
  spurious winners.
- **Establish a noise floor** (re-evaluate identical geometry) as the
  significance threshold for every later claim (~0.4% here).

**Rhymes with the PFN line:** the sister project's "normalisation-leak →
too-good-to-be-true NLL" failure is the same species of bug — *a metric that
looks great because it's measuring an artifact, not the quantity of interest.*

---

## 4. Feasibility as a cheap hard constraint inside the acquisition

When feasibility is **cheap** (geometry validator, ~ms) relative to the objective
(FEA, minutes), do **not** train a feasibility GP. Instead:

- Use the validator as a **hard filter inside the acqf optimiser**: generate
  candidates, keep the feasible ones, optimise the acquisition over them.
- The feasible region is **thin** in the raw box (uniform points ≈ 0% feasible;
  σ=0.1 perturbations of a feasible design only ~34% feasible). Two fixes:
  (a) a **data-driven box** scaled to the warm-start region, and
  (b) generate candidates by **perturbing feasible anchors** at a range of step
  sizes, not by sampling the box.
- This worked cleanly on live FEA: **0 failed/infeasible evaluations** across a
  full run.

**The feasibility-resolution trap (important, general).** The feasibility check
must run at the **same or finer resolution as the final evaluation.** We checked
feasibility on the n_per=160 decode but re-evaluated finals at n_per=320 — and
**3 of 4 BO-found front designs were infeasible at the finer resolution.** BO, an
adversary that hugs the constraint boundary, exploited the coarse check's blind
spot (a coarse polygon misses self-intersections a fine one catches). *Any
cheap-constraint-as-filter BO must validate at evaluation fidelity, or add a
threshold margin.*

---

## 5. Comparison hygiene for warm-started BO

- **Decode-bias matching.** If BO scores designs through a decode that differs
  from the baseline's geometry pipeline, there is a systematic offset (the −0.9%
  above). The comparison must be made at **matched fidelity**: either re-evaluate
  the BO front at high fidelity (bias removed) *or* push the baseline through the
  same decode (bias cancels). Reporting the raw mixed-fidelity HV is wrong — we
  initially mis-read an 8–12% HV "gap" that was mostly this.
- **Select the warm-start by reliable labels.** Our first warm-start was ranked
  by the *cheap/coarse* (3 mm) objective and missed the true converged-front
  designs (a design that was 4.26% ripple at 3 mm was 2.80% converged). Re-seeding
  by **converged** labels fixed it. *Never select seeds with the proxy you don't
  trust for the final metric.*

---

## 6. Headline contribution — the **objective-headroom diagnostic**

A richer parameterisation can only beat a baseline via BO if **the objective has
room above the baseline front**. This must be checked *before* investing in the
optimisation.

- Concretely: re-evaluate the **baseline's union Pareto front at converged
  fidelity** (the "bar"). If it occupies a **tight band**, no surrogate can
  dominate it, and the limiting factor is the *problem*, not the method.
- On the SynRM single operating point the bar's T_mean spanned only **~1.8%** —
  near-saturated. BO seeded from the true converged front added **+0.1% HV in 40
  iterations**; after bias correction its front was made entirely of re-encoded
  baseline designs, "beating" the bar only by noise-level margins. **No genuine
  domination — because there was no headroom.** (Figures:
  `step5_pilot2_vs_bar.png` shows the near-vertical T_mean wall;
  `front_geometry_compare.png` shows three *distinct* geometries — smooth 6λ,
  angular 3BL, and a BO hybrid — at *identical* performance, the signature of a
  saturated objective.)
- **This is the direct analogue of the PFN line's "prior-capacity diagnostic":**
  a cheap, pre-declared test that decides whether a sample-efficiency comparison
  is even worth running. Here it is "objective headroom"; there it is "does the
  prior beat the raw oracle". Both gate the expensive claim.

**Transferable rule:** *parameterisation richness only pays off when the objective
has headroom over the baseline; measure the bar first. A flexible family that
ties the baseline at a saturated objective is a parameterisation success and an
optimisation no-op.*

---

## 7. Engineering enablers (worth reusing)

- **License-aware parallelism.** Probe the actual constraint: here 1-core FEA
  solves draw **0 HPC tokens** → parallelism is seat-limited (~25), not
  HPC-limited (4). Embarrassingly-parallel phases (bar, warm-start) run as many
  1-core solves; the q=1 BO loop is the serial part (→ use q-batch acquisition to
  parallelise it). Saved ~8× wall-clock on the bar.
- **Resumable parallel batch drivers** (per-item output, shard-by-worker) and
  **mock-testable BO runners** (NN-lookup objective) — catch wiring bugs with zero
  simulator cost before spending FEA hours.

---

## 8. Open methodological questions (the forward agenda)

1. **Create headroom, then test the parameterisation.** A multi-operating-point /
   multi-load objective (or torque-per-loss) widens the objective space so designs
   separate — the precondition for a real richness-vs-baseline test. Until then
   the parameterisation claim is untestable on this objective.
2. **High-dim BO method choice in thin-feasible, off-manifold spaces.** D≈100 with
   DSP-GP (√D LogNormal lengthscale prior) + qLogEHVI ran mechanically, but the
   feasible manifold is thin and the interesting designs are *off* the baseline
   manifold. Candidates to compare: **TuRBO** (trust regions suit thin feasible
   regions), **SAASBO** (sparse-axis priors), latent-space BO, and structured /
   learned priors over the control vector.
3. **PFN amortisation over the geometric prior (bridge to the sister project).**
   The control-vector space has strong structure (control points are ordered,
   nested, feasibility-constrained). A PFN meta-trained on a *geometric* prior —
   random feasible barrier sets, or random machines — could amortise both the
   surrogate and the feasibility manifold. This is the natural merge point of the
   two projects.
4. **3D generalisation.** Tensor-product Bézier patches / T-splines as the
   surface analogue; the encoder, fidelity, feasibility-resolution and headroom
   lessons transfer directly. The open question is dimensionality (surfaces →
   D≈10²–10³) and whether the same BO methods scale.

---

## 9. Artifact index (evidence for the above)

- Parameterisation + encoder: `machine_design/bezier_generator.py`;
  tests `notebooks/bezier_generator_test.py`, `notebooks/bezier_superset_proto.py`.
- Fidelity gates: `notebooks/step0_fidelity.py`, `step3_fidelity_gate.py`,
  `step4_fea_check.py`; converged mesh = rotor 0.5 + airgap 0.5 mm.
- BO machinery: `machine_design/bezier_bo.py` (data-driven box, feasible sampler,
  feasibility-constrained acqf), `notebooks/run_bezier_live_mo.py` (qLogEHVI +
  DSP-GP), `notebooks/bezier_bo_smoke.py`.
- Bar / diagnostic: `notebooks/run_bar_converged.py`,
  `docs/tables/bar_converged.csv` (HV*_bar = 0.0892, 7-pt front, T_mean band
  ~1.8%).
- Pilot + analysis: `notebooks/bar_to_warmstart.py`, `reeval_front.py`,
  `plot_pilot.py`, `plot_geom_compare.py`; figures
  `docs/figures/step5_pilot2_vs_bar.png`, `front_geometry_compare.png`.
- License probe: `notebooks/probe_hpc.py`.
- Full chronology: `docs/PARAMETERISATION_v2.md` (living log).
