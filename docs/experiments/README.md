# Experiments log — `newparam` line

Chronological record of gates and replications. Each entry: setup, result,
verdict. Replications follow **H-REPL** ([../README.md](../README.md)).

---

## E1 — Geometry gate (no FEA) — **PASS**

Script: [../../notebooks/radialspline_geometry_gate.py](../../notebooks/radialspline_geometry_gate.py)
Figures: `notebooks/RadialSpline_random.png`, `notebooks/RadialSpline_warmstart.png`

Checks that `RadialSplineGenerator` (D=114, N=3, K=18) produces valid,
plausible geometry and that warm-start round-trips the existing designs.

- **Random feasibility: 1000/1000 (100%)** with prior policy 2 (realistic-biased,
  heavy tails).
- **Warm-start round-trip:** all three Hackl families (OneLambda/SixLambdas/
  ThreeBrokenLines) re-encode to RadialSpline and decode back **50/50 feasible**,
  at **~0.35 mm** radial RMSE.
- Renders show wide nested arcs dipping toward the shaft at the d-axis, central
  rib (white gap at 45°) present by construction.

Verdict: the parameterisation is geometrically sound and faithfully contains the
known-good region.

---

## E2 — Latent-space viability gate (§11.2 / PARAMETERISATION §13 test 6)

Script: [../../notebooks/radialspline_latent_gate.py](../../notebooks/radialspline_latent_gate.py)
Data: re-encoded FEA designs from `../MachineDesign/results` (read-only),
OneLambda, 2500 designs. Torques used to **measure** only (VAE selected on
reconstruction — H-CITE/§13 hygiene).

Held-out test n=400, T_mean, best config per representation:

| Representation | RMSE | Spearman ρ |
|---|---|---|
| GP on native Hackl X (7-D) — *info ceiling* | 0.049 | +0.96 |
| **GP on raw RadialSpline X (114-D)** | **0.053** | **+0.96** |
| GBM on raw RadialSpline (114-D) | 0.105 | +0.92 |
| GP on VAE latent (β=0.01, best) | 0.154 | +0.87 |

Findings:
1. **Re-encoding is faithful & GP-friendly:** raw 114-D GP+ARD matches the
   native 7-D ceiling. The expected curse-of-dimensionality floor **did not
   appear** — ARD lengthscales absorb the redundancy (the known designs lie on a
   smooth low-D manifold).
2. **A static VAE latent is the *worst* representation** — it loses information
   (and posterior-collapses at β=1.0). For our always-valid decoder, the VAE
   bottleneck is not justified.
3. **Caveat (scope of the gate):** the only FEA we have is the *existing Hackl
   designs* (intrinsically 7/12/13-D). The richer off-manifold designs the
   parameterisation is *for* are untested. So "raw GP works" is established only
   on the known low-D manifold.

Verdict: parameterisation **viable** (faithful, GP-friendly). VAE/LOL-BO
deprioritised. Direct HD-BO on raw 114-D is the path — see
[../METHODS.md](../METHODS.md). The off-manifold value proposition needs a real
BO loop on the emulator/ANSYS to test.

---

## E3 — H-REPL: Vanilla-HDBO (Hvarfner et al. 2024, DSP) — **REPRODUCED**

Folder: [`../../replications/vanilla_hdbo/`](../../replications/vanilla_hdbo/)
(PDF: [`../refs/hvarfner2024_vanillahdbo.pdf`](../refs/hvarfner2024_vanillahdbo.pdf))

Reproduced the top-ranked HD-BO method on its own simplest benchmark (embedded
Hartmann-6), per H-REPL, before applying it to RadialSpline.

- **Mechanism (unit tests, 5/5):** lengthscale-prior mode `= exp(√2−3)·√D`
  (≈0.50 at D=6), scales as √D, matches BoTorch's shipped port exactly. DSP fixes
  σ_f²=1; control uses Γ(3,6)+ScaleKernel.
- **Behaviour** (final log₁₀-regret, 5 seeds, LogEI, 20 init):

  | Ambient D | DSP (√D prior) | default Γ(3,6) | random |
  |---|---|---|---|
  | 25  | −1.48 ± 0.81 | −2.04 ± 1.08 | −0.08 ± 0.28 |
  | 100 | **−1.28 ± 0.86** | **−0.00 ± 0.21** | −0.04 ± 0.17 |

  At D=25 DSP ≈ default (prior matters little at low D — matches paper). At D=100
  **the default prior collapses to random while DSP keeps optimizing** — the
  paper's central claim. Figures: `replications/vanilla_hdbo/results/regret_*.png`.

Verdict: the √D-scaled lengthscale prior works as claimed and is correctly wired
in our stack — empirical basis for adopting it as the headline optimiser on the
114-D RadialSpline space. Full write-up: `replications/vanilla_hdbo/RESULTS.md`.

## E4 — Application: RadialSpline DSP-GP-BO, Phase-2 shakedown (emulator oracle)

Script: [`../../notebooks/run_radialspline_bo.py`](../../notebooks/run_radialspline_bo.py)
Figure: `notebooks/RadialSpline_bo_results/radialspline_bo_OneLambda.png`

First application of the validated DSP strategy (E3) to the SynRM design: vanilla
global GP-BO with BoTorch's √D-scaled LogNormal prior + LogEI over the **114-D
RadialSpline box**, maximising T_mean. Oracle = GBM emulator on the re-encoded
Hackl designs (OneLambda). Repair decoder ⇒ **every box point feasible, no
rejection loop**; each candidate is decoded→re-encoded before scoring.

Best T_mean vs budget (emulator oracle):

| budget | DSP-GP-BO | random | DSP−random gap | figure |
|---|---|---|---|---|
| 60 iters (3 seeds, 20 init)  | 4.22 ± 0.06 (93.4%) | 4.16 ± 0.05 (92.2%) | 0.06 | `…/radialspline_bo_OneLambda.png` |
| **250 iters (2 seeds, 30 init)** | **4.36 ± 0.06 (96.6%)** | 4.15 ± 0.05 (91.8%) | **0.21** | `…/radialspline_bo_OneLambda_it250.png` |

(% of FEA on-manifold best = 4.52; seed-0 DSP@250 reached 4.43 = 98%.)

**Findings (honest — corrected after a budget check):**
1. **Machinery validated end-to-end** — DSP-GP-BO over 114-D + repair feasibility +
   decode/re-encode/emulator objective runs correctly.
2. **60 iters was too few** (≈0.7×D). A 250-iter run shows DSP-GP-BO keeps
   climbing toward the on-manifold best (96.6%, seed-0 98%) while random plateaus
   at ~92%, and the **DSP-over-random gap widens 3.5× with budget** (0.06→0.21).
   ⚠️ This **overturns the initial E4 reading** that the plateau was "oracle-bound,
   more iters won't help" — that was premature. The emulator carries enough signal
   for DSP-GP-BO to navigate 114-D to the known-good region.
3. **Encouraging for tractability:** DSP-GP-BO recovers ~97% of the known-good
   (low-D Hackl) optimum within **~250 evals in 114-D** — the *same* budget the
   existing low-D (≤13-D) sweeps used. The √D prior keeps the extra dimensions
   from wrecking sample efficiency; consistent with a low *effective* dimension
   (E2). Still **on/near-manifold**: the emulator caps at 4.52 and cannot reward
   genuinely off-manifold improvements, so whether RadialSpline can *exceed* 4.52
   remains the FEA question.

**Verdict:** strategy + machinery ✅ and budget-scaling confirmed. The open
scientific question (can the rich space beat the narrow one?) needs **live ANSYS
(Phase 3)** for real off-manifold torque; ~250+ evals appears sufficient for the
high-D BO to converge, which makes Phase 3 budget-realistic.

## E5 — H-REPL: SAASBO (Eriksson & Jankowiak 2021) + effective-dimension probe

Folder: [`../../replications/saasbo/`](../../replications/saasbo/). Uses BoTorch's
`SaasFullyBayesianSingleTaskGP` + NUTS; module/tests/probe added.

**Mechanism (unit tests, 3/3):** on embedded Hartmann-6 whose true active dims are
known, SAASBO assigns shorter median lengthscales to active axes — recovers 4/6
(D=20, n=40) and **6/6** (D=50) in the shortest-lengthscale set. Active-dim
recovery is the behaviour the method exists for. ✅

**Effective-dimension probe on the simulator (re-encoded RadialSpline, OneLambda,
n=150):** the headline outcome. SAASBO lengthscale spectrum shows a clear elbow at
**~10–12 active axes of 114** (figure `replications/saasbo/effdim_OneLambda.png`),
dominated by **barrier-0 outer control points near the q-axis end**, `theta_lo`,
and a few barrier-1 features.

Interpretation:
- OneLambda is *intrinsically 7-D*; SAASBO finds ~12 axis-aligned active dims —
  *more* than 7 because the 7 Hackl DOF are **rotated** relative to the local
  B-spline control-point axes. Empirical confirmation of the axis-alignment caveat
  (§ discussion): a **Chebyshev/modal radial basis** would likely compress the
  active set closer to 7 and make SAASBO sharper.
- **Budget implication:** effective dim ≈ 12 ⇒ ~10–20× ≈ **120–240 FEA evals**,
  consistent with E4 (DSP reached 97% by ~250 iters). So Phase-3 at a few-hundred
  evals is budget-realistic for OneLambda. (SixLambdas/ThreeBrokenLines, 12/13-D,
  will have higher effective dim — probe each before sizing their budgets.)
- Caveat: measured on the Hackl manifold (only torque data we have); the
  off-manifold effective dim is unknown (and is the point of the rich space).

**Elementary BO comparison (SAASBO vs DSP vs random, Hartmann-6 D=50, 10 init +
25 iters, 3 seeds):**

| method | log₁₀ regret | active-dim recovery | runtime |
|---|---|---|---|
| SAASBO | −0.27 ± 0.21 | **6/6 every seed** | ~220 s/run |
| DSP | **−0.71 ± 0.25** | — | ~8 s/run |
| random | −0.05 ± 0.25 | — | instant |

Reading: SAASBO's **subspace recovery is flawless (6/6)** — its defining behaviour,
reproduced. But on **regret at this small budget DSP beat SAASBO** (and at ~27×
lower cost — NUTS is expensive). This is consistent with the 2024 literature, where
DSP and SAASBO trade wins (Hvarfner Fig. 5). **Verdict:** mechanism ✅; SAASBO is
not a free regret win over the (newer, cheaper) DSP prior here.

**Division of labor (decision for this line):** use **SAASBO as the
effective-dimension *diagnostic*** (the ~12-active-dim probe that sizes the FEA
budget), and **DSP as the *optimizer*** in Phase 3 (competitive regret, ~27×
cheaper, no NUTS at D=114 in the BO loop).

**Basis ablation (E5 follow-up, no FEA) — hypothesis refuted.** Re-expressing each
boundary in a **Chebyshev (modal)** basis did **not** shrink the active set toward
7: active dims (ls<10) went **12 (B-spline) → 17 (Chebyshev)** — slightly *worse*
(`replications/saasbo/basis_ablation_{,.py}`, fig `basis_ablation_OneLambda.png`;
spectra nearly overlap). Corrected physics reading: the torque-relevant features
(d-axis channel widths) are **localized**, so the *local* B-spline basis is already
well axis-aligned, whereas a global modal basis spreads a localized feature across
many coefficients. **Decision: keep the B-spline basis.** Effective dim ≈12 stands;
Phase-3 budget ~150–250 evals unchanged.

> **⚠️ MESH CAVEAT (added after E9).** E7 and E8 were run at the default rotor
> mesh `max_length = 3 mm`, which **E9 shows is not converged**: re-sampling the
> *same* geometry swings torque ~0.13 N·m (~3%) at 3 mm, collapsing to ~0.02 at
> 0.5 mm, and the converged torque is ~0.12 N·m (~2.7%) *below* the 3 mm value.
> Consequently the **quantitative conclusions below are confounded by meshing**:
> the "RadialSpline −1.9% (E7) / −8.8% HV (E8) vs Hackl" gaps are within the
> 3 mm artifact. Honest restatement: **RadialSpline reaches torque/HV
> indistinguishable from Hackl within FEA pipeline noise** — not "worse". The
> *qualitative* findings (pipeline works, BO works, warm-start works) stand.
> Re-evaluation at the converged 0.5 mm mesh is in progress.

## E7 — Phase 3: LIVE ANSYS DSP-GP-BO over RadialSpline (OneLambda) — the real result

Runner: [`../../notebooks/run_radialspline_live.py`](../../notebooks/run_radialspline_live.py).
Recorded run: `results_radialspline_live/OneLambda_dsp_live_ws40/` (200 eval_*.npz
with full torque series + geometry, evals.csv, config.json, run.log;
`best_trajectory.png`). Isolated bayes venv (botorch+pyaedt) + git worktree +
AEDT v242 copy — pfn track untouched. **~20 s/eval, 200/200 ok, 0 failures.**

Setup: single-objective (maximise T_mean); **40 warm-start anchors** = re-encoded
Hackl OneLambda designs spread across Hackl T∈[2.06,4.52]; **160 DSP-GP-BO iters**.

| | T_mean (live ANSYS) |
|---|---|
| Warm-start best (re-encoded Hackl) | **4.297** |
| **DSP-GP-BO best (160 iters)** | **4.435** |
| OneLambda Hackl optimum (warm-start source) | 4.517 |
| Global Hackl best (SixLambdas; warm-start was OneLambda-only) | 4.557 |

**Findings:**
1. **Pipeline + strategy validated on real FEA.** DSP-GP-BO navigates the 114-D
   space and **improves +3.2% over the warm-start ceiling** (4.30→4.43); **87/160
   BO evals beat the ceiling** — a genuine optimisation signal, not noise.
2. **Re-encoding fidelity has a real cost.** The Hackl optimum (4.52) re-encodes
   to a best of 4.30 under live FEA — a **−0.22 N·m loss** from the B-spline
   refit + repair + rib. Warm-start anchors otherwise reproduce Hackl torque well
   (T≈2→2.0, etc.).
3. **RadialSpline did NOT beat the narrow parameterisation** on max T_mean:
   4.435 vs OneLambda 4.517 (**−1.9%**), and vs the global Hackl best 4.557
   (SixLambdas, **−2.7%**) — note the warm-start here was OneLambda-only (E8 pools
   all three). BO recovered most of the re-encoding handicap but not all.
4. **The emulator was vindicated-as-misleading.** E4's emulator suggested ~4.5
   was trivially reachable and rated wild designs ~4.0; live FEA showed wild
   designs ≈0.1 N·m (smoke) and a true ceiling capped by re-encoding. **Only the
   live run gives the honest answer** — the whole reason Phase 3 was necessary.

**Caveats / open questions:**
- **Single-objective only.** The best design's ripple is 23.9% (unoptimised). The
  multi-objective Pareto (T_mean vs ripple) is where a richer geometry more
  plausibly wins — **untested** (needs T_ripple-aware acquisition, E6).
- **Re-encoding loss is reducible** — a more faithful encoder / less aggressive
  repair / finer rib could lift the 4.30 ceiling toward 4.52, changing the verdict.
- OneLambda is the easiest case (7-D); SixLambdas/ThreeBrokenLines may differ.
- Not apples-to-apples: Hackl 4.52 came from a 250-eval *multi-objective* EHVI
  sweep; this was 200-eval *single-objective*.

**Verdict (per D7, a first-class negative result):** the live pipeline works and
DSP-GP-BO optimises the rich space effectively, but RadialSpline does **not** beat
the dedicated Hackl parameterisation on single-objective T_mean for OneLambda. The
value proposition now hinges on (a) reducing the re-encoding loss and (b) the
multi-objective / harder-parameterisation cases.

## E8 — Phase 3 MULTI-OBJECTIVE: pooled warm-start qLogEHVI (the decisive test)

Runner: [`../../notebooks/run_radialspline_live_mo.py`](../../notebooks/run_radialspline_live_mo.py)
(+ [`radialspline_reencode_mo.py`](../../notebooks/radialspline_reencode_mo.py), pooled N=6896).
Run: `results_radialspline_live/pooled_mo_live/` (200 evals, full recording;
`pareto_vs_hackl.png`). Objectives (T_mean ↑, T_ripple ↓); ModelListGP of two
DSP-prior GPs; qLogEHVI; **60 warm-start pooled across all 3 families** (1λ=15,
6λ=26, 3BL=19 — the pooled Pareto front + diversity) + 140 BO iters. 200/200 ok.

Hypervolume (ref=(1.957, −0.622)):

| Pareto front | HV |
|---|---|
| **Original Hackl library (real FEA — the true bar)** | **1.541** |
| Re-encoded warm-start (encoding-degraded) | 1.360 |
| RadialSpline live MO-BO | **1.405** |

**Findings:**
1. **BO improves over the (degraded) warm-start** front: HV 1.360 → 1.405 (+3.3%);
   all 6 final Pareto points are BO-discovered — qLogEHVI works on the rich space.
2. **But RadialSpline does NOT beat the original Hackl front: −8.8% HV** (1.405 vs
   1.541). `pareto_vs_hackl.png` shows the Hackl front (SixLambdas 4.42@2.7% …
   4.56@17.6%; ThreeBrokenLines 4.50@5.95%) **dominating** the RadialSpline front
   (4.17–4.38 @ 4.2–5.3%), shifted ~0.1–0.15 N·m left at equal ripple.
3. **The binding handicap is re-encoding fidelity, confirmed twice.** The
   re-encoded warm-start front (1.360) is already −12% below the original Hackl
   front (1.541) — *purely from the encode → repair → decode geometry change*
   (~0.35 mm boundary RMSE → ~9–12% HV, since torque is very sensitive to bridge/
   channel widths). BO recovered part (→1.405) but cannot reach 1.541 from a
   degraded start in 114-D.

**Verdict (D7, first-class negative result):** across single-objective (E7,
−1.9% vs OneLambda / −2.7% vs the global Hackl best **4.557, SixLambdas**) AND
multi-objective (E8, −8.8% HV), **the unified RadialSpline parameterisation does
not beat the dedicated Hackl parameterisations.** The optimiser (DSP-GP-BO,
qLogEHVI) and the pipeline work; the parameterisation's expressiveness is real;
but the **encoder/decoder fidelity loss dominates** — known-good designs degrade
~9–12% when re-encoded, and BO can't recover it. *The bottleneck is the
representation's round-trip fidelity, not BO or the parameterisation's reach.*

## E9 — FEA mesh convergence: the "fidelity loss" is largely a coarse-mesh artifact

Scripts: [`../../notebooks/confirm_2d_fidelity.py`](../../notebooks/confirm_2d_fidelity.py)
(A/B/C: Hackl vs 2-D-spline vs `r(θ)` re-encode),
[`../../notebooks/mesh_convergence.py`](../../notebooks/mesh_convergence.py)
(same geometry, native vs resample, mesh sweep). Figure:
`results_radialspline_live/mesh_convergence.png`.

**Trigger:** re-sampling the *same* Hackl curve (no shape change) swung FEA torque
~0.13 N·m at the default 3 mm rotor mesh — a discretisation artifact, not geometry.

**Convergence (OneLambda max-T), Hackl-native T_mean vs rotor mesh:**

| mesh (mm) | 3.0 | 1.5 | 0.75 | 0.50 | 0.35 | 0.25 |
|---|---|---|---|---|---|---|
| T_mean | 4.517 | 4.530 | 4.418 | 4.392 | 4.404 | 4.404 |
| same-shape artifact \|Δ\| | 0.131 | 0.127 | 0.044 | 0.018 | 0.017 | 0.037 |

**Findings:**
1. **Converged T_mean ≈ 4.40 N·m at mesh ≈ 0.5 mm.** The **3 mm value (4.517) is
   biased HIGH by ~0.12 N·m (~2.7%)** — coarse mesh fails to resolve the 0.5 mm
   bridges (note the non-monotonic drop only once mesh < bridge width).
2. **The same-geometry discretisation artifact collapses 0.13 → ~0.02 by 0.5 mm**
   → the E7/E8 "re-encoding loss" and the `r(θ)`-vs-2-D differences were largely
   meshing noise. A ~**0.5–1% residual FEA noise floor** remains even at fine mesh.
3. **The `r(θ)` "11 mm geometry error" mattered far less than its RMS implied;**
   conversely a low-RMS smooth 2-D fit can lose torque by *rounding corners*
   (it wrecked the low-ripple point) — geometry RMS is a poor torque proxy.

**Baseline comparison re-evaluated at 0.5 mm** (`reeval_converged.py`, 8
Pareto-front designs/family; fig `family_pareto_converged.png`):

| family | max-T 3 mm → 0.5 mm | HV 3 mm → 0.5 mm | mesh bias |
|---|---|---|---|
| OneLambda | 4.517 → 4.426 | 1.510 → 1.442 | −0.091 |
| SixLambdas | 4.557 → 4.452 | 1.538 → **1.472** | −0.105 |
| ThreeBrokenLines | 4.516 → **4.457** | 1.508 → 1.463 | −0.060 |

**The ranking does not survive mesh convergence.** On max-T, 3 mm said
SixLambdas > {OneLambda ≈ 3BL}; at 0.5 mm it **flips** to 3BL ≳ SixLambdas >
OneLambda — because the coarse-mesh bias is **family-dependent** (−0.06 for 3BL
vs −0.10/−0.11 for the smooth families). At converged mesh the three are within
**~1–2%** (≈ the ~1% noise floor): **SixLambdas ≈ ThreeBrokenLines, marginally
ahead of OneLambda**; on HV SixLambdas (1.472) ≳ 3BL (1.463) > OneLambda (1.442).
**Conclusion: at 3 mm the parameterisation ranking is not reliable; "SixLambdas
is clearly best" overstates it.** Worth flagging to the co-authors — parameterisation
comparisons need mesh-converged (~0.5 mm) FEA.

**Pipeline fix going forward:** `Design.compute` now takes `mesh_length`; default
BO/eval should use **~0.5 mm** so all future runs are converged (cost is modest,
~minutes/eval).

## Next

- **Settle the baseline** with the 0.5 mm Pareto-front re-eval (E9 in progress);
  report converged per-family HV with ~1% noise bars.
- **Re-do E7/E8 conclusions at 0.5 mm** (best designs + Hackl bar) before any
  RadialSpline-vs-Hackl claim.
- **Then** the representation question (2-D corner-preserving superset) — but only
  judged against *converged* FEA, since geometry RMS proved misleading.
- Switch BO default to mesh ≈ 0.5 mm; ~minutes/eval, still affordable.
