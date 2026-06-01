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

## Next

- **Phase 3 (live ANSYS)** — real off-manifold result; ~150–250 evals now looks
  sufficient for OneLambda. Outputs to bayes `~/Public/PFN/` (CLAUDE.md §6).
- ~~Basis ablation~~ — **done (negative)**: Chebyshev basis did not help; keep
  B-spline (see E5 above).
- **MORBO (multi-objective, E6)** stays queued; needs T_ripple re-encoded (E2
  cached only T_mean).
