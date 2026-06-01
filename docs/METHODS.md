# HD-BO method survey for the RadialSpline line

Optimiser candidates for the 114-D `RadialSpline` parameterisation against
expensive FEA (budget ≈ 250–1000 evals). Ranked by expected suitability to
**our** setting:

- 114-D continuous box; an **always-valid repair decoder** (every box point
  decodes to a feasible design — so we do **not** need a generative model to
  enforce validity);
- empirically, **plain GP+ARD already matches the low-D ceiling at 114-D** on
  re-encoded known designs (see [experiments/](experiments/)) — the
  curse-of-dimensionality floor did not appear;
- low intrinsic dimension on known designs, but the space is meant to express
  much **richer off-manifold** designs (untested — no FEA there yet);
- no objective gradients; eventually **multi-objective** (T_mean, T_ripple).

Citations follow **H-CITE** ([README](README.md)): ✅ = authors/title/venue
externally verified; items under "to verify" are **not** yet checked and must
not be cited as fact.

---

## ⚠️ DGBFGP is not a BO method

**DGBFGP** = *"Bayesian Basis Function Approximation for Scalable Gaussian
Process Priors in Deep Generative Models"* (Balık, Sinelnikov, Ong, Lähdesmäki,
**ICML 2025**, OpenReview `dewZTXKwli`) ✅. It is a **VAE-with-GP-prior generative
model** for high-dimensional **time-series** (conditional generation /
forecasting), with a basis-function approximation for a scalable GP prior. It is
**not** an optimiser. It could only enter our pipeline as a *generative decoder*
inside a latent-BO loop — which we are not pursuing (we have a repair decoder).
Recorded so the name is not mistaken for a BO algorithm.

---

## Ranking

### Tier 1 — best fit, start here

**1. Vanilla BO with dimension-scaled lengthscale prior** ✅
*Hvarfner, Hellsten, Nardi — "Vanilla Bayesian Optimization Performs Great in
High Dimensions", ICML 2024, arXiv:2402.02229.*
Core idea: scaling the GP lengthscale prior with D removes the degeneracy that
hurts vanilla BO in high dimensions; standard BO then beats specialised HD-BO
methods, with **no structural restriction**. Fit: directly explains our empirical
result; continuous, gradient-free, ~trivial in BoTorch; lowest-risk baseline and
plausibly our headline optimiser. Single-objective (compose with qEHVI/qNEHVI
for MO).

**2. SAASBO** ✅
*Eriksson, Jankowiak — "High-Dimensional Bayesian Optimization with Sparse
Axis-Aligned Subspaces", UAI 2021, arXiv:2103.00349.*
Core idea: half-Cauchy sparsity prior on ARD lengthscales → automatic
relevant-axis selection in a fully-Bayesian GP. Fit: built for exactly our
regime (expensive, ≤1000 evals, low intrinsic dim). RadialSpline axes (θ's,
control radii) are semi-interpretable, so axis-aligned sparsity is plausible.
NUTS cost is fine at our budget.

### Tier 2 — strong; the trust-region line

**3. TuRBO** ✅
*Eriksson, Pearce, Gardner, Turner, Poloczek — "Scalable Global Optimization via
Local Bayesian Optimization", NeurIPS 2019, arXiv:1910.01739.*
Local GP in an adaptive trust-region box; robust to hundreds of dims;
gradient-free. Fit: gold-standard HD default; pairs cleanly with the
always-valid decoder. Single-objective; robustness fallback if vanilla degrades
off-manifold.

**4. MORBO** ✅ — *the multi-objective endgame*
*Daulton, Eriksson, Balandat, Bakshy — "Multi-Objective Bayesian Optimization
over High-Dimensional Search Spaces", UAI 2022, arXiv:2109.10964.*
Multi-objective TuRBO: multiple coordinated trust regions, hypervolume-driven;
demonstrated on a 222-parameter design problem. Fit: the natural method for
(T_mean, T_ripple) at 114-D.

**5. BAxUS** ✅ *(venue/year: NeurIPS 2022; arXiv posted 2023 — DBLP confirm pending)*
*Papenmeier, Nardi, Poloczek — "Increasing the Scope as You Learn: Adaptive
Bayesian Optimization in Nested Subspaces", arXiv:2304.11468.*
Nested random subspace embeddings that **grow** with data + TuRBO; no need to
guess the intrinsic dimension; strong theory. Fit: principled when an active
subspace exists but its dimension is unknown — true for us. Continuous.

### Tier 3 — possible, with caveats (to verify before citing)

- **Additive-GP BO** — objective = sum of low-D group terms. Caveat: saturation /
  flux-sharing likely couples barriers, breaking additivity. Speculative.
- **REMBO / ALEBO / HeSBO** — optimise in a random **linear** embedding. Caveat:
  linear-subspace assumption; our torque-relevant manifold (geometry + repair) is
  nonlinear. Largely superseded by BAxUS / SAASBO; cite as lineage.

### Tier 4 — deprioritised / document why-not

- **LOL-BO** ✅ *(Maus, Jones, Moore, Kusner, Bradshaw, Gardner — NeurIPS 2022,
  arXiv:2201.11872)*, with successors **CoBO** and **InvBO**. Built for
  *discrete/structured* spaces (molecules) where a generative model is required
  to enforce validity. **Our always-valid decoder removes that need**, and the
  latent gate showed a static VAE *loses* information ([experiments/](experiments/)).
  Status: considered, deprioritised — validity is handled by the decoder, not a VAE.
- **Bounce** — combinatorial/mixed spaces; ours is continuous. Cite as mixed-space SOTA.
- **GIT-BO / BOIDS / Adaptive Linear Embedding** — 2024–25 watch-list. GIT-BO uses
  tabular foundation models (TabPFN-like) for HD-BO — intriguing given the parent
  project's PFN thread, but new/unproven. Monitor.

### Framing references ✅

- **Binois, Wycoff — "A Survey on High-dimensional Gaussian Process Modeling with
  Application to Bayesian Optimization", ACM TELO 2(2):1–26, 2022,
  DOI:10.1145/3545611, arXiv:2111.05040.** The right *continuous* HD-BO survey
  (variable selection, additive, embeddings) — use to position our comparison.
- González-Duque, Michael, Bartels, Zainchkovskyy, Hauberg, Boomsma — *"A survey
  and benchmark of high-dimensional Bayesian optimization of discrete sequences"*,
  NeurIPS 2024 D&B, arXiv:2406.04739 ✅. **Discrete-sequence (molecule/protein)
  focused** — tangential to our continuous problem; note, don't lean on it.

---

## To verify (candidates — NOT yet externally checked)

Per H-CITE, confirm authors/title/venue via Crossref/arXiv/DBLP before any of
these are cited as fact:

- REMBO — Wang et al., "Bayesian Optimization in a Billion Dimensions via Random
  Embeddings", JAIR 2016 (orig. IJCAI 2013).
- HeSBO — Nayebi, Munteanu, Poloczek, "A Framework for Bayesian Optimization in
  Embedded Subspaces", ICML 2019.
- ALEBO — Letham, Calandra, Rai, Bakshy, "Re-Examining Linear Embeddings for
  High-Dimensional Bayesian Optimization", NeurIPS 2020.
- Additive — Kandasamy, Schneider, Póczos, "High Dimensional Bayesian Optimisation
  and Bandits via Additive Models", ICML 2015.
- Bounce — Papenmeier, Nardi, Poloczek, NeurIPS 2023, arXiv:2307.00618 (have id;
  authors not API-verified).
- CoBO — Lee et al., "Advancing Bayesian Optimization via Learning Correlated
  Latent Space", NeurIPS 2023.
- InvBO — Chu et al., 2024 (latent inversion).
- GIT-BO — arXiv:2505.20685 (2025). BOIDS — arXiv:2412.12918 (2024).

---

## Replication plan (H-REPL)

Before trusting any method on RadialSpline+FEA, reproduce it on **its own
paper's elementary setup** and confirm the qualitative claim, logging the result
under [experiments/](experiments/). Order matches the ranking:

| Method | Elementary setup to reproduce first | Claim to recover | Code |
|---|---|---|---|
| Vanilla-scaled GP | high-D synthetic (e.g. Hartmann/Branin embedded), default vs D-scaled prior | scaled prior ≫ default in high-D | BoTorch (priors) |
| SAASBO | BoTorch SAASBO tutorial / their Hartmann-in-high-D | recovers sparse active dims; beats vanilla at small n | BoTorch tutorial |
| TuRBO | 100-D Ackley/Rastrigin (paper benchmark) | TuRBO ≫ global GP-EI in high-D | BoTorch TuRBO tutorial |
| MORBO | released synthetic (e.g. DTLZ / their benchmark) | HV ≫ baselines at high-D MO | `facebookresearch/morbo` |
| BAxUS | benchmark with a known active subspace | finds subspace without guessing its dim | authors' release |

Only after the elementary check passes do we wire the method to the
`RadialSplineGenerator` pool / FEA emulator.
