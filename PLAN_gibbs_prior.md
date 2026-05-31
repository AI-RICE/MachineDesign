# Plan: Gibbs (anisotropic Paciorek–Schervish) PFN prior

**Goal**: Train a PFN on a non-stationary kernel prior that strictly subsumes
the wide-Matérn-ARD prior currently used by C2, then test whether it beats
GP-EI (mean regret 0.008) on FEA-OOD BO. Motivated by the §6.7 implicit-kernel
finding that FEA's effective length scales span ℓ ∈ [0.6, 13.8] — strong
non-stationarity a stationary kernel cannot capture.

## The prior

For each PFN training task, sample independently:

1. **Per-dim length-scale field** (log-linear in x):

   $$\log\ell_d(x_d) = a_d + b_d \cdot \tilde{x}_d, \quad \tilde{x}_d = 2(x_d - \text{mid}_d)/\text{range}_d \in [-1, 1]$$

   with $a_d \sim \mathcal{N}(0, \sigma_a^2)$, $b_d \sim \mathcal{N}(0, \sigma_b^2)$.
   Initial ranges: $\sigma_a = 1.4$ (matches current wide GP prior), $\sigma_b \in \{0.5, 1.0, 1.5\}$
   sweep — the experiment.

2. **Outputscale and noise**: same log-Normal / log-Uniform draws as
   `GPPriorConfig` (`log_outputscale_std=0.7`, `log_noise ∈ [-10, -2]`).

3. **Kernel** — anisotropic Paciorek–Schervish (diagonal Σ), Matérn-5/2 flavor:

   $$k(x, x') = \sigma_f^2 \cdot \prod_{d=1}^D \sqrt{\frac{2\ell_d(x)\ell_d(x')}{\ell_d^2(x) + \ell_d^2(x')}} \cdot m_{5/2}(d_{\mathrm{eff}})$$

   where $d_{\mathrm{eff}}^2 = \sum_d 2(x_d - x'_d)^2 / (\ell_d^2(x) + \ell_d^2(x'))$ and
   $m_{5/2}(r) = (1 + \sqrt{5}r + 5r^2/3)\,e^{-\sqrt{5}r}$.

   $b_d = 0 \;\forall d$ recovers the stationary Matérn-ARD prior C2 was trained on.

4. **Sample y**: Cholesky of $K + \sigma_n^2 I$ with the existing adaptive-jitter loop.

5. **Per-task normalisation**: **CONTEXT-ONLY** (the leak-fix invariant — see §13 of
   ReluctanceDrive/CLAUDE.md). Do not include target y in the mean/std.

## Implementation phases + validation gates

Build phases 1–5 with **all unit tests green before any training**. Then run phase
6 gates on a **60k-step confirmation run** before launching the full 1M.

### Phase 1 — Length-scale field `ell_d(x_d)` (≈10 lines, 6 tests)

`tests/test_gibbs_kernel.py::TestLengthScaleField`:

1. Shape: `(N, D)` in → `(N, D)` out.
2. Stationary recovery: `b = 0` ⇒ `log_ell` constant along x_d.
3. Sign convention: `b > 0` ⇒ ℓ at x = +1 larger than at x = −1.
4. Magnitude: `a = 0, b = 1, x = +1` ⇒ ℓ = e ≈ 2.718.
5. Positivity: `ell` > 0 for all `|a|, |b| ≤ 3`, no overflow.
6. Vectorised matches per-point loop on 20 random inputs.

### Phase 2 — Paciorek–Schervish kernel matrix (≈30 lines, 8 tests)

`TestGibbsKernel`:

7. Symmetry: `K(X, X) == K(X, X).T` (atol = 1e-10).
8. Diagonal value: `diag(K(X, X)) ≈ outputscale` (normalising factor = 1 on diagonal).
9. **Stationary recovery (Matérn-5/2)**: `b = 0` and constant ℓ ⇒ kernel equals
   `_matern` in `gp_prior_sampler.py` with the same ℓ on 50 random (X, X') pairs.
10. Stationary recovery (RBF cross-check): swap to RBF; should match
    `outputscale · exp(-Σ_d (x_d-x'_d)² / (2 ell_d²))` analytically.
11. PSD: `eigvalsh(K + 1e-6 I) ≥ 0` for 30 random parameter draws.
12. Normalising factor bound: `√(2 ℓ ℓ' / (ℓ² + ℓ'²)) ∈ (0, 1]` at all pairs.
13. Cauchy–Schwarz: `|K(x, x')| ≤ √(K(x,x) K(x',x'))` on 100 random pairs.
14. Slow reference cross-check: 5-line scalar loop matches vectorised version
    on N = 8, D = 3.

### Phase 3 — y sampling (≈10 lines, 4 tests)

`TestSampling`:

15. Cholesky succeeds with the adaptive-jitter loop on 50 random configs at N = 64.
16. Empirical covariance: 5000 samples with fixed K → `cov(Y)` matches `K + noise²I`
    (max abs diff < 0.05).
17. Marginal variance: `var(y(x_i)) ≈ outputscale + noise²` across many task samples.
18. Stationary reduction matches `GPPriorSampler` empirical 2-point covariance (within MC error).

### Phase 4 — `GibbsPriorSampler` class (≈50 lines, 3 tests)

`TestSamplerAPI`:

19. Interface parity with `GPPriorSampler`: same `sample(rng, n_context, n_target, normalise)`
    signature, same `PFNTask` shapes.
20. **Normalisation is context-only** (regression test for the leak): with
    `normalise=True`, `corr(z_target, -sum_z_context) < 0.5` (was ~1.0 in the
    leaked sampler). 50 samples.
21. Determinism: same seed ⇒ same `(X, y)`.

### Phase 5 — CLI + training plumbing (2 smoke tests)

22. `python -m machine_design.pfn.train --prior gibbs --steps 100` on CPU: completes,
    loss finite, decreases from step 0.
23. Checkpoint round-trip: save a 100-step gibbs checkpoint, load via
    `load_checkpoint`, `PFNSurrogate.posterior` on a small `(X_ctx, y_ctx, X_query)`
    runs without error.

### Phase 6 — Scientific-validity gates (post-training, 4 tests)

Run on a **60k-step confirmation training** (≈40 min on bayes GPU) before committing
to the 1M run.

24. **Leak-test on Gibbs checkpoint** (re-use `/tmp/leak_test.py` template): with a
    fresh FEA context, PFN predictions across query points have
    `pred_std / true_std > 0.3` (not constant). Same check that caught the
    original leak; catches any new leak introduced by the Gibbs sampler.
25. `m4_pfn_vs_gp_indist.py`: in-distribution ρ ≥ 0.5 at 60k steps (matches
    B2/C2's 50k trajectory). If much lower, training is broken.
26. **Empirical non-stationarity check**: sample 500 Gibbs functions with
    `σ_b = 1.5`; bin x_d into halves; the local RMSE of second differences
    differs between halves consistent with the sign of `b_d`. Catches "I implemented
    something but b doesn't actually do anything."
27. **FEA-OOD BO benchmark** (`m5_bo_benchmark.py --target fea --checkpoint gibbs_60k.pt`):
    mean regret at minimum matches C2's 0.013; **target is to beat GP-EI's 0.008**.

## Decision tree after the 60k confirmation

- **All gates pass + (27) beats GP**: launch the full 1M run with the winning
  `σ_b`; ship as the headline FEA-OOD result.
- **All gates pass + (27) matches C2 (~0.013) but doesn't beat GP**: log-linear
  ℓ_d isn't enough non-stationarity. Escalate to hyper-GP $\log\ell(x)$ (full
  Gibbs) — about 50 more lines, similar test plan.
- **(24) fails** (predictions constant): leak in the Gibbs sampler — debug
  immediately, do not waste a 1M run.
- **(25) fails** (in-dist ρ low): training is broken (wrong lr, bad scaling,
  numerical instability) — fix before continuing.
- **(26) fails** (no empirical non-stationarity): the `b_d` parameter isn't
  taking effect — debug the sampler implementation.

## Implementation files

- `machine_design/pfn/gibbs_prior_sampler.py` (new, ≈150 lines)
- `tests/test_gibbs_kernel.py` (new, the unit tests above)
- `machine_design/pfn/train.py` (minor: add `gibbs` to `--prior` choices; route to `GibbsPriorSampler`)
- Optional `notebooks/m5_bo_benchmark.py` already covers the FEA-OOD test (no changes needed).

## Estimated effort

- Phases 1–5 with all unit tests: **3–4h** on Mac (mostly the kernel math + tests).
- Phase 6 confirmation training: **~40 min** on bayes (60k steps).
- Full 1M run (if 60k gates pass): **~10h** on bayes.

Total to a headline FEA-OOD number: ~14h elapsed, of which only ~4h is hands-on coding.
