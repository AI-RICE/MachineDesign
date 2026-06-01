# Replication results — Hvarfner et al. 2024 (DSP)

Independent reproduction on the paper's simplest embedded synthetics. Setup
deviations from the paper (recorded for honesty): **20 Sobol init** (paper: 30),
**5 seeds** (paper: 20), Matérn-5/2 ARD, analytic **LogEI**, MAP fit,
`raw_samples=256, num_restarts=4`. Metric: `log10(best − f_opt)`, f_opt=−3.32237.

## Mechanism check (unit tests) — ✅ PASS

`test_dsp.py` 5/5: lengthscale-prior mode `= exp(√2−3)·√D` (≈0.500 at D=6),
scales exactly as √D, and **matches BoTorch's shipped port**
(`get_covar_module_with_dim_scaled_prior`) to float precision. DSP fixes σ_f²=1
(bare Matérn); the control uses ScaleKernel + Γ(3,6).

## Behavioural check — Hartmann-6 embedded

Final `log10`-regret, mean ± std over 5 seeds:

| Ambient D | DSP (√D prior) | default Γ(3,6) | random | figure |
|---|---|---|---|---|
| 25 | −1.48 ± 0.81 | −2.04 ± 1.08 | −0.08 ± 0.28 | `results/regret_hartmann6_d25.png` |
| 100 | **−1.28 ± 0.86** | **−0.00 ± 0.21** | −0.04 ± 0.17 | `results/regret_hartmann6_d100.png` |

### Reading of D=25

- **BO works and is wired correctly:** both GP-BO variants reach `log10`-regret
  ≈ −1.5…−2 by ~80 iterations, vs random stuck near 0 — a ~1.5–2 decade gap.
- **DSP reaches the paper's reported magnitude** on Hartmann-6/25D (Fig. 5:
  ≈ −1.5…−2 by ~100 evals). ✅
- **DSP ≈ default-Γ(3,6) at D=25** (default marginally ahead, well within the
  n=5 variance). This is **consistent with the paper**: the prior scaling barely
  matters at low ambient dimension (their Fig. 6 shows the two priors identical
  at D=6). The DSP *advantage* is predicted to appear only as D grows — which is
  what the D=100 run tests.

### Reading of D=100 — the paper's headline, reproduced

- **The default-Γ(3,6) prior collapses to random search** (−0.00 vs random −0.04):
  in 100-D the standard prior yields an uninformative GP, exactly the failure the
  paper diagnoses (Fig. 5 / §4.2 boundary–uninformative-model issue).
- **DSP keeps optimizing** (−1.28; best seed −3.0), a ≈1.3-decade gap over both
  default and random. In `results/regret_hartmann6_d100.png` the default-Γ curve
  sits *on top of* random while DSP descends.
- Corroborating signal: default-prior runs finished in **4–5 s** (acquisition
  degenerates → trivial optimisation), DSP runs took **~100–140 s** (informative
  model → real acquisition work).
- Magnitude vs paper: their 20-rep / 200-iter Hartmann-100D DSP reaches ≈ −1.5;
  our 5-rep / 100-iter run reaches −1.28 (mean), trending the same way. The gap
  is consistent with fewer iterations/seeds and 20 vs 30 init.

## Verdict — ✅ REPRODUCED

| Claim | Status |
|---|---|
| Lengthscale-prior mode scales as √D (Eq. 4), ≈0.50 at D=6, matches BoTorch port | ✅ exact (unit tests) |
| DSP reaches paper-level regret on the simplest synthetic | ✅ (Hartmann-6 25D) |
| Prior choice barely matters at low ambient D | ✅ (DSP ≈ default at D=25) |
| **As D grows, default-prior BO degrades to random while DSP keeps working** | ✅ (D=100: default = random, DSP ≈ −1.3) |

**Conclusion:** the method works as claimed and is correctly wired in our stack.
The √D-scaled LogNormal lengthscale prior (BoTorch
`get_covar_module_with_dim_scaled_prior`, or our explicit `build_dsp_gp`) is the
piece that makes a *global* GP usable in high D — the empirical basis for adopting
it as the headline optimiser on the 114-D RadialSpline space.

## Reproduce

```bash
.venv/bin/python replications/vanilla_hdbo/run_replication.py --func hartmann6 --dim 25  --seeds 5 --init 20 --iters 80
.venv/bin/python replications/vanilla_hdbo/run_replication.py --func hartmann6 --dim 100 --seeds 5 --init 20 --iters 100
.venv/bin/python replications/vanilla_hdbo/plot_results.py --func hartmann6 --dim 25
.venv/bin/python replications/vanilla_hdbo/plot_results.py --func hartmann6 --dim 100
```

Optional next: D=300 / Levy-4 to widen the sweep; more seeds to tighten the bands.
