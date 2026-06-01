# Replication: Vanilla BO Performs Great in High Dimensions (Hvarfner et al., ICML 2024)

H-REPL replication of the **top-ranked** HD-BO method from
[`../../docs/METHODS.md`](../../docs/METHODS.md), validated on the paper's own
**simplest** benchmarks before we apply it to RadialSpline+FEA.

Paper: Hvarfner, Hellsten, Nardi, *"Vanilla Bayesian Optimization Performs Great
in High Dimensions"*, ICML 2024, arXiv:2402.02229 (PDF archived at
[`../../docs/refs/hvarfner2024_vanillahdbo.pdf`](../../docs/refs/hvarfner2024_vanillahdbo.pdf)).

## What the method is (their "DSP")

A one-line change to vanilla GP-BO: scale the GP **lengthscale prior** with the
dimensionality D. Paper Eq. 4:

> `ℓ_i ~ LogNormal(μ₀ + log(D)/2, σ₀)`, with **μ₀=√2, σ₀=√3** → the prior **mode
> scales as √D** (≈0.50 at D=6). Signal variance **fixed σ_f²=1**. Matérn-5/2
> ARD kernel, constant mean, standardized outputs, **LogEI** acquisition, MAP fit.

The claim: this alone makes vanilla BO competitive with / better than
specialised HD-BO (TuRBO, SAASBO, BAxUS…), with **no structural assumptions**.

## Why it matters for us

Our latent gate (`../../docs/experiments/`) already showed a plain GP+ARD matches
the low-D ceiling on re-encoded designs at 114-D. The DSP prior is the principled
reason *why* a global GP can work in high D — making this the natural headline
optimiser for RadialSpline. We reproduce it here to confirm we wire it correctly
and recover the paper's behaviour before trusting it on FEA.

## Files

| File | Role |
|---|---|
| `dsp_prior.py` | DSP prior + GP factories: `build_dsp_gp` (the method) and `build_default_gp` (the Γ(3,6) control). Prior implemented explicitly and cross-checked against BoTorch's shipped port. |
| `benchmarks.py` | Embedded Hartmann-6 / Levy-4 in an ambient D-cube (Fig. 5 tasks). |
| `run_replication.py` | LogEI BO loop; DSP vs default-Γ(3,6) vs random; resumable; per-seed traces in `results/`. |
| `test_dsp.py` | Unit tests: √D mode formula, BoTorch-port cross-check, GP build (σ_f²=1 vs ScaleKernel), benchmark optima + embedding. |
| `RESULTS.md` | Reproduced numbers vs the paper. |

## Run

```bash
.venv/bin/python -m pytest replications/vanilla_hdbo/test_dsp.py -q   # or: python test_dsp.py
.venv/bin/python replications/vanilla_hdbo/run_replication.py --func hartmann6 --dim 25 --seeds 5 --init 20 --iters 80
```

## Success criteria (pre-declared)

1. **Mechanism** (unit test): lengthscale-prior mode = `exp(√2−3)·√D`, ≈0.50 at
   D=6, and matches BoTorch's port. ✅ (5/5 tests pass)
2. **Behaviour** (Fig. 5): on Hartmann-6 / Levy-4 embedded in D≥25, the DSP prior
   reaches markedly **lower log-regret than the default-Γ(3,6) prior**, and DSP
   reaches the regret magnitude the paper reports (Hartmann-6 25D: log₁₀-regret
   ≈ −1.5…−2 by ~100 evals). See `RESULTS.md`.

Note: we use 20 Sobol init (paper uses 30) and fewer seeds (5 vs 20) for compute;
deviations are recorded in `RESULTS.md`. This is a qualitative/behavioural
reproduction of the simplest benchmark, not a bit-exact rerun.
