<title>Reluctance5f — H0/H1 separability study</title>

# Are machine design and current control separable in a 5-phase SynRM?

**Goal.** Decide whether the rotor geometry of a 5-phase synchronous reluctance
machine can be optimized *independently* of the dq1/dq3 current composition
(**H0**, design-then-control), or whether geometry and current setpoints must be
**co-optimized** (**H1**). This is the leaf's headline question.

---

## Problem setup

**Objective (both hypotheses, identical).** Maximize the FEA mean electromagnetic
torque

> maximize  `T_mean(geometry, Id1, Iq1, Id3, Iq3)`
> subject to  peak phase current ≤ **I_max = 10 A**, and barrier feasibility.

Torque is computed by ANSYS Maxwell 2D (`Design2`, 5-phase, 40-slot, 2 pole-pairs)
over **one full electrical period** (`Nper=1`) — the short `1/(2m)` window biases
torque in a dq3-dependent way, which would corrupt this very comparison.

**Geometry variables.** `HacklGenerator_SixLambdas` — 12 dimensions (3 inner +
3 outer barrier angles, 3 inner + 3 outer Bézier λ), 3 flux barriers per pole.

**Current variables — parameterized on the peak-current boundary (3 dims).**
The phase current is the imposed waveform `Im1·cos(θ+ε1) + Im3·cos(3θ+ε3)`. We
optimize:

| symbol | meaning | range |
|---|---|---|
| `φ1` | dq1 current angle (ε1) | 0 – 90° |
| `ρ`  | injection ratio Im3/Im1 | 0 – 0.5 |
| `Δ`  | relative dq3 phase (ε3 − 3φ1) | 0 – 360° |

The **peak is analytic and cheap** — it depends only on `ρ, Δ`:

> `peak = Im1 · max_θ |cos θ + ρ·cos(3θ + Δ)|`

so we set `Im1 = I_max / max_θ|·|` and **every candidate sits exactly on the 10 A
budget** (no FEA needed for the constraint; verified to 10.000 A in validation).
`φ1` only rotates dq1 and does not change the peak.

---

## The two procedures

**H0 — sequential (design-then-control).**
1. **Stage 1:** maximize torque over **geometry + φ1** with `ρ = 0` (pure 1st
   harmonic) → optimal 1st-harmonic rotor `G*`.
2. **Stage 2:** **freeze `G*`**, optimize the **currents (φ1, ρ, Δ)** under the
   peak constraint → `T_seq` at point **P0 = (G*, c*)**.

**H1 — local joint probe.** A trust-region joint BO over **geometry + currents**
in a small box around **P0**, seeded at P0. Result `T_h1`, gap **ΔT = T_h1 − T_seq**.

---

## Why the local probe is a rigorous H0/H1 test

At **P0**, stage 2 already made the currents optimal for the fixed geometry, so

> `∂T/∂c = 0` at P0.

There is no first-order torque left in the *current* directions. By the envelope
theorem, the only way the joint problem can beat H0 is by **moving geometry**, and

> `dT*/dG = ∂T/∂G` **evaluated at the dq3-inclusive currents `c*`**.

Stage 1 guaranteed `∂T/∂G = 0` at the dq3 = **0** currents. Therefore **ΔT
measures exactly the geometry × composition interaction** — "is it worth
re-shaping the rotor once you know you will inject 3rd harmonic?" — with nothing
else confounding it:

- **ΔT ≈ 0** (within the FEA noise floor) → P0 is a joint local optimum →
  **separable (H0):** optimize geometry once at the fundamental, tune control after.
- **ΔT > 0** → the optimal geometry depends on the dq1/dq3 split →
  **coupled (H1):** the current composition must be a design variable.

The geometry move `‖G_h1 − G*‖` and the injection `ρ` at the H1 optimum say
*which rotor features* the 3rd harmonic wants — the mechanism behind any H1.

**Note (sharper than the original AGENTS.md framing):** stage 2 already grants
the *current-side* dq3 benefit on the fixed geometry, so ΔT is purely the
geometry-adaptation term, not the dq3 benefit as a whole.

---

## Run configuration

- BO: single-objective, LogEI on a `SingleTaskGP` (standardized), q=1, resumable
  per-eval checkpoints in `results_h0h1/`.
- Budget (1 seed): stage1 24 init + 96 BO (13-dim); stage2 8 + 24 (3-dim);
  H1 20 + 30 (15-dim, trust region). ≈ 200 FEA.
- Cost: ~3.2 min / FEA (full period, 12-dim mesh) → stage 1 ≈ 6.5 h.
- Compute: bayes, ANSYS v242, `~/Public/MachineDesign-5f`, venv `venv_5f`.

## Caveats / honesty

1. **A null result is *local* separability**, not global. Insurance: an extra
   H1 restart, and re-running the whole study with a **richer geometry** if ΔT≈0
   (a too-poor geometry space can hide a real interaction; SixLambdas is already
   the richest Hackl generator).
2. **Optimizer noise** — to claim "two-stage is (not) worse," repeat over a few
   seeds and report the ΔT distribution, not a single number.
3. **FEA noise floor** is discretization-level (deterministic per geometry); we
   calibrate the "real gain" threshold (~0.5–1 %) by re-evaluating P0.
