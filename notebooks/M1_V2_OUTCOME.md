# M1 v2 outcome — probe-set evaluation (lumped-v2.0-prefrozen)

**Single structural change vs v1**: per-edge geometric cross-section
(`machine_design/lumped/cross_section.py`) replacing v1's per-kind constants.

## Result: PASS — ρ = +0.633 ≥ 0.5

Full probe (n=60, same designs as v1 — `results1/`):

| Comparison                  | v1 ρ    | v2 ρ        | p (v2)  |
|-----------------------------|---------|-------------|---------|
| **T_proxy vs T_FEA**        | +0.402  | **+0.633**  | 5.6e-8  |
| W_d vs T_FEA                | −0.128  | +0.020      | 0.88    |
| W_q vs T_FEA                | −0.436  | **−0.715**  | 1.3e-10 |
| T_proxy vs ripple_FEA       | −0.060  | −0.106      | 0.42    |

Per-generator:

| Generator                       | v1 ρ    | v2 ρ        | v2 verdict |
|---------------------------------|---------|-------------|------------|
| HacklGenerator_OneLambda (7-D)  | +0.332  | **+0.669**  | **PASS**   |
| HacklGenerator_SixLambdas (12-D)| +0.540  | **+0.529**  | **PASS**   |
| HacklGenerator_3BrokenLines (13-D) | +0.423 | +0.430    | fail (just) |

The largest jump is on OneLambda — v1's worst parameterization is now v2's
best. The 7-D parameterization varies *mainly* via angular barrier
positions, which the per-edge geometric cross-section (radial flux area
proportional to angular extent × radius) picks up correctly. v1's constant
per-kind widths flattened this signal.

## What v2 changed

`machine_design/lumped/cross_section.py::edge_cross_section_m2` computes
the perpendicular cross-section per edge based on the edge's geometric
role:

- **Within-channel iron** (tangential): width = local iron-pocket radial
  thickness at the edge midpoint, derived from the bracketing barriers'
  crossings at that angle.
- **Airgap** (radial): width = one slot-pitch arc length at the airgap
  radius — each tooth couples to its angular share of airgap.
- **Barrier** (radial cross-channel): width = local barrier angular extent
  at the rail's angle × radius.
- **Surface ↔ channel** (radial): width = pole's angular share / `n_col`
  × surface radius.
- **Yoke ring, tooth slot leakage, tooth body, shaft link**: stator
  textbook widths, computed from `MachineSpec` (slot pitch, yoke height).

The remaining defaults (M350-50A linear `μ_r = 1000`, sinusoidal MMF
current injection at airgap nodes with shaft as ground) are unchanged
from v1.

## Diagnostic note: W_q dominates the saliency signal

`W_q vs T_FEA: ρ = −0.715` is *stronger* than `T_proxy vs T_FEA: ρ =
+0.633`. This is consistent with the q-axis reluctance being the more
sensitive measure of design quality — barriers' primary job is to *block*
q-axis flux, so q-axis coenergy carries most of the design-quality signal,
while d-axis coenergy is relatively flat (saturated by the iron channels'
intrinsic high permeance in v2's linear model). v3+ saturation would
flatten W_q further at high B and may make the W_d − W_q difference more
informative for absolute torque magnitude predictions.

## Structural revisions used in v2

**1 out of §11's per-freeze budget of 2.** No further revisions because
the probe already passes. v3 work (saturation, winding distribution,
anti-periodic BC) is for **absolute magnitude calibration** and is no
longer needed for the rank-correlation target.

## v3 priorities (deferred)

These would push from "rank-correlation passes" to "absolute torque is
quantitatively meaningful":

1. **`μ(B)` saturation** — M350-50A datasheet curve, Newton iteration on
   edge reluctances. Likely tightens correlation for high-torque designs
   that operate near the saturation knee.
2. **Proper winding distribution** from `slot_params` (9-slot, 1-spp,
   coil-pitch-9) instead of pure sinusoid.
3. **Anti-periodic BC** for cleaner q-axis solve (would remove the
   shaft-grounding artefact that currently routes some flux radially).
4. **T_ripple proxy** via rotor sweep over one electrical period
   (a separate loop of static solves, ~20 positions).

None of these are blocking for the PFN-prior recipe — the v2 model passes
the rank-correlation discipline and can serve as the **matched prior**
for PFN meta-training right now.
