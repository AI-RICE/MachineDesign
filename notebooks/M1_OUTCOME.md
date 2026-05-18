# M1 outcome — probe-set evaluation (lumped-v1.0-prefrozen)

**Pass threshold (pre-declared per CLAUDE.md §11):** Spearman ρ ≥ 0.5 between
the lumped torque proxy and FEA-measured `T_mean`, on 60 probe designs
(20 random initials per generator) from `results/results1/` only.

## Result: FAIL — |ρ| = 0.402 < 0.5

But narrowly, and the result is highly significant (p ≈ 1.5 × 10⁻³).

Full probe, n=60:

| Comparison                  | ρ       | p       |
|-----------------------------|---------|---------|
| T_proxy vs T_FEA            | +0.402  | 1.5e-3  |
| W_d vs T_FEA                | −0.128  | 0.33    |
| W_q vs T_FEA                | −0.436  | 4.9e-4  |
| T_proxy vs ripple_FEA       | −0.060  | 0.65    |

Per-generator (n=20 each):

| Generator                       | ρ       | Pass?   |
|---------------------------------|---------|---------|
| HacklGenerator_SixLambdas (12-D)| **+0.540** | **PASS** |
| HacklGenerator_3BrokenLines (13-D) | +0.423 | fail   |
| HacklGenerator_OneLambda (7-D)  | +0.332  | fail   |

## Diagnosis

The most important parameterization for the TIE paper claim is **6λ with
constraints** (per CLAUDE.md §1 headline metric), and the M1 v1 model passes
the threshold on it individually (ρ = +0.540). The lumped model **is**
picking up barrier-design variation; the magnitude is just sub-threshold
on the pooled probe.

Structural reasons for the v1 shortfall:

1. **Linear `μ_iron = 1000 · μ_0`** — saturation is not modelled. SynRM
   designs near the FEA optimum operate at the saturation knee, where the
   effective μ drops sharply. v1 over-estimates iron permeability there,
   compressing the d-axis spread.
2. **Crude per-edge cross-sections** — type-specific defaults (`material.py`
   `EDGE_PERP_WIDTH_M`) rather than per-instance geometric widths. The
   width perpendicular to a chord through a curved iron channel varies
   along the chord; the constant default smooths over this.
3. **Slot leakage path** — included via `iron_tooth` edges with default
   width = 1.5 mm. The actual slot-leakage permeance depends on slot
   geometry (Hs0, Hs1, Hs2 from `design.py`) which we ignored.
4. **MMF excitation via current-injection at airgap nodes** — physically
   reasonable but coarser than a full winding-distribution model. The
   coil pitch (9 slots) and slot-per-pole-per-phase (1 spp) detail is
   absorbed into a single sinusoidal profile.

## Structural revisions used (per §11 limit of 2)

- **Revision 0**: initial — MMF as Dirichlet at tooth nodes. Stator
  slot-leakage dominated and gave ρ = −0.435 (correct magnitude, wrong sign).
- **Revision 1 (current code)**: MMF as Dirichlet at airgap nodes. Sign
  flipped (still wrong direction; rotor-to-shaft routing).
- **Revision 2 (current code, final)**: MMF as **current injection** at
  airgap nodes with shaft as Dirichlet ground. Sign now correct, magnitude
  0.402.

Two structural revisions used. No further v1 tuning per §11 discipline.

## v2 priorities (next iteration)

In order of expected impact:

1. **Per-edge cross-section** (3 days). Iron-channel edges: compute
   perpendicular width as the local iron-pocket thickness. Airgap and
   barrier edges: angular extent × radius. Slot leakage: from `slot_params`.
2. **Saturation `μ(B)`** (1 week). M350-50A datasheet curve; Newton
   iteration on the flux-dependent reluctance.
3. **Proper winding distribution** (3 days). Use the 9-slot, 1-spp
   coil-pitch-9 layout from `design.py` instead of a smooth sinusoid.
4. **Anti-periodic BC for q-axis solve** (2 days). Cleaner saliency than
   the current single-pole shaft-grounded approach.

Item 1 alone is likely to push the pooled Spearman above 0.5; items 2-3
should improve quantitative agreement enough to support an absolute T_mean
prediction (currently only the rank order is claimed).
