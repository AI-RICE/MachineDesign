# M1 v3 outcome — probe-set evaluation (lumped-v3.0-prefrozen)

Single structural change vs v2: μ(B) saturation
(`machine_design/lumped/saturation.py` + `bh.py` for the M350-50A curve).
Each iron edge's permeability is updated iteratively from the local flux
density until ψ converges.

## Result: PASS — ρ = +0.771 (p = 5.6 × 10⁻¹³)

Trajectory:

| Version | Full probe ρ | Best per-gen | Worst per-gen | Pre-declared 0.5 |
|---------|-------------|---------------|----------------|-------------------|
| v1      | +0.402      | SixL +0.540   | OneL +0.332    | FAIL              |
| v2      | +0.633      | OneL +0.669   | 3BL  +0.430    | PASS              |
| **v3**  | **+0.771**  | **3BL +0.926**| SixL +0.508    | **PASS**          |

Per-generator (n=20 each):

| Generator                       | v2 ρ    | v3 ρ        | Δ        |
|---------------------------------|---------|-------------|----------|
| HacklGenerator_3BrokenLines (13-D)| +0.430| **+0.926**  | +0.50 ★ |
| HacklGenerator_OneLambda (7-D)  | +0.669  | **+0.723**  | +0.05    |
| HacklGenerator_SixLambdas (12-D)| +0.529  | **+0.508**  | −0.02    |

3BrokenLines gets the biggest lift — its barrier polylines have angular
discontinuities at the broken-line vertices, and saturation makes those
discontinuities matter (the iron channel cross-section narrows abruptly
near a vertex, driving up local B and triggering local saturation).

Auxiliary correlations:

| Comparison                  | v2 ρ    | v3 ρ        |
|-----------------------------|---------|-------------|
| W_d vs T_FEA                | +0.020  | **+0.542**  |
| W_q vs T_FEA                | −0.715  | **−0.725**  |
| T_proxy vs ripple_FEA       | −0.106  | **−0.305**  |

W_d, which was uncorrelated with T_FEA in v2, becomes a useful predictor
once saturation flattens its top-end (passes alone at p < 1e-5). Ripple
correlation also improves and reaches significance — the saturated model
captures some of the harmonic content driving T_ripple, even though the
solver still doesn't sweep rotor positions.

## v3 implementation

- `bh.py` — piecewise-linear M350-50A `μ_r(B)` table from datasheet
  anchor points (5000 at B=0, 1200 at B=1.5 T knee, 40 at B=2.1 T).
- `reluctance.compute_edge_reluctances(net, mu_iron_per_edge=…)` —
  accepts per-edge μ overrides for the iteration.
- `reluctance.edge_geometry_lengths(net)` — exposes
  `(length_iron_m, length_air_m, A_m2)` so the saturation loop can recover
  flux density without redoing polygon work each iteration.
- `saturation.solve_with_saturation(net, mmf_mode, mmf_amp=200, …)` —
  fixed-point iteration with under-relaxation (`relax=0.5`), 1e-3 tolerance,
  8-iteration cap. Calibrated MMF amplitude (200 A-turns ≈ the fundamental
  airgap MMF per pole for the SIMOTICS GP-VSD4000 reference machine at its
  rated current).

Cost: ~270 ms per design (2 saturated solves × ~135 ms each). 60-design
probe runs in ~36 s end-to-end including geometry + barrier generation.

## Structural revisions used: 1 of §11's per-freeze budget of 2

The probe passes well above threshold (0.77 vs 0.5), so no further
v3-budget revisions. The remaining v3 priorities from `M1_V2_OUTCOME.md`
(proper winding distribution, anti-periodic BC, T_ripple sweep) are
**deferred** — they're for further refinement, not for crossing the
discipline threshold.

## v4 priorities (lower urgency)

1. **T_ripple proxy** via rotor sweep over one electrical period —
   20-30 saturated solves per design (~6 s); needed if BO is to optimise
   ripple jointly with T_mean.
2. **Proper winding distribution** from `slot_params` — captures the
   q=3, coil-pitch-9 layout instead of pure sinusoidal MMF. Likely
   tightens the SixLambdas correlation (which was the only generator
   to slightly regress in v3).
3. **Anti-periodic BC** for the q-axis solve — cleaner than current
   shaft-as-Dirichlet; minor expected improvement.

None are blocking for the PFN-prior recipe. The v3 model gives strong
rank correlation (ρ ≈ 0.77 pooled, 0.93 on the trickiest parameterisation)
and is ready to serve as the **matched prior for PFN meta-training**.
