# References — lumped-reluctance prior

All numerical constants and curves used by the lumped-reluctance code in this
directory must be cited here, with a public source. The data-hygiene protocol
in `applications/ReluctanceDrive/CLAUDE.md` §11 forbids tuning these against
FEA-evaluated torques in `MachineDesign/results/`.

## Reference machine geometry

Constants in `geometry.py` come from `machine_design/design.py:set_parameters`
in the ICEM2026 project (Adam, Laksař, Šmídl, et al.). That code targets a
36-slot, 4-pole SynRM with:

- Stator gap diameter: 79 mm → rotor outer radius 39.5 mm minus 0.225 mm airgap
- Stator yoke diameter: 125 mm
- Airgap: 0.225 mm
- Shaft diameter: 25 mm → rotor inner radius 12.5 mm
- Stack length: 85 mm
- 36 stator slots, 2 pole pairs (4 poles)
- 1-layer winding, coil pitch = 9 slots
- 68 turns per coil
- Lamination: Cogent Power M350-50A

These values are reproduced here as documentation; the lumped code reads them
from `geometry.MachineSpec`.

## Lamination B–H curve

**M350-50A non-oriented electrical steel**, Cogent Power Ltd. (now part of
Tata Steel). Standard datasheet curve. Reproduction:

- B. Heller, V. Hamata, "Harmonic Field Effects in Induction Machines",
  Elsevier, 1977 — for analytical envelope.
- M350-50A datasheet, Cogent Power — anchor points (TODO: add B–H tabulation
  when saturation enters the model in M1).

## Three-phase winding model

Standard textbook treatment. Cited references:

- J. Pyrhönen, T. Jokinen, V. Hrabovcová, "Design of Rotating Electrical
  Machines", Wiley, 2nd ed., 2014 — Chapters 2 (winding distribution) and 7
  (magnetic circuit).
- T. A. Lipo, "Introduction to AC Machine Design", Wiley/IEEE, 3rd ed., 2017
  — Chapters 2–4.

## Reluctance / coenergy formulae

- Pyrhönen et al., Chapter 3 (reluctance of magnetic circuits with airgaps).
- Lipo, Chapter 2 (MMF distribution and Fourier decomposition).

## Hackl-style barrier parameterization

- A. Hackl, "Bayesian optimisation of synchronous reluctance machine barrier
  geometry", as cited in `applications/ReluctanceDrive/ICEM2026___Machine_design.pdf`.
- Polyline-level definitions are read directly from
  `machine_design/generators.py::HacklGenerator_*` (public code in the same
  repository).

---

**Adding a new constant:** open a PR that adds the numeric value AND its
reference here. PRs without a citation are rejected.
