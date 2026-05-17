"""Reference-machine geometry.

A dataclass describing the SynRM under study, with all radii / counts taken
from the public spec in `machine_design/design.py` (in turn from the
ICEM2026 reference machine). No ANSYS / pyaedt dependency.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MachineSpec:
    """Geometric spec of a SynRM. All radii in millimetres, angles in degrees."""

    # Radial geometry
    rotor_r_min: float          # shaft / rotor inner radius
    rotor_r_max: float          # rotor outer radius (just inside the airgap)
    airgap: float               # radial airgap thickness
    stator_yoke_r_outer: float  # stator outer radius

    # Topology
    n_slots: int                # number of stator slots
    pole_pairs: int             # number of pole pairs
    stack_length: float = 85.0  # axial stack length (mm)

    # Duck-typed adapter so the existing `generators.py` BarrierGenerator
    # accepts a MachineSpec wherever it expects a `Design`.
    @property
    def stator_r_inner(self) -> float:
        return self.rotor_r_max + self.airgap

    @property
    def n_poles(self) -> int:
        return 2 * self.pole_pairs

    @property
    def pole_pitch_deg(self) -> float:
        return 360.0 / self.n_poles

    @property
    def slot_pitch_deg(self) -> float:
        return 360.0 / self.n_slots

    @property
    def slots_per_pole(self) -> float:
        return self.n_slots / self.n_poles


# Reference machine from `machine_design/design.py:set_parameters`.
# (DiaStatorGap 79 mm, Airgap 0.225 mm, DiaShaft 25 mm, DiaStatorYoke 125 mm,
#  SlotNumber 36, PolePairs 2.)
REFERENCE_MACHINE = MachineSpec(
    rotor_r_min=25.0 / 2,
    rotor_r_max=79.0 / 2 - 0.225,
    airgap=0.225,
    stator_yoke_r_outer=125.0 / 2,
    n_slots=36,
    pole_pairs=2,
    stack_length=85.0,
)
