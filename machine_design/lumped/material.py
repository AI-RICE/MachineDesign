"""Material constants for the lumped-reluctance solver.

Per `applications/ReluctanceDrive/CLAUDE.md` §11, all constants here come
from public sources cited in `REFERENCES.md`. **No FEA-fit constants.**
v1 uses linear iron (constant `μ_iron`); saturation `μ(B)` is a v2 task.
"""

from __future__ import annotations

from math import pi

# Vacuum permeability (exact, SI). Source: 2019 SI redefinition.
MU_0: float = 4.0e-7 * pi  # H/m

# Linear relative permeability for M350-50A (Cogent Power) in the unsaturated
# region (B < 1 T). Cited in REFERENCES.md. Real μ_iron(B) is non-linear and
# drops sharply above ~1.5 T; v1 ignores this, see CLAUDE.md §4 priority note.
MU_R_IRON_LINEAR: float = 1000.0

MU_IRON: float = MU_R_IRON_LINEAR * MU_0  # H/m

# Crude per-edge cross-section (perpendicular width × stack length).
# v1 uses type-specific defaults; v2 will compute per-edge geometry.
DEFAULT_PERP_WIDTH_M: float = 2.0e-3   # 2 mm — legacy fallback

# Per-edge-kind perpendicular widths (m). Order-of-magnitude estimates so
# that the relative airgap-vs-barrier-vs-iron reluctances reflect the real
# machine — what matters for Spearman is that barriers don't completely
# dominate over the airgap and vice versa.
EDGE_PERP_WIDTH_M: dict[str, float] = {
    "iron_yoke":     5.0e-3,   # yoke body, deep iron
    "iron_tooth":    1.5e-3,   # slot leakage path (narrow)
    "yoke_to_tooth": 5.0e-3,   # tooth body
    "iron_rotor":    3.0e-3,   # rotor iron channel midline
    "iron_surface":  1.0e-3,   # rotor surface rim (thin)
    "shaft_link":    8.0e-3,   # shaft region is large, parallel paths
    "airgap":        6.9e-3,   # one slot-pitch arc at airgap radius
    "barrier":      10.0e-3,   # typical barrier face perpendicular extent
}
