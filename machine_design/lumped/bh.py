"""B-H curve for M350-50A (Cogent Power) electrical steel.

Datasheet anchor points (low-field initial permeability through deep
saturation). Used by `saturation.solve_with_saturation` to update each
iron edge's local `μ_r` according to its flux density.

Source: M350-50A datasheet, Cogent Power Ltd. (now Tata Steel). Cited in
`machine_design/lumped/REFERENCES.md`. Anchor points are representative
of the published curves and are NOT fitted to FEA. Per CLAUDE.md §11,
the B–H curve is a structural input, not a tuning knob.
"""

from __future__ import annotations

import numpy as np


# Anchor points (B in Tesla, μ_r = B / (μ_0 · H) dimensionless).
# Below 0 T: linear initial permeability. Above ~1.5 T: knee, sharp drop.
_B_ANCHOR = np.array([0.0, 0.5, 1.0, 1.3, 1.5, 1.7, 1.9, 2.1, 2.5, 4.0])
_MU_R_ANCHOR = np.array([5000.0, 4500.0, 3500.0, 2200.0, 1200.0, 400.0, 120.0, 40.0, 10.0, 1.0])


def mu_r_bh(B_tesla: float) -> float:
    """Relative permeability of M350-50A at the given flux density (T)."""
    return float(np.interp(abs(B_tesla), _B_ANCHOR, _MU_R_ANCHOR))


def mu_r_bh_array(B: np.ndarray) -> np.ndarray:
    """Vectorised version of `mu_r_bh` for many edges at once."""
    return np.interp(np.abs(B), _B_ANCHOR, _MU_R_ANCHOR)
