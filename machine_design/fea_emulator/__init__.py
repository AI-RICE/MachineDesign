"""FEA emulator (v1, M4.5).

Per CLAUDE.md §6.5 this subpackage is **deliberately disjoint from
`machine_design/lumped/`**: a separate cross-section, separate file
layout, separate hash. The two must not import each other. Pre-commit
lint is recommended (a `grep` would do).

Role: stand in for ANSYS during Phase 2 development. Trained on the 7,500
FEA-evaluated designs in `results/results*/results_<gen>_<constrained>.npz`,
so by construction it is FEA-data-INFORMED. It must NEVER reach the lumped
prior or PFN training pipeline — that would be the L1 leakage protocol
forbids.
"""

# IMPORTANT: do not import from `machine_design.lumped` anywhere in this
# subpackage. See CLAUDE.md §6.5 ("CI-enforced isolation").

from .data import LoadedFEADesigns, load_fea_designs
from .model import FEAEmulator, cv_evaluate

__all__ = [
    "FEAEmulator",
    "LoadedFEADesigns",
    "cv_evaluate",
    "load_fea_designs",
]
