"""PFN-specific scaffolding for the SynRM matched-prior recipe.

The lumped solver (`machine_design/lumped/`) is the parameterisation-agnostic
physical artifact; this subpackage wraps it into a prior over which a PFN
can meta-train (M2), plus the BoTorch-compatible surrogate that the BO
loop will call at inference time (M4).
"""

from .checkpoint import LoadedPFN, load_checkpoint
from .library import (
    LumpedLibrary,
    LumpedLibraryEntry,
    build_library,
    load_library,
)
from .model import PFNBoModel
from .prior_sampler import PFNTask, PriorSampler
from .surrogate import PFNSurrogate
from .train import ModelConfig, TrainConfig, train

__all__ = [
    "LumpedLibrary",
    "LumpedLibraryEntry",
    "build_library",
    "load_library",
    "PFNTask",
    "PriorSampler",
    "PFNBoModel",
    "TrainConfig",
    "ModelConfig",
    "train",
    "LoadedPFN",
    "load_checkpoint",
    "PFNSurrogate",
]
