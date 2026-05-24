"""Load a trained PFN checkpoint into a ready-to-use `PFNBoModel`.

Returns the model + its training-time z-score normalisation so callers
can convert between BO-space (real T units) and PFN-space (z-scored).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .library import load_library
from .model import PFNBoModel


@dataclass
class LoadedPFN:
    model: PFNBoModel
    device: torch.device
    generator_name: str
    input_dim: int
    library_path: Path
    library_sha256: str
    lumped_tag: str | None
    granularity_mode: str
    y_mean: float           # training-time z-score mean (over `T_proxy` at the
                            # training-time granularity weighting)
    y_std: float
    x_mean: np.ndarray | None  # per-dim x normalisation from training; None on legacy checkpoints
    x_std: np.ndarray | None


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(path: str | Path, device: torch.device | None = None) -> LoadedPFN:
    """Load a checkpoint saved by `machine_design.pfn.train`."""
    path = Path(path)
    device = device or _device()
    payload = torch.load(path, map_location=device, weights_only=False)

    model_cfg = payload["model_config"]
    model = PFNBoModel(**model_cfg).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    # Re-derive the training-time y-normalisation from the library file.
    # For GP-prior PFNs (§12.5.P4) there is no library; per-context y-norm at
    # inference (PFNSurrogate.from_loaded_with_real_Y) handles the scale, so
    # the stored y_mean/y_std are effectively unused but we must set defaults.
    prior_kind = payload.get("prior_kind", "lumped")
    lib_path_str = payload.get("library_path", "")
    if prior_kind == "gp" or lib_path_str in ("", "gp-prior-no-library"):
        lib_path = Path("gp-prior-no-library")
        y_mean = 0.0
        y_std = 1.0
    else:
        lib_path = Path(lib_path_str)
        library = load_library(lib_path)
        g = payload["granularity_mode"]
        if g == "random":
            y_all = np.concatenate(list(library.T_proxy.values()))
        else:
            y_all = library.T_proxy[g]
        y_mean = float(y_all.mean())
        y_std = float(y_all.std() + 1e-12)

    x_mean = payload.get("x_mean")
    x_std = payload.get("x_std")
    return LoadedPFN(
        model=model,
        device=device,
        generator_name=payload["generator_name"],
        input_dim=payload["input_dim"],
        library_path=lib_path,
        library_sha256=payload["library_sha256"],
        lumped_tag=payload.get("lumped_tag"),
        granularity_mode=payload.get("granularity_mode", "GP"),
        y_mean=y_mean,
        y_std=y_std,
        x_mean=np.asarray(x_mean, dtype=np.float32) if x_mean is not None else None,
        x_std=np.asarray(x_std, dtype=np.float32) if x_std is not None else None,
    )
