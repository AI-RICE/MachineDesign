"""PFN-α model wrapper for the SynRM matched-prior recipe.

Wraps the upstream `pfns` package (`TableTransformer` +
`FullSupportBarDistribution`) into a single `nn.Module` whose forward
signature matches the convention used by `experiments/benchmarking/`.
Self-contained so the `MachineDesign` repo can be used standalone as long
as `pfns` is on the Python path (typically via `pip install -e
<PFN4BOrevisited>/src/pfn`).

For α (marginal × marginal, per CLAUDE.md §5), each output gets its own
PFN. v1 has one output (T_proxy) so one PFN suffices; if T_ripple lands
in v4, train a second PFN with the same architecture against ripple
targets.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from torch import nn


def _ensure_pfns_on_path() -> None:
    """Make sure `pfns` is importable.

    Search order:
      1. Already importable — do nothing.
      2. `MACHINE_DESIGN_PFNS_PATH` env var → prepend to `sys.path`.
      3. Walk up from this file until a `src/pfn` directory is found
         (works when MachineDesign is embedded in PFN4BOrevisited).
    """
    try:
        import pfns  # noqa: F401
        return
    except ImportError:
        pass

    env_path = os.environ.get("MACHINE_DESIGN_PFNS_PATH")
    if env_path:
        sys.path.insert(0, env_path)
        try:
            import pfns  # noqa: F401
            return
        except ImportError:
            pass

    here = Path(__file__).resolve()
    for ancestor in [here.parent, *here.parents]:
        candidate = ancestor / "src" / "pfn"
        if candidate.is_dir():
            sys.path.insert(0, str(candidate))
            try:
                import pfns  # noqa: F401
                return
            except ImportError:
                continue
    raise ImportError(
        "Could not locate the `pfns` package. Either `pip install -e <src/pfn>` "
        "or set MACHINE_DESIGN_PFNS_PATH=<path/to/src/pfn>."
    )


_ensure_pfns_on_path()

import math  # noqa: E402

import torch.nn.functional as F  # noqa: E402

from pfns.model.bar_distribution import FullSupportBarDistribution  # noqa: E402
from pfns.model.transformer import TableTransformer                  # noqa: E402


class GaussianHead(nn.Module):
    """Heteroscedastic-Gaussian output head, mirroring the interface of
    `FullSupportBarDistribution` (`forward(output, target) -> per-row NLL`,
    `.mean(output)`, `.variance(output)`).

    This is the Transformer-Neural-Process-style predictor: the network emits
    a per-target `(mean, raw_scale)` and the predictive is `N(mean, std^2)`
    with `std = softplus(raw_scale) + min_std`. Softplus (not exp of a
    log-variance) keeps the scale strictly positive and bounded in gradient
    early in training, which avoids the variance blow-up a log-σ head can hit.

    Unlike the bar head this predictive is necessarily unimodal — exactly the
    GP-shaped assumption we want when asking whether a Gaussian head tracks a
    GP posterior more faithfully than the 100-bin Riemann head.
    """

    def __init__(self, min_std: float = 1e-3) -> None:
        super().__init__()
        self.min_std = float(min_std)

    def _unpack(self, output: torch.Tensor):
        mu = output[..., 0]
        std = F.softplus(output[..., 1]) + self.min_std
        return mu, std

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mu, std = self._unpack(output)
        var = std ** 2
        return 0.5 * torch.log(2.0 * math.pi * var) + 0.5 * (target - mu) ** 2 / var

    def mean(self, output: torch.Tensor) -> torch.Tensor:
        mu, _ = self._unpack(output)
        return mu

    def variance(self, output: torch.Tensor) -> torch.Tensor:
        _, std = self._unpack(output)
        return std ** 2


class PFNBoModel(nn.Module):
    """Transformer-based PFN. The output head is either a binned (Riemann)
    `FullSupportBarDistribution` (`head="bar"`, the default) or a
    heteroscedastic Gaussian (`head="gaussian"`, TNP-style)."""

    def __init__(
        self,
        num_bins: int = 100,
        y_low: float = -5.0,
        y_high: float = 5.0,
        ninp: int = 128,
        nhead: int = 4,
        nhid: int = 512,
        nlayers: int = 4,
        head: str = "bar",
    ) -> None:
        super().__init__()
        self.head = head
        self.y_low = y_low
        self.y_high = y_high
        if head == "gaussian":
            out_dim = 2
            self.criterion: nn.Module = GaussianHead()
        elif head == "bar":
            out_dim = num_bins
            self.criterion = FullSupportBarDistribution(
                borders=torch.linspace(y_low, y_high, num_bins + 1)
            )
        else:
            raise ValueError(f"head must be 'bar' or 'gaussian', got {head!r}")
        # `num_bins` doubles as the decoder output width everywhere downstream
        # (train.py / surrogate.py reshape to (-1, model.num_bins)). For the
        # Gaussian head that width is 2; the criterion interprets the columns.
        self.num_bins = out_dim
        self.inner = TableTransformer(
            decoder_dict={"standard": (None, out_dim)},
            batch_first=False,
            ninp=ninp,
            nhead=nhead,
            nhid=nhid,
            nlayers=nlayers,
        )

    def forward(self, inp, single_eval_pos=None):
        """`inp` = `(style, x, y)` where:
        - `x` is `(n_total, batch, D)`
        - `y` is `(n_ctx, batch)` (context labels; PFN reads the first
          `single_eval_pos` rows as context and predicts the rest).
        - `style` may be `None`.
        """
        style, x, y = inp
        out = self.inner(x=x, y=y, style=style, only_return_standard_out=True)
        if isinstance(out, dict):
            out = out["standard"]
        return out
