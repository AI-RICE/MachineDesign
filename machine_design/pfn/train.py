"""PFN-α training loop reading from a `LumpedLibrary` via `PriorSampler`.

One PFN per (parameterisation, output) per CLAUDE.md D6 (flavour A). v1
trains on `T_proxy` alone since v3 lumped exposes only that scalar. Each
training step samples a batch of in-context tasks at a random granularity
from the library, runs the transformer in PFN mode, and minimises the
binned (Riemann) NLL of the held-out target.

Checkpoint contents (per CLAUDE.md §11 data-hygiene protocol):

- `state_dict`        — model weights
- `model_config`      — architecture hyperparameters (so `make_model` is recoverable)
- `train_config`      — training hyperparameters (lr schedule, batch size, …)
- `library_path`      — relative path to the library `.npz`
- `library_sha256`    — content hash of that file
- `lumped_tag`        — git tag of the lumped solver that produced the library
- `granularity_mode`  — `'random'` or a single-granularity name
- `steps`, `final_loss`, `held_out_nll`
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .library import load_library
from .model import PFNBoModel
from .prior_sampler import PriorSampler


@dataclass
class TrainConfig:
    steps: int = 20_000
    batch_size: int = 64
    n_context_min: int = 16
    n_context_max: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 500
    seed: int = 0
    grad_clip: float = 1.0
    noise_std: float = 0.02
    val_frac: float = 0.1
    val_every: int = 1000


@dataclass
class ModelConfig:
    num_bins: int = 100
    y_low: float = -5.0
    y_high: float = 5.0
    ninp: int = 128
    nhead: int = 4
    nhid: int = 512
    nlayers: int = 4


def _sha256_of_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _device(override: str | None = None) -> torch.device:
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _lr_lambda(total_steps: int, warmup: int):
    def f(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return f


def _stack_batch(
    tasks: list,
    n_context: int,
    n_target: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert a list of PFNTask into PFN tensors.

    Returns:
      x_seq : (n_total, B, D)  with n_total = n_context + n_target
      y_ctx : (n_context, B)
      y_tgt : (n_target,  B)
    """
    B = len(tasks)
    D = tasks[0].x_context.shape[1]
    x_seq = np.zeros((n_context + n_target, B, D), dtype=np.float32)
    y_ctx = np.zeros((n_context, B), dtype=np.float32)
    y_tgt = np.zeros((n_target, B), dtype=np.float32)
    for b, t in enumerate(tasks):
        x_seq[:n_context, b, :] = t.x_context
        x_seq[n_context:, b, :] = t.x_target
        y_ctx[:, b] = t.y_context
        y_tgt[:, b] = t.y_target
    return (
        torch.from_numpy(x_seq),
        torch.from_numpy(y_ctx),
        torch.from_numpy(y_tgt),
    )


def train(
    library_path: Path,
    output_path: Path,
    train_cfg: TrainConfig,
    model_cfg: ModelConfig,
    granularity_mode: str = "random",
    log_every: int = 100,
    lumped_tag: str | None = None,
    device_override: str | None = None,
) -> None:
    """Run a full training schedule and save the checkpoint."""
    device = _device(device_override)
    print(f"Device: {device}", flush=True)
    print(f"Loading library: {library_path}", flush=True)
    library = load_library(library_path)
    print(f"  generator: {library.generator_name}, N={len(library)}, D={library.params.shape[1]}", flush=True)
    lib_sha = _sha256_of_file(library_path)
    print(f"  sha256: {lib_sha[:16]}…", flush=True)

    # Hold out a slice for validation NLL — never seen during training.
    n_val = max(8, int(train_cfg.val_frac * len(library)))
    n_train = len(library) - n_val
    if n_val < 4 or n_train < 4 * train_cfg.n_context_max:
        raise RuntimeError(
            f"library too small: n_train={n_train}, n_val={n_val}, "
            f"need n_train ≥ 4·n_context_max = {4 * train_cfg.n_context_max}"
        )
    rng = np.random.default_rng(train_cfg.seed)
    perm = rng.permutation(len(library))
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]
    print(f"  train: {n_train} rows, val (held-out): {n_val} rows", flush=True)

    def _slice_library(library, idx):
        # Lightweight in-memory slice — keep the dataclass interface.
        from copy import copy
        new = copy(library)
        new.params = library.params[idx]
        new.T_proxy = {g: arr[idx] for g, arr in library.T_proxy.items()}
        new.W_d_fine = library.W_d_fine[idx]
        new.W_q_fine = library.W_q_fine[idx]
        return new

    train_lib = _slice_library(library, train_idx)
    val_lib = _slice_library(library, val_idx)
    train_sampler = PriorSampler(train_lib, granularity=granularity_mode)
    val_sampler = PriorSampler(val_lib, granularity=granularity_mode)

    # Per-dimension X normalisation: PFNs are sensitive to input scale. The
    # default linear encoder over raw `params` (D values ranging from O(0.25)
    # to O(44) across dimensions) was empirically collapsing the model to the
    # marginal predictor. We z-score per dimension using the full library so
    # the encoder sees inputs with comparable magnitudes across all D.
    x_mean = library.params.mean(axis=0).astype(np.float32)
    x_std = (library.params.std(axis=0) + 1e-12).astype(np.float32)
    x_mean_t = torch.from_numpy(x_mean).to(device)
    x_std_t = torch.from_numpy(x_std).to(device)
    print(f"  x normalisation (per-dim, from library): mean={x_mean.tolist()}", flush=True)
    print(f"                                            std ={x_std.tolist()}", flush=True)

    # Model.
    model = PFNBoModel(**asdict(model_cfg)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=_lr_lambda(train_cfg.steps, train_cfg.warmup_steps)
    )

    n_target = 1
    losses: list[float] = []  # one entry per log_every steps (mean over window)
    val_history: list[tuple[int, float]] = []
    loss_accum: torch.Tensor | None = None  # device-side running sum, sync'd once per log_every
    loss_count = 0

    t0 = time.time()
    for step in range(1, train_cfg.steps + 1):
        n_context = int(rng.integers(train_cfg.n_context_min, train_cfg.n_context_max + 1))
        tasks = train_sampler.sample_batch(
            rng,
            batch_size=train_cfg.batch_size,
            n_context=n_context,
            n_target=n_target,
        )
        x_seq, y_ctx, y_tgt = _stack_batch(tasks, n_context, n_target)
        if train_cfg.noise_std > 0.0:
            y_ctx = y_ctx + train_cfg.noise_std * torch.randn_like(y_ctx)
        x_seq = x_seq.to(device)
        y_ctx = y_ctx.to(device)
        y_tgt = y_tgt.to(device)
        x_seq = (x_seq - x_mean_t) / x_std_t

        logits = model((None, x_seq, y_ctx), single_eval_pos=n_context)
        if logits.shape[0] == n_target:
            logits_tgt = logits
        else:
            logits_tgt = logits[n_context:]
        logits_tgt = logits_tgt.reshape(-1, model.num_bins)
        targets = y_tgt.reshape(-1)
        nll = model.criterion(logits_tgt, targets).mean()

        optimizer.zero_grad(set_to_none=True)
        nll.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        # Accumulate loss on-device; sync only at log boundaries. On MPS the
        # per-step `.item()` previously forced a host↔device copy every step,
        # dominating the wall clock.
        detached = nll.detach()
        loss_accum = detached if loss_accum is None else loss_accum + detached
        loss_count += 1

        if step % log_every == 0:
            recent = float((loss_accum / loss_count).item())
            losses.append(recent)
            loss_accum = None
            loss_count = 0
            elapsed = time.time() - t0
            lr_now = scheduler.get_last_lr()[0]
            print(f"  step {step:>6}/{train_cfg.steps}  loss={recent:.4f}  lr={lr_now:.2e}  "
                  f"elapsed={elapsed:.1f}s", flush=True)

        if step % train_cfg.val_every == 0 or step == train_cfg.steps:
            val_nll = _eval_held_out(
                model, val_sampler, rng, train_cfg, device, n_eval=64,
                x_mean_t=x_mean_t, x_std_t=x_std_t,
            )
            val_history.append((step, val_nll))
            print(f"    held-out NLL @ step {step}: {val_nll:.4f}", flush=True)

    final_loss = losses[-1] if losses else float("nan")
    final_val_nll = val_history[-1][1] if val_history else float("nan")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "model_config": asdict(model_cfg),
        "train_config": asdict(train_cfg),
        "library_path": str(library_path),
        "library_sha256": lib_sha,
        "lumped_tag": lumped_tag,
        "granularity_mode": granularity_mode,
        "generator_name": library.generator_name,
        "input_dim": int(library.params.shape[1]),
        "steps": train_cfg.steps,
        "final_train_loss": final_loss,
        "final_held_out_nll": final_val_nll,
        "val_history": val_history,
        "x_mean": x_mean,  # per-dim x normalisation, applied at training; surrogate must mirror
        "x_std": x_std,
    }
    torch.save(payload, output_path)
    print(f"\nSaved checkpoint: {output_path}")
    print(f"  final train loss: {final_loss:.4f}")
    print(f"  final held-out NLL: {final_val_nll:.4f}")


def _eval_held_out(
    model: PFNBoModel,
    sampler: PriorSampler,
    rng: np.random.Generator,
    cfg: TrainConfig,
    device: torch.device,
    n_eval: int = 64,
    x_mean_t: torch.Tensor | None = None,
    x_std_t: torch.Tensor | None = None,
) -> float:
    """Mean NLL across `n_eval` independent held-out tasks."""
    model.eval()
    losses: list[float] = []
    n_context = (cfg.n_context_min + cfg.n_context_max) // 2
    n_target = 1
    with torch.no_grad():
        for _ in range(n_eval):
            task = sampler.sample(rng, n_context=n_context, n_target=n_target)
            x_seq, y_ctx, y_tgt = _stack_batch([task], n_context, n_target)
            x_seq = x_seq.to(device)
            y_ctx = y_ctx.to(device)
            y_tgt = y_tgt.to(device)
            if x_mean_t is not None and x_std_t is not None:
                x_seq = (x_seq - x_mean_t) / x_std_t
            logits = model((None, x_seq, y_ctx), single_eval_pos=n_context)
            logits_tgt = logits[-n_target:].reshape(-1, model.num_bins)
            targets = y_tgt.reshape(-1)
            losses.append(float(model.criterion(logits_tgt, targets).mean().item()))
    model.train()
    return float(np.mean(losses))


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("library", type=Path, help="path to .npz library")
    ap.add_argument("--out", type=Path, required=True, help="output checkpoint .pt")
    ap.add_argument("--granularity", default="random",
                    choices=["random", "COARSE", "MEDIUM", "FINE"])
    # Train.
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise-std", type=float, default=0.02)
    ap.add_argument("--n-context-min", type=int, default=16)
    ap.add_argument("--n-context-max", type=int, default=64)
    # Model.
    ap.add_argument("--num-bins", type=int, default=100)
    ap.add_argument("--ninp", type=int, default=128)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--nhid", type=int, default=512)
    ap.add_argument("--nlayers", type=int, default=4)
    # Misc.
    ap.add_argument("--lumped-tag", default="lumped-v3.0-prefrozen")
    ap.add_argument("--device", default=None,
                    help="override device: 'cpu', 'mps', 'cuda'. Default: auto.")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    train_cfg = TrainConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        noise_std=args.noise_std,
        n_context_min=args.n_context_min,
        n_context_max=args.n_context_max,
    )
    model_cfg = ModelConfig(
        num_bins=args.num_bins,
        ninp=args.ninp, nhead=args.nhead, nhid=args.nhid, nlayers=args.nlayers,
    )
    train(
        library_path=args.library,
        output_path=args.out,
        train_cfg=train_cfg,
        model_cfg=model_cfg,
        granularity_mode=args.granularity,
        lumped_tag=args.lumped_tag,
        device_override=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
