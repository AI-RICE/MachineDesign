"""Single-objective PFN-EI on live ANSYS, paired against Sadda's GP-EHVI trace.

The published GP-EHVI baseline in `/home/sadda/Projects/MachineDesign/results/`
(read-only, see CLAUDE.md §3) already supplies the head-to-head GP numbers
for our HV comparison. We don't need to rerun GP — we only need PFN-EI on
the same problem with the same 50 initials, then compare trajectories.

Single-objective:
- Surrogate: PFNSurrogate wrapping the per-parameterisation T_mean PFN
  trained at lumped-v3.0-prefrozen (see machine_design/pfn/train.py and the
  fixed pipeline: per-task y-norm, per-dim x-norm, per-context y-norm at
  inference).
- Acquisition: qLogExpectedImprovement on T_mean (Y[:, 0]).
- Initials: identical to Sadda's seed-1 baseline (results/metadata.csv +
  results/design_*.pkl), mirrored under our working tree.
- Per-iteration: rejection-sample candidates for feasibility (matching
  run_optimization.py), evaluate via live ANSYS Maxwell2d; record full
  (T_mean, -ripple/100) Y for post-hoc HV comparison even though only the
  T_mean head drives acquisition.

Output (resumable):
- `sweeps_pfn_live/results_<method>_False.npz`  — train_X (raw params), train_Y (2D).
- `sweeps_pfn_live/<method>_history.csv`        — per-iter best-T, current-HV, n_feasible_attempts.

If a future T_ripple PFN materialises, the full multi-objective EHVI variant
(`run_optimization_pfn_live_mo.py`) becomes feasible; this script is the
single-obj stepping stone.
"""

from __future__ import annotations

import os
import time
import csv
from pathlib import Path

import numpy as np
import torch
from botorch import fit_gpytorch_mll  # noqa: F401  (kept for symmetry with GP path)
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.transforms import normalize, unnormalize

from machine_design import (
    HacklGenerator_OneLambda,
    init_points,
    load_design,
    objective,
    objective_transform,
)
from machine_design.pfn import PFNSurrogate, load_checkpoint


# PFN was trained at float32 and surrogate.py.posterior() casts inputs to
# float32. We deliberately do NOT set the BoTorch-recommended float64 default
# here: doing so causes the PFN to be re-materialised at float64 (some inner
# positional / buffer tensors propagate the default dtype) while inputs are
# still float32, producing "mat1 and mat2 must have the same dtype" errors
# during qLogEI's acquisition forward pass. Without input_transform=Normalize
# the BoTorch float64 recommendation does not buy us much here, since the
# PFN's internal x_mean/x_std-based normalisation handles scale.
torch.set_default_dtype(torch.float32)


# ---------------------------------------------------------------------------
# Configuration — mirrors run_optimization.py defaults so the comparison is
# structurally identical (same EHVI machinery, same feasibility filter, same
# n_evals=250 budget, same num_cores=4 ANSYS parallelism).
# ---------------------------------------------------------------------------
aedt_version = "2024.1"
n_evals = 250
r_stator_end = 0.7
offset = 0.7 / 2
num_cores = 4
batch_size = 4
max_candidate_tries = 10
objective_fallback = {"torque": 1.0, "ripple": 40.0}
ref_no_cons = {"torque": 4.0, "ripple": 30.0}  # HV reference point (no-cons cell)

project_name = "SynRM_test"
design_name = "Design01"

path_data = os.path.join(os.getcwd(), "data")
root_init = "results"
out_dir = Path("sweeps_pfn_live")
out_dir.mkdir(parents=True, exist_ok=True)

# Which parameterisation we run (paired with its T_mean PFN checkpoint).
GENERATOR_CLS = HacklGenerator_OneLambda
pfn_checkpoint_path = Path("checkpoints/OneLambda_pfn.pt")
use_constraints = False                       # match GP cell to compare against
output_name = out_dir / "results_HacklGenerator_OneLambda_False.npz"
history_csv = out_dir / "HacklGenerator_OneLambda_history.csv"

file_name_aedt = f"{path_data}/{project_name}.aedt"


def _print(msg: str) -> None:
    print(msg, flush=True)


_print(f"=== PFN-EI live ANSYS ===")
_print(f"  generator:    {GENERATOR_CLS.__name__}")
_print(f"  constraints:  {use_constraints}")
_print(f"  PFN ckpt:     {pfn_checkpoint_path}")
_print(f"  n_evals:      {n_evals}  (50 initials + 50 BO iters at batch={batch_size})")
_print(f"  out dir:      {out_dir.resolve()}")

design = load_design(file_name_aedt, project_name, design_name, aedt_version)
generator = GENERATOR_CLS(design, r_stator_end, offset=offset)
method = generator.__class__.__name__

ref_no_cons_torque, ref_no_cons_ripple = objective_transform(ref_no_cons["torque"], ref_no_cons["ripple"])
objective_fallback_tuple = (objective_fallback["torque"], objective_fallback["ripple"])
ref_point = torch.tensor([ref_no_cons_torque, ref_no_cons_ripple], dtype=torch.float32)

if output_name.exists():
    _print(f"  resuming from {output_name}")
    data = np.load(output_name)
    train_X_raw = torch.from_numpy(data["train_X"])  # raw param space
    train_Y = torch.from_numpy(data["train_Y"])
else:
    _print(f"  loading initials from {root_init}/")
    train_X_raw, train_Y = init_points(root_init, method)
    _print(f"  initials: n = {len(train_X_raw)}")

# bounds in raw param space + normalised [0,1]^d helpers.
bounds = torch.from_numpy(np.vstack(generator.bounds))
bounds_normalized = normalize(bounds, bounds)
D = bounds.shape[1]

# Load PFN once; reused each step via from_loaded_with_real_Y.
loaded = load_checkpoint(pfn_checkpoint_path)
# torch.set_default_dtype(float64) above causes the PFN's nn.Linear layers
# to be created with float64 weights. The PFN was trained at float32 and
# surrogate.py.posterior() casts inputs to float32, producing a
# "mat1 and mat2 must have the same dtype" mismatch. Force the model
# back to float32 to match the inference path.
loaded.model = loaded.model.to(torch.float32)
_print(f"  PFN loaded: D={loaded.input_dim}, x_mean/std present={loaded.x_mean is not None}, dtype=float32")


def _objective_lambda(Xs_norm):
    """Wrap `objective` so it receives [0,1]^d candidates and ANSYS returns 2D Y."""
    return objective(
        Xs_norm, design, generator, bounds, num_cores,
        objective_fallback=objective_fallback_tuple,
    )


def _penalty_objective(n_penalty: int) -> torch.Tensor:
    obj = objective_transform(None, None, objective_fallback=objective_fallback_tuple)
    return torch.tensor(obj, dtype=torch.float32).repeat(n_penalty, 1)


def _hv(Y: torch.Tensor) -> float:
    mask = is_non_dominated(Y)
    if not mask.any():
        return 0.0
    hv = Hypervolume(ref_point=ref_point)
    return float(hv.compute(Y[mask]))


def _write_history(step: int, n: int, best_T: float, hv: float, n_acq: int, n_unif: int) -> None:
    new = not history_csv.exists()
    with history_csv.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["step", "n_evals", "best_T", "HV", "n_acq_calls", "n_uniform_tries"])
        w.writerow([step, n, f"{best_T:.6f}", f"{hv:.6f}", n_acq, n_unif])


def _is_feasible(x_np: np.ndarray) -> bool:
    params = generator.X_to_params(x_np)
    generator.set_parameters(params)
    barriers = generator.generate_barriers()
    barriers = generator.split_barriers(barriers)
    return bool(generator.feasible_barriers(barriers))


step = 0
while len(train_X_raw) < n_evals:
    step += 1
    t0 = time.time()
    _print(f"\n--- iter {step}  (current n = {len(train_X_raw)} / {n_evals}) ---")

    # PFN surrogate is constructed in RAW param space (it normalises internally
    # via the per-dim x_mean/x_std baked into the checkpoint; see CLAUDE.md
    # pipeline-fix discussion).
    train_Y_T = train_Y[:, 0:1]  # only T_mean drives PFN acquisition
    surr = PFNSurrogate.from_loaded_with_real_Y(loaded, train_X_raw, train_Y_T)
    best_T = float(train_Y_T.max().item())
    # Convert best_f to surrogate's per-context normalised space.
    best_f_norm = (best_T - surr.y_mean) / surr.y_std
    acq = qLogExpectedImprovement(model=surr, best_f=best_f_norm)

    # Rejection-sampling for feasibility, matching run_optimization.py's
    # batch-of-q approach. We work in RAW bounds because the PFN is in raw
    # space; the surrogate's internal x-norm handles scale.
    candidates_feasible: list[torch.Tensor] = []
    candidates_infeasible: list[torch.Tensor] = []
    n_acqf_calls = 0
    for _ in range(max_candidate_tries):
        cand, _ = optimize_acqf(
            acq_function=acq, bounds=bounds.to(torch.float32),
            q=batch_size, num_restarts=10, raw_samples=256,
        )
        n_acqf_calls += 1
        for c in cand:
            c64 = c.to(torch.float32)
            if _is_feasible(np.asarray(c64.cpu().numpy(), dtype=np.float64)):
                candidates_feasible.append(c64)
            else:
                candidates_infeasible.append(c64)
            if len(candidates_feasible) >= batch_size:
                break
        if len(candidates_feasible) >= batch_size:
            break

    # Fallback to uniform-feasible if optimize_acqf came up short.
    n_uniform_tries = 0
    while len(candidates_feasible) < batch_size and n_uniform_tries < 200:
        n_uniform_tries += 1
        lo, hi = generator.bounds
        c = np.asarray(np.random.default_rng().uniform(lo, hi), dtype=np.float64)
        if _is_feasible(c):
            candidates_feasible.append(torch.from_numpy(c))

    n_missing = batch_size - len(candidates_feasible)
    if candidates_feasible:
        cand_raw = torch.stack(candidates_feasible)
        # `objective(...)` expects normalised candidates; normalise raw → [0,1].
        cand_norm = normalize(cand_raw, bounds)
        new_Y_all = _objective_lambda(cand_norm)
    else:
        cand_raw = torch.empty((0, D), dtype=torch.float32)
        new_Y_all = torch.empty((0, 2), dtype=torch.float32)
    if n_missing > 0:
        cand_raw = torch.cat([cand_raw, torch.stack(candidates_infeasible[:n_missing])], dim=0)
        new_Y_all = torch.cat([new_Y_all, _penalty_objective(n_missing)], dim=0)

    train_X_raw = torch.cat([train_X_raw, cand_raw])
    train_Y = torch.cat([train_Y, new_Y_all])

    best_T_now = float(train_Y[:, 0].max().item())
    hv_now = _hv(train_Y)
    elapsed = time.time() - t0
    _print(f"  added {len(cand_raw)} candidates  (n_acq={n_acqf_calls}, n_unif={n_uniform_tries}, missing={n_missing})")
    _print(f"  n now: {len(train_Y)}    best T = {best_T_now:.4f}    HV = {hv_now:.4f}    iter wall = {elapsed:.1f}s")

    np.savez(output_name, train_X=train_X_raw.numpy(), train_Y=train_Y.numpy())
    _write_history(step, len(train_Y), best_T_now, hv_now, n_acqf_calls, n_uniform_tries)
    _print(f"  saved {output_name} + history.csv")

_print(f"\n=== PFN-EI live ANSYS run complete ===")
_print(f"  final n = {len(train_Y)}, best T = {float(train_Y[:, 0].max().item()):.4f}, HV = {_hv(train_Y):.4f}")
design.close_project()
