"""Smoke test of the GP-EHVI baseline (`run_optimization.py`) on live ANSYS.

Trimmed scope to validate the ANSYS + bayes pipeline end-to-end before
committing to a multi-hour multi-cell sweep:

- Single generator: OneLambda (the 7-D, smallest parameterisation).
- Single constraint setting: `use_constraints=False`.
- `n_evals = 90` (i.e. 50 initials + 40 BO evaluations = 10 BO iterations
  at `batch_size=4`).
- Outputs land in `sweeps_baseline_smoke/` (not the canonical `results_*.npz`
  filenames) so a successful smoke does not collide with a future full
  baseline run.

If this completes cleanly (`SynRM_test.aedt` solves, hypervolume increases
across iterations, NPZ saved), we know the pipeline is healthy and can
move to the full `run_optimization.py` for the headline GP-EHVI baseline.

Resumable: re-running the script picks up where it left off via the
existing `results_OneLambda_False.npz` inside `sweeps_baseline_smoke/`.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood

from machine_design import (
    HacklGenerator_OneLambda,
    init_points,
    load_design,
    objective,
    objective_transform,
)


torch.set_default_dtype(torch.float64)


aedt_version = "2024.1"
n_evals = 90  # 50 initials + 40 BO (10 iters × batch=4)
r_stator_end = 0.7
offset = 0.7 / 2
num_cores = 4
batch_size = 4
max_candidate_tries = 10
objective_fallback = {"torque": 1.0, "ripple": 40.0}
ref_no_cons = {"torque": 4.0, "ripple": 30.0}

project_name = "SynRM_test"
design_name = "Design01"

# We expect to be invoked from MachineDesign root; results/ + data/ are
# mirrored from /home/sadda/Projects/MachineDesign/ (see project CLAUDE.md §3).
path_data = os.path.join(os.getcwd(), "data")
root_init = "results"
out_dir = Path("sweeps_baseline_smoke")
out_dir.mkdir(parents=True, exist_ok=True)
file_name_aedt = f"{path_data}/{project_name}.aedt"

print(f"=== smoke GP-EHVI baseline ===", flush=True)
print(f"  generator: OneLambda (only)", flush=True)
print(f"  use_constraints: False (only)", flush=True)
print(f"  n_evals total: {n_evals}", flush=True)
print(f"  output dir: {out_dir.resolve()}", flush=True)
print(f"  aedt project: {file_name_aedt}", flush=True)

design = load_design(file_name_aedt, project_name, design_name, aedt_version)
generator = HacklGenerator_OneLambda(design, r_stator_end, offset=offset)

ref_no_cons_torque, ref_no_cons_ripple = objective_transform(ref_no_cons["torque"], ref_no_cons["ripple"])
objective_fallback_tuple = (objective_fallback["torque"], objective_fallback["ripple"])

method = generator.__class__.__name__
output_name = out_dir / f"results_{method}_False.npz"

if output_name.exists():
    print(f"  resuming from {output_name}", flush=True)
    data = np.load(output_name)
    train_X = torch.from_numpy(data["train_X"])
    train_Y = torch.from_numpy(data["train_Y"])
else:
    print(f"  loading initials from {root_init}/", flush=True)
    train_X, train_Y = init_points(root_init, method)
    print(f"  initial n = {len(train_X)}", flush=True)

bounds = torch.from_numpy(np.vstack(generator.bounds))
bounds_normalized = normalize(bounds, bounds)
train_X = normalize(train_X, bounds)

def objective_lambda(Xs):
    return objective(Xs, design, generator, bounds, num_cores, objective_fallback=objective_fallback_tuple)

def penalty_objective(n_penalty):
    obj = objective_transform(None, None, objective_fallback=objective_fallback_tuple)
    y = torch.tensor(obj, dtype=torch.float64)
    return y.repeat(n_penalty, 1)

constraints = None
ref_point = torch.tensor([ref_no_cons_torque, ref_no_cons_ripple])

iter_idx = 0
while len(train_X) < n_evals:
    iter_idx += 1
    print(f"\n--- iter {iter_idx}  (current n = {len(train_X)} / {n_evals}) ---", flush=True)

    model = SingleTaskGP(train_X, train_Y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    pareto_Y = train_Y[is_non_dominated(train_Y)]
    partitioning = NondominatedPartitioning(ref_point=ref_point, Y=pareto_Y)
    acq = qLogExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point.tolist(),
        partitioning=partitioning,
        constraints=constraints,
    )

    # Rejection-sampling for feasibility (same protocol as run_optimization.py).
    candidates_feasible = []
    candidates_infeasible = []
    for _ in range(max_candidate_tries):
        candidates, _ = optimize_acqf(
            acq_function=acq,
            bounds=bounds_normalized,
            q=batch_size,
            num_restarts=10,
            raw_samples=128,
        )
        for candidate in candidates:
            candidate_unnormalized = unnormalize(candidate, bounds)
            params = generator.X_to_params(candidate_unnormalized.numpy())
            generator.set_parameters(params)
            barriers = generator.generate_barriers()
            barriers = generator.split_barriers(barriers)
            feasible = generator.feasible_barriers(barriers)
            if feasible:
                candidates_feasible.append(candidate)
            else:
                candidates_infeasible.append(candidate)
            if len(candidates_feasible) >= batch_size:
                break
        if len(candidates_feasible) >= batch_size:
            break

    n_missing = batch_size - len(candidates_feasible)
    if len(candidates_feasible) > 0:
        candidates_all = torch.stack(candidates_feasible)
        new_Y_all = objective_lambda(candidates_all)
    else:
        candidates_all = torch.empty((0, bounds.shape[1]), dtype=torch.float64)
        new_Y_all = torch.empty((0, 2), dtype=torch.float64)
    if n_missing > 0:
        candidates_all = torch.cat([candidates_all, torch.stack(candidates_infeasible[:n_missing])], dim=0)
        new_Y_all = torch.cat([new_Y_all, penalty_objective(n_missing)], dim=0)

    train_X = torch.cat([train_X, candidates_all])
    train_Y = torch.cat([train_Y, new_Y_all])

    print(f"  n now: {len(train_Y)}", flush=True)
    print(f"  current Pareto front:\n{train_Y[is_non_dominated(train_Y)]}", flush=True)

    np.savez(output_name, train_X=unnormalize(train_X, bounds), train_Y=train_Y)
    print(f"  saved {output_name}", flush=True)

print("\n=== smoke complete ===", flush=True)
design.close_project()
