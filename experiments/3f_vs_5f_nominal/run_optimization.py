# export ANSYSEM_ROOT241=/data/AnsysEM/v241/Linux64

import os

import numpy as np
import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood

from machine_design.config import load_config
from machine_design.designs import load_design
from machine_design.optimization import HacklGenerator_SixLambdas
from optimization import (
    init_points,
    objective,
    objective_transform,
    )

torch.set_default_dtype(torch.float64)

phases=3

if phases==3:
    from motors.synrm_3f_60s import Computation, Geometry
    current_n=2
    current_bounds=np.array([[0.0, 0.0], [4.0, 4.0]])  #Id, Iq bounds for 3f
    ref_loss=20
else:
    from motors.synrm_5f_60s import Computation, Geometry
    current_n=4
    current_bounds=np.array([[0.0, 0.0, 0.0, 0.0], [10.0, 10.0, 10.0, 10.0]])  #Id1, Iq1, Id3, Iq3 bounds for 5f
    ref_loss=40

config = load_config()
aedt_version = config["aedt_version"]
num_cores = config["num_cores"]
n_evals = 250
r_stator_end = 0.7
offset = 0.7 / 2
batch_size = 4
max_candidate_tries = 10
t_target=20.0
objective_fallback = {"loss": ref_loss, "torque": 1.0, "ripple": 40.0}
ref_cons = {"loss": ref_loss, "ripple": 10.0}
# ref_no_cons = {"torque": 4.0, "ripple": 30.0}

project_name = f"SynRM_{phases}f_nominal"
design_name = "Design01"
path_data = os.path.join(os.getcwd(), "data")
root_init = "results"
os.makedirs(path_data, exist_ok=True)
file_name_aedt = f"{path_data}/{project_name}.aedt"

geometry = Geometry()
computation = Computation(geometry)
design = load_design(file_name_aedt, project_name, design_name, aedt_version, geometry, computation)
generator = HacklGenerator_SixLambdas(design, r_stator_end, offset=offset)

objective_fallback_tuple = (objective_fallback["loss"], objective_fallback["torque"], objective_fallback["ripple"])

method = generator.__class__.__name__
output_name = f"results_{method}_{phases}f.npz"

if os.path.exists(output_name):
    data = np.load(output_name)
    train_X = torch.from_numpy(data["train_X"])
    train_Y = torch.from_numpy(data["train_Y"])
else:
    train_X, train_Y = init_points(root_init, method)

bounds = torch.from_numpy(np.hstack([np.vstack(generator.bounds), current_bounds]))
bounds_normalized = normalize(bounds, bounds)
train_X = normalize(train_X, bounds)

def objective_lambda(Xs):
    return objective(Xs, design, generator, current_n, bounds, num_cores, objective_fallback=objective_fallback_tuple)

def penalty_objective(n_penalty):
    obj = objective_transform(None, None, None, objective_fallback=objective_fallback_tuple)
    y = torch.tensor(obj, dtype=torch.float64)
    return y.repeat(n_penalty, 1)

def torque_constraint(Y):
    return t_target-Y[..., 1]


def ripple_constraint(Y):
    return -100*Y[..., 2]-ref_cons["ripple"]  #  train_Y[...,2] stores -TorRippleRms/100


constraints = [torque_constraint, ripple_constraint]
loss_objective=GenericMCObjective(lambda Y, X=None: Y[...,0])

while len(train_X) < n_evals:
    # Fit surrogate
    model = SingleTaskGP(train_X, train_Y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)


    # Define acquisition function
    acq = qLogNoisyExpectedImprovement(
        model=model,
        X_baseline=train_X,
        objective=loss_objective,
        constraints=constraints,
        prune_baseline=True,
            )

    # Optimize acquisition function to select candidate points. Reject unfeasible points
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
            barrier_X=candidate_unnormalized[:-current_n]
            params = generator.X_to_params(barrier_X.numpy())

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

    assert len(candidates_feasible) + len(candidates_infeasible) >= batch_size
    n_missing = batch_size - len(candidates_feasible)

    # Fill missing candidates from infeasible
    if len(candidates_feasible) > 0:
        candidates_all = torch.stack(candidates_feasible)
        new_Y_all = objective_lambda(candidates_all)
    else:
        candidates_all = torch.empty((0, bounds.shape[1]), dtype=torch.float64)
        new_Y_all = torch.empty((0, 3), dtype=torch.float64)
    if n_missing > 0:
        candidates_all = torch.cat([candidates_all, torch.stack(candidates_infeasible[:n_missing])], dim=0)
        new_Y_all = torch.cat([new_Y_all, penalty_objective(n_missing)], dim=0)

    train_X = torch.cat([train_X, candidates_all])
    train_Y = torch.cat([train_Y, new_Y_all])

    feasible=(train_Y[:,1]>=t_target) & (-100*train_Y[:, 2]<=ref_cons["ripple"])
    print(len(train_Y), feasible.sum().item())
    print(train_Y[feasible][:,0].max() if feasible.any() else None)

    # Save candidates
    np.savez(output_name, train_X=unnormalize(train_X, bounds), train_Y=train_Y)

design.close_project()
