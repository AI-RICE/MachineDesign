# export ANSYSEM_ROOT241=/data/AnsysEM/v241/Linux64

import os

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
    HacklGenerator_3BrokenLines,
    HacklGenerator_OneLambda,
    HacklGenerator_SixLambdas,
    init_points,
    load_design,
    objective,
    objective_transform,
)

torch.set_default_dtype(torch.float64)


aedt_version = "2024.1"
n_evals = 250
r_stator_end = 0.7
offset = 0.7 / 2
num_cores = 4
batch_size = 4
max_candidate_tries = 10
objective_fallback = {"torque": 1.0, "ripple": 40.0}
ref_cons = {"torque": 4.0, "ripple": 10.0}
ref_no_cons = {"torque": 4.0, "ripple": 30.0}

project_name = "SynRM_test"
design_name = "Design01"
path_data = os.path.join(os.getcwd(), "data")
root_init = "results"
os.makedirs(path_data, exist_ok=True)
file_name_aedt = f"{path_data}/{project_name}.aedt"

design = load_design(file_name_aedt, project_name, design_name, aedt_version)
generators = [
    HacklGenerator_OneLambda(design, r_stator_end, offset=offset),
    HacklGenerator_SixLambdas(design, r_stator_end, offset=offset),
    HacklGenerator_3BrokenLines(design, r_stator_end, offset=offset),
]

ref_cons_torque, ref_cons_ripple = objective_transform(ref_cons["torque"], ref_cons["ripple"])
ref_no_cons_torque, ref_no_cons_ripple = objective_transform(ref_no_cons["torque"], ref_no_cons["ripple"])
objective_fallback_tuple = (objective_fallback["torque"], objective_fallback["ripple"])

for generator in generators:
    for use_constraints in [True, False]:
        method = generator.__class__.__name__
        output_name = f"results_{method}_{use_constraints}.npz"

        if os.path.exists(output_name):
            data = np.load(output_name)
            train_X = torch.from_numpy(data["train_X"])
            train_Y = torch.from_numpy(data["train_Y"])
        else:
            train_X, train_Y = init_points(root_init, method)

        bounds = torch.from_numpy(np.vstack(generator.bounds))
        bounds_normalized = normalize(bounds, bounds)
        train_X = normalize(train_X, bounds)

        def objective_lambda(Xs):
            return objective(Xs, design, generator, bounds, num_cores, objective_fallback=objective_fallback_tuple)

        def penalty_objective(n_penalty):
            obj = objective_transform(None, None, objective_fallback=objective_fallback_tuple)
            y = torch.tensor(obj, dtype=torch.float64)
            return y.repeat(n_penalty, 1)

        def ripple_constraint(Y):
            ripple = -Y[..., 1]
            ripple_max = -ref_cons_ripple
            return ripple - ripple_max

        if use_constraints:
            constraints = [ripple_constraint]
            ref_point = torch.tensor([ref_cons_torque, ref_cons_ripple])
        else:
            constraints = None
            ref_point = torch.tensor([ref_no_cons_torque, ref_no_cons_ripple])

        while len(train_X) < n_evals:
            # Fit surrogate
            model = SingleTaskGP(train_X, train_Y)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            # Compute Pareto front
            pareto_Y = train_Y[is_non_dominated(train_Y)]
            partitioning = NondominatedPartitioning(ref_point=ref_point, Y=pareto_Y)

            # Define acquisition function
            acq = qLogExpectedHypervolumeImprovement(
                model=model,
                ref_point=ref_point.tolist(),
                partitioning=partitioning,
                constraints=constraints,
            )

            # Optimize acquisition function to select candidate points. Reject unfeasible points
            candidates_feasible = []
            candidates_infeasible = []
            n_acqf_calls = 0
            t_acqf = 0.0
            for _ in range(max_candidate_tries):
                n_needed = batch_size - len(candidates_feasible)

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

            assert len(candidates_feasible) + len(candidates_infeasible) >= batch_size
            n_missing = batch_size - len(candidates_feasible)

            # Fill missing candidates from infeasible
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

            print(len(train_Y))
            print(train_Y[is_non_dominated(train_Y)])

            # Save candidates
            np.savez(output_name, train_X=unnormalize(train_X, bounds), train_Y=train_Y)

design.close_project()
