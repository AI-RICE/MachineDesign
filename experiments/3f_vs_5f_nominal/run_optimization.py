# export ANSYSEM_ROOT241=/data/AnsysEM/v241/Linux64

import os

import importlib

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
    current_inequality_constraints,
    current_peak,
    init_points,
    objective,
    objective_transform,
)

torch.set_default_dtype(torch.float64)

config = load_config()
aedt_version = config["aedt_version"]
num_cores = config["num_cores"]
n_evals = 20
r_stator_end = 0.7
offset = 0.7 / 2
batch_size = 4
max_candidate_tries = 10
t_target = 6.0
current_scale = 0.584  # according to PR79
i_max_3f = 2.0
i_max_5f = current_scale * i_max_3f
current_half_bounds_3f = i_max_3f  # case 1, 3f
current_half_bounds_5f_i3zero = i_max_5f  # case 2, 5f, id3=iq3=0
current_half_bounds_5f = (4 / np.pi) * i_max_5f  # case 3, 5f, free id3 and iq3
v_max = 311.13

for case, module_name, n_phases, current_bounds, i_max, v_max in [  # i_max and v_max are set temperally to check the code can work.
    (1, "motors.synrm_3f_60s", 3, np.array([[-current_half_bounds_3f] * 2, [current_half_bounds_3f] * 2]), i_max_3f, v_max),
    (2, "motors.synrm_5f_60s", 5, np.array([[-current_half_bounds_5f_i3zero] * 2, [current_half_bounds_5f_i3zero] * 2]), i_max_5f, v_max),
    (3, "motors.synrm_5f_60s", 5, np.array([[-current_half_bounds_5f] * 4, [current_half_bounds_5f] * 4]), i_max_5f, v_max),
]:
    module = importlib.import_module(module_name)
    Computation, Geometry = module.Computation, module.Geometry
    geometry = Geometry()
    current_n = current_bounds.shape[1]
    harmonics = [1] if n_phases == 3 else [1, 3]
    harmonics_free = harmonics[: current_n // 2]
    case_name = f"{n_phases}f_case{case}"

    objective_fallback = {"loss": 40, "torque": 1.0, "ripple": 40.0, "voltage": 1000.0}
    ref_cons = {"ripple": 10.0}
    _, _, ref_cons_ripple, _ = objective_transform(0, 0, ref_cons["ripple"])

    project_name = f"SynRM_{n_phases}f_nominal"
    design_name = "Design01"
    path_data = os.path.join(os.getcwd(), "data")
    root_init = f"results/results_{case_name}"
    os.makedirs(path_data, exist_ok=True)
    file_name_aedt = f"{path_data}/{project_name}.aedt"

    computation = Computation(geometry)
    design = load_design(file_name_aedt, project_name, design_name, aedt_version, geometry, computation)
    generator = HacklGenerator_SixLambdas(design, r_stator_end, offset=offset)

    objective_fallback_tuple = (objective_fallback["loss"], objective_fallback["torque"], objective_fallback["ripple"], objective_fallback["voltage"])

    method = generator.__class__.__name__
    output_name = f"results/results_{method}_{case_name}.npz"

    if os.path.exists(output_name):
        data = np.load(output_name)
        train_X = torch.from_numpy(data["train_X"])
        train_Y = torch.from_numpy(data["train_Y"])
    else:
        train_X, train_Y = init_points(root_init, method)

    bounds = torch.from_numpy(np.hstack([np.vstack(generator.bounds), current_bounds]))
    bounds_normalized = normalize(bounds, bounds)
    current_constraints = current_inequality_constraints(harmonics_free, i_max, bounds, current_n)
    train_X = normalize(train_X, bounds)

    def objective_lambda(Xs):
        return objective(Xs, design, generator, current_n, bounds, num_cores, harmonics, objective_fallback=objective_fallback_tuple)

    def penalty_objective(n_penalty):
        obj = objective_transform(None, None, None, None, objective_fallback=objective_fallback_tuple)
        y = torch.tensor(obj, dtype=torch.float64)
        return y.repeat(n_penalty, 1)

    def torque_constraint(Y):  # torque constraints
        return t_target - Y[..., 1]

    def ripple_constraint(Y):  # ripple constraints
        ripple = -Y[..., 2]
        ripple_max = -ref_cons_ripple
        return ripple - ripple_max

    def voltage_constraint(Y):  # voltage constraints
        return Y[..., 3] - v_max

    constraints = [torque_constraint, ripple_constraint, voltage_constraint]  # computed by Ansys results.
    loss_objective = GenericMCObjective(lambda Y, X=None: Y[..., 0])

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
                inequality_constraints=current_constraints,
                q=batch_size,
                num_restarts=10,
                raw_samples=128,
            )

            for candidate in candidates:
                candidate_unnormalized = unnormalize(candidate, bounds)
                barrier_X = candidate_unnormalized[:-current_n]
                current_X = candidate_unnormalized[-current_n:]
                params = generator.X_to_params(barrier_X.numpy())

                generator.set_parameters(params)
                barriers = generator.generate_barriers()
                barriers = generator.split_barriers(barriers)
                feasible_geom = generator.feasible_barriers(barriers)

                peak_current = current_peak(current_X.numpy(), harmonics_free)
                feasible_current = i_max is None or peak_current <= i_max  # currents constraints
                feasible = feasible_geom and feasible_current
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
            new_Y_all = torch.empty((0, 4), dtype=torch.float64)
        if n_missing > 0:
            candidates_all = torch.cat([candidates_all, torch.stack(candidates_infeasible[:n_missing])], dim=0)
            new_Y_all = torch.cat([new_Y_all, penalty_objective(n_missing)], dim=0)

        train_X = torch.cat([train_X, candidates_all])
        train_Y = torch.cat([train_Y, new_Y_all])

        feasible = (train_Y[:, 1] >= t_target) & (train_Y[:, 2] >= ref_cons_ripple)
        print(len(train_Y), feasible.sum().item())
        print(train_Y[feasible][:, 0].max() if feasible.any() else None)

        # Save candidates
        np.savez(output_name, train_X=unnormalize(train_X, bounds), train_Y=train_Y)

design.close_project()
