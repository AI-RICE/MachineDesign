# export ANSYSEM_ROOT241=/data/AnsysEM/v241/Linux64

from collections.abc import Iterable
import os
import numpy as np
import pandas as pd
import pickle
from machine_design import Design, analyze_results, plot_barriers
from machine_design import FourStupid, HacklGenerator_OneLambda, HacklGenerator_TwoLambdas, HacklGenerator_OneLambdaTheta
import torch
torch.set_default_dtype(torch.float64)
from torch import Tensor
from botorch.utils.transforms import normalize, unnormalize
import torch

from botorch.models import SingleTaskGP
from botorch import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.optim import optimize_acqf

def objective_transform(TorAvg, TorRippleRms):
    if pd.isnull(TorAvg):
        return -99999, -99999
    else:
        return TorAvg, -TorRippleRms/100

def objective(Xs: Tensor, *args) -> Tensor:
    vals = [objective_single(X, *args) for X in Xs]
    return torch.stack(vals, dim=0)

def objective_single(X: Tensor, design, generator, bounds, NUM_CORES) -> Tensor:
    X = unnormalize(X, bounds)
    params = generator.X_to_params(X.numpy())

    generator.set_parameters(params)
    barriers = generator.generate_barriers()
    barriers = generator.split_barriers(barriers)

    design.add_rotor()

    for barrier in barriers:
        design.add_rotor_barrier(barrier)

    # Compute the torque
    Tor = design.compute(NUM_CORES)
    if Tor is None:
        TorAvg, TorRippleRms = np.nan, np.nan
    else: 
        TorAvg, _, TorRippleRms = analyze_results(Tor)

    # Delete the rotor
    design.delete_rotor()

    f1, f2 = objective_transform(TorAvg, TorRippleRms)
    return torch.tensor([f1, f2])

def init_points(root, method):
    results = pd.read_csv(f"{root}/metadata.csv")
    results = results[~results["T"].isnull()]
    results = results[results["method"] == method]
    results["path"] = [f"results/design_{row['method']}_{row['design']}.pkl" for _, row in results.iterrows()]

    Xs, Ys = [], []
    for _, r in results.iterrows():
        with open(r["path"], 'rb') as f:
            params = pickle.load(f)
        
        X = []
        for x in params:
            if isinstance(x, Iterable):
                for y in x:
                    X.append(y)
            else:
                X.append(x)
        Y = objective_transform(r["T"], r["ripple"])

        Xs.append(torch.Tensor(X))
        Ys.append(torch.Tensor(Y))

    return torch.stack(Xs), torch.stack(Ys)

project_name = "SynRM_test"
design_name = "Design01"
path_data = os.path.join(os.getcwd(), 'data')
os.makedirs(path_data, exist_ok=True)
file_name_aedt = f'{path_data}/{project_name}.aedt'

# Define constants
AEDT_VERSION = "2024.1"
NUM_CORES = 4
NG_MODE = True  #non-graphical mode
CLS_EXIT = True #close on exit

if not os.path.exists(file_name_aedt):
    design = Design.create(
        project_name, design_name, file_name_aedt,
        version=AEDT_VERSION,
        non_graphical=NG_MODE,
        new_desktop=False,
        close_on_exit=CLS_EXIT,
    )
else:
    design = Design.load(
        file_name_aedt,
        version=AEDT_VERSION,
        non_graphical=NG_MODE,
        new_desktop=False,
        close_on_exit=CLS_EXIT,
    )

max_ripple = 0.1
n_iters = 40
batch_size = 4
r_stator_end = 0.7
offset = 0.7 / 2
generator = HacklGenerator_OneLambda(design, r_stator_end, offset=offset)
# generator = FourStupid(design, r_stator_end, offset=offset)
# generator = HacklGenerator_TwoLambdas(design, r_stator_end, offset=offset)
bounds = torch.from_numpy(np.vstack(generator.bounds))

root_init = "results"
method = generator.__class__.__name__
train_X, train_Y = init_points(root_init, method)
train_X = normalize(train_X, bounds)
bounds_normalized = normalize(bounds, bounds)
ref_point = torch.tensor([3.8,-max_ripple])

def objective_lambda(Xs):
    return objective(Xs, design, generator, bounds, NUM_CORES)

def ripple_constraint(Y):
    ripple = -Y[...,1]
    return ripple - max_ripple

for _ in range(n_iters):
    # Fit surrogate
    model = SingleTaskGP(train_X, train_Y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    # Compute Pareto front
    pareto_Y = train_Y[is_non_dominated(train_Y)]
    partitioning = NondominatedPartitioning(
        ref_point=ref_point,
        Y=pareto_Y
    )

    # Define acquisition function
    acq = qLogExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point.tolist(),
        partitioning=partitioning,
        constraints=[ripple_constraint],        
    )

    # Optimize acquisition function to select candidate points. Reject unfeasible points
    candidates_feasible = []
    while True:
        candidates, _ = optimize_acqf(
            acq_function=acq,
            bounds=bounds_normalized,
            q=batch_size,
            num_restarts=10,
            raw_samples=128,
        )
        for candidate in candidates:
            candidate_normalized = unnormalize(candidate, bounds)
            params = generator.X_to_params(candidate_normalized.numpy())

            generator.set_parameters(params)
            barriers = generator.generate_barriers()
            barriers = generator.split_barriers(barriers)
            feasible = generator.feasible_barriers(barriers)
            if feasible:
                candidates_feasible.append(candidate)
            if len(candidates_feasible) >= batch_size:
                break
        if len(candidates_feasible) >= batch_size:
            break
    candidates = torch.stack(candidates_feasible)

    # Evaluate candidates
    new_Y = objective_lambda(candidates)
    train_X = torch.cat([train_X, candidates])
    train_Y = torch.cat([train_Y, new_Y])
    np.savez(f"results_{method}.npz", train_X=unnormalize(train_X, bounds), train_Y=train_Y)

design.close_project()