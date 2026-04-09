# export ANSYSEM_ROOT241=/data/AnsysEM/v241/Linux64

import os
import pickle
from collections.abc import Iterable

import numpy as np
import pandas as pd
from pyro import barrier
import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor

from machine_design import (
    Design,
    HacklGenerator_OneLambda,
    RandomBarrierGenerator,
    analyze_results,
)

torch.set_default_dtype(torch.float64)


def objective_transform(TorAvg, TorRippleRms):
    if pd.isnull(TorAvg):
        return -99999, -99999
    else:
        return TorAvg, -TorRippleRms / 100


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


def init_points(root, method, barrier):
    results = pd.read_csv(f"{root}/metadata.csv")
    results = results[~results["T"].isnull()]
    results = results[results["method"] == method]
    results["path"] = [f"results/design_{row['method']}_{row['design']}_barriers_{barrier}.pkl" for _, row in results.iterrows()]

    results = results[results["path"].apply(os.path.exists)]
    print(f"Processing {len(results)} valid files (skipping missing files)")

    Xs, Ys = [], []
    for _, r in results.iterrows():
        with open(r["path"], "rb") as f:
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
path_data = os.path.join(os.getcwd(), "data")
os.makedirs(path_data, exist_ok=True)
file_name_aedt = f"{path_data}/{project_name}.aedt"

# Define constants
AEDT_VERSION = "2025.2"
NUM_CORES = 24
NG_MODE = True  # non-graphical mode
CLS_EXIT = True  # close on exit

if not os.path.exists(file_name_aedt):
    design = Design.create(
        project_name,
        design_name,
        file_name_aedt,
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

root_init = "results"
for n_barrier in range(3, 6):
    generator = RandomBarrierGenerator(design, r_stator_end, offset=offset)
    method = generator.__class__.__name__
    generator.n_barriers = int(n_barrier)
    bounds = torch.tensor(np.vstack(generator.bounds), dtype=torch.float64)
    train_X, train_Y = init_points(root_init, method, n_barrier)
    train_barriers = torch.full((train_X.shape[0],), int(n_barrier))
    train_X = normalize(train_X, bounds)
    bounds_normalized = normalize(bounds, bounds)
    ref_point = torch.tensor([3.8, -max_ripple])

    for x in train_X:
        assert len(x) == 2*n_barrier, f"train_X has wrong length {len(x)} for barrier {n_barrier}"


    def objective_lambda(Xs):
        return objective(Xs, design, generator, bounds, NUM_CORES)


    def ripple_constraint(Y):
        ripple = -Y[..., 1]
        return ripple - max_ripple


    for _ in range(n_iters):
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
                params = generator.X_to_params(candidate_normalized.numpy(), n_barrier=n_barrier)

                generator.set_parameters(params)
                barriers = generator.generate_barriers()
                barriers = generator.split_barriers(barriers)
                feasible = generator.feasible_barriers(barriers)
                if feasible:
                    candidates_feasible.append((candidate, n_barrier))
                if len(candidates_feasible) >= batch_size:
                    break
            if len(candidates_feasible) >= batch_size:
                break
        candidates = torch.stack([candidate for candidate, _ in candidates_feasible])
        barrier_list = [barrier for _, barrier in candidates_feasible]

        # Evaluate candidates
        new_Y = objective_lambda(candidates)
        train_X = torch.cat([train_X, candidates])
        train_Y = torch.cat([train_Y, new_Y])

        new_barriers = torch.tensor(barrier_list, dtype=torch.int64)
        train_barriers = torch.cat([train_barriers, new_barriers])
        np.savez(f"results_{method}_{n_barrier}.npz", train_X=unnormalize(train_X, bounds), train_Y=train_Y, barriers=train_barriers.numpy())

design.close_project()