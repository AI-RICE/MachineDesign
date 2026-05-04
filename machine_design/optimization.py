import pickle
from collections.abc import Iterable

import numpy as np
import pandas as pd
import torch
from botorch.utils.transforms import unnormalize
from torch import Tensor

from .geometry import analyze_results


def objective_transform(TorAvg, TorRippleRms, objective_fallback=None):
    if pd.isnull(TorAvg):
        if objective_fallback is None:
            raise ValueError("At least one of TorAvg and objective_fallback must be non-null.")
        torque_fallback, ripple_fallback = objective_fallback
        return objective_transform(torque_fallback, ripple_fallback)
    else:
        return TorAvg, -TorRippleRms / 100


def objective(Xs: Tensor, *args, **kwargs) -> Tensor:
    vals = [objective_single(X, *args, **kwargs) for X in Xs]
    return torch.stack(vals, dim=0)


def objective_single(X: Tensor, design, generator, bounds, NUM_CORES, **kwargs) -> Tensor:
    X = unnormalize(X, bounds)
    params = generator.X_to_params(X.numpy())

    generator.set_parameters(params)
    barriers = generator.generate_barriers()
    barriers = generator.split_barriers(barriers)

    design.add_rotor()

    for barrier in barriers:
        design.add_rotor_barrier(barrier)

    # Compute the torque
    try:
        Tor = design.compute(NUM_CORES)
        TorAvg, _, TorRippleRms = analyze_results(Tor)
    except Exception:
        TorAvg, TorRippleRms = np.nan, np.nan

    # Delete the rotor
    design.delete_rotor()

    f1, f2 = objective_transform(TorAvg, TorRippleRms, **kwargs)
    return torch.tensor([f1, f2])


def init_points(root, method):
    results = pd.read_csv(f"{root}/metadata.csv")
    results = results[~results["T"].isnull()]
    results = results[results["method"] == method]
    results["path"] = [f"results/design_{row['method']}_{row['design']}.pkl" for _, row in results.iterrows()]

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
