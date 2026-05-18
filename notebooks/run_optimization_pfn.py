"""PFN-BO loop for SynRM rotor optimisation (M4).

Mirrors `notebooks/run_optimization.py` (the GP-EHVI baseline) but with a
PFN surrogate instead of a `SingleTaskGP`. Phase 1: the oracle is the
**v3 saturated lumped solver** (~270 ms/eval). The full PFN-vs-GP
comparison against FEA / FEA-emulator data is M4.5 work.

Single-objective for now (max T_proxy), since v3 lumped exposes only that
scalar. When a T_ripple proxy lands in v4, swap to multi-objective.

Run:
    python notebooks/run_optimization_pfn.py checkpoints/OneLambda_pfn.pt --n-init 50 --n-bo 50
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize, unnormalize

from machine_design.generators import (
    HacklGenerator_3BrokenLines,
    HacklGenerator_OneLambda,
    HacklGenerator_SixLambdas,
)
from machine_design.lumped import (
    GRANULARITY_FINE,
    REFERENCE_MACHINE,
    build_network,
    lumped_torque_proxy_saturated,
)
from machine_design.pfn import PFNSurrogate, load_checkpoint


GEN_LOOKUP = {
    "OneLambda": HacklGenerator_OneLambda,
    "SixLambdas": HacklGenerator_SixLambdas,
    "ThreeBrokenLines": HacklGenerator_3BrokenLines,
}


def _flatten_params(params) -> np.ndarray:
    flat: list[float] = []
    for x in params:
        if hasattr(x, "__iter__"):
            for y in x:
                flat.append(float(y))
        else:
            flat.append(float(x))
    return np.array(flat, dtype=float)


def _evaluate_lumped(gen, params, scale: float = 1.0) -> tuple[float, bool]:
    """Return (T_proxy, feasible). Returns (nan, False) on infeasible barriers."""
    gen.set_parameters(params)
    barriers = gen.generate_barriers()
    if not gen.feasible_barriers(barriers):
        return float("nan"), False
    net = build_network(REFERENCE_MACHINE, GRANULARITY_FINE, barrier_polylines=barriers)
    res = lumped_torque_proxy_saturated(net, mmf_amp=200.0)
    return float(res.T_proxy * scale), True


def _sample_initial_points(gen_cls, n_init: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, list]:
    rng = np.random.default_rng(seed)
    bounds_lo, bounds_hi = gen_cls(REFERENCE_MACHINE, r_stator_end=0.7, offset=0.35).bounds
    gen = gen_cls(REFERENCE_MACHINE, r_stator_end=0.7, offset=0.35)
    X_list, Y_list, kept_params = [], [], []
    tries = 0
    while len(X_list) < n_init and tries < n_init * 10:
        tries += 1
        np.random.seed(int(rng.integers(0, 2**31 - 1)))
        params = gen.random_parameters()
        T, feasible = _evaluate_lumped(gen, params)
        if not feasible:
            continue
        X_list.append(_flatten_params(params))
        Y_list.append(T)
        kept_params.append(params)
    if len(X_list) < n_init:
        raise RuntimeError(f"Could not sample {n_init} feasible initials in {tries} tries")
    X = np.stack(X_list)
    Y = np.array(Y_list)
    return X, Y, kept_params


def run_pfn_bo(
    checkpoint: Path,
    n_init: int = 50,
    n_bo: int = 50,
    q: int = 1,
    seed: int = 0,
    max_acq_tries: int = 20,
) -> dict:
    loaded = load_checkpoint(checkpoint)
    gen_cls = GEN_LOOKUP[loaded.generator_name]
    gen = gen_cls(REFERENCE_MACHINE, r_stator_end=0.7, offset=0.35)
    bounds_lo, bounds_hi = gen.bounds
    bounds_t = torch.tensor(np.stack([bounds_lo, bounds_hi]), dtype=torch.float32)

    print(f"PFN-BO replay")
    print(f"  generator: {loaded.generator_name}  D={loaded.input_dim}")
    print(f"  n_init={n_init}, n_bo={n_bo}, q={q}")
    print()

    t0 = time.time()
    X_np, Y_np, _ = _sample_initial_points(gen_cls, n_init=n_init, seed=seed)
    print(f"  initial set: {n_init} feasible designs in {time.time()-t0:.1f}s")
    print(f"  best initial T_proxy: {Y_np.max():.3e}")

    # Standardise Y for stable BO; record best running max.
    X = torch.tensor(X_np, dtype=torch.float32)
    Y = torch.tensor(Y_np, dtype=torch.float32).unsqueeze(-1)
    bounds_unit = torch.stack([torch.zeros(loaded.input_dim), torch.ones(loaded.input_dim)])
    best_history = [float(Y.max().item())]

    for step in range(1, n_bo + 1):
        surr = PFNSurrogate.from_loaded_with_real_Y(loaded, X, Y)
        # Surrogate posterior is in normalised PFN-space; normalise best_f to match.
        best_f_norm = (float(Y.max().item()) - loaded.y_mean) / loaded.y_std
        acq = qLogExpectedImprovement(model=surr, best_f=best_f_norm)

        # optimize_acqf works in unit cube; we map back to bounds.
        X_norm = normalize(X, bounds_t)
        # qLogEI uses MC inside; the optimizer just maximises the acq.
        # We use the PFN surrogate's posterior directly.
        # Since our model takes inputs in PFN-space (= bounds-space, not unit-cube),
        # we run optimize_acqf in PFN-space directly.
        cand, _ = optimize_acqf(
            acq_function=acq,
            bounds=bounds_t,
            q=q,
            num_restarts=5,
            raw_samples=64,
        )

        # Reject infeasible by rejection sampling.
        cand_np = cand.detach().cpu().numpy()
        feasible_eval = []
        for c in cand_np:
            T, feasible = _evaluate_lumped(gen, gen.X_to_params(c))
            if feasible:
                feasible_eval.append((c, T))
            else:
                tries = 0
                while tries < max_acq_tries:
                    tries += 1
                    np.random.seed(int(np.random.randint(2**31 - 1)))
                    fallback_params = gen.random_parameters()
                    T, feasible = _evaluate_lumped(gen, fallback_params)
                    if feasible:
                        feasible_eval.append((_flatten_params(fallback_params), T))
                        break

        if not feasible_eval:
            print(f"  step {step}: no feasible eval; skipping")
            best_history.append(best_history[-1])
            continue

        new_X = np.stack([fe[0] for fe in feasible_eval])
        new_Y = np.array([fe[1] for fe in feasible_eval])
        X = torch.cat([X, torch.tensor(new_X, dtype=torch.float32)])
        Y = torch.cat([Y, torch.tensor(new_Y, dtype=torch.float32).unsqueeze(-1)])
        cur_best = float(Y.max().item())
        best_history.append(cur_best)
        print(f"  step {step:>3}  best_T_proxy = {cur_best:.3e}")

    total_t = time.time() - t0
    print(f"\nFinished in {total_t:.1f}s ({total_t/60:.1f} min)")
    print(f"Best T_proxy: {best_history[-1]:.3e}  (init {best_history[0]:.3e})")
    return {
        "best_history": best_history,
        "X": X.cpu().numpy(),
        "Y": Y.cpu().numpy(),
        "elapsed_s": total_t,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--n-init", type=int, default=50)
    ap.add_argument("--n-bo", type=int, default=50)
    ap.add_argument("--q", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None,
                    help="optional .npz to dump the BO trace")
    args = ap.parse_args()

    result = run_pfn_bo(args.checkpoint, n_init=args.n_init, n_bo=args.n_bo,
                        q=args.q, seed=args.seed)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.out, **{
            "best_history": np.array(result["best_history"]),
            "X": result["X"],
            "Y": result["Y"],
            "elapsed_s": result["elapsed_s"],
        })
        print(f"Saved trace: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
