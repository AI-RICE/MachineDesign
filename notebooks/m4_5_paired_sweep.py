"""Paired GP-EHVI vs PFN-BO sweep on the FEA emulator (M4.5).

Single-objective (max T_mean) for v1, since v3 lumped exposes only T_proxy
and we have only the T-PFN trained. Multi-objective with T_ripple is a
v4 + M4.6 extension (needs a T_ripple PFN — see CLAUDE.md §5).

Both surrogates start from the same `n_init` feasible random initials.
At each BO step they propose `q` candidates; we reject-sample for
feasibility against the generator and evaluate via the emulator.

Track running max T_mean per surrogate; report `n_evals_to_reach(target)`
for `target = 95%` of the GP's final value (or, when a saved GP trace
is available, `HV_GP@250`).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

from machine_design.fea_emulator import FEAEmulator
from machine_design.generators import (
    HacklGenerator_3BrokenLines,
    HacklGenerator_OneLambda,
    HacklGenerator_SixLambdas,
)
from machine_design.lumped import REFERENCE_MACHINE
from machine_design.pfn import PFNSurrogate, load_checkpoint


GENERATORS = {
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


def _make_initials(gen, n_init: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Sample n_init feasible random designs."""
    X_list = []
    params_list = []
    tries = 0
    while len(X_list) < n_init and tries < n_init * 20:
        tries += 1
        np.random.seed(int(rng.integers(0, 2**31 - 1)))
        params = gen.random_parameters()
        gen.set_parameters(params)
        barriers = gen.generate_barriers()
        if not gen.feasible_barriers(barriers):
            continue
        X_list.append(_flatten_params(params))
        params_list.append(params)
    if len(X_list) < n_init:
        raise RuntimeError(f"Could not sample {n_init} feasible initials in {tries} tries")
    return np.stack(X_list), params_list


def _evaluate_emulator(emulator: FEAEmulator, gen, X: np.ndarray) -> tuple[float, float, bool]:
    """Predict (T_mean, T_ripple, feasible). NaN if infeasible."""
    params = gen.X_to_params(X)
    gen.set_parameters(params)
    barriers = gen.generate_barriers()
    if not gen.feasible_barriers(barriers):
        return float("nan"), float("nan"), False
    T, R = emulator.predict(X.reshape(1, -1))
    return float(T[0]), float(R[0]), True


def _bounds_tensor(gen) -> torch.Tensor:
    lo, hi = gen.bounds
    return torch.tensor(np.stack([lo, hi]), dtype=torch.float32)


def _is_feasible(gen, x_np: np.ndarray) -> bool:
    params = gen.X_to_params(x_np)
    gen.set_parameters(params)
    barriers = gen.generate_barriers()
    return bool(gen.feasible_barriers(barriers))


def _acquire_feasible(
    acq,
    bounds: torch.Tensor,
    gen,
    q: int,
    rng: np.random.Generator,
    max_acqf_tries: int = 10,
    max_uniform_tries: int = 200,
) -> tuple[np.ndarray, dict]:
    """Return q feasible candidates from `acq`, falling back to uniform-feasible.

    Mirrors the rejection loop in `notebooks/run_optimization.py`: call
    `optimize_acqf` up to `max_acqf_tries` times, take any feasible candidates.
    If still short, fill from uniform-random-feasible draws (same protocol as
    the random initials), so a BO step always makes some progress instead of
    silently stalling on a bounds-violating proposal.

    Returns (candidates np.ndarray of shape (q, D), debug dict).
    """
    feasible: list[np.ndarray] = []
    n_acqf_calls = 0
    for _ in range(max_acqf_tries):
        # Bump restarts + raw samples vs the BoTorch default; with raw params
        # in [0.25, 44] the L-BFGS-B optimizer needs more initial conditions
        # to land on a good local optimum (the SciPy ABNORMAL failures we
        # saw came from a too-thin restart set on an ill-conditioned surface).
        cand, _ = optimize_acqf(
            acq_function=acq, bounds=bounds, q=max(q, 4),
            num_restarts=10, raw_samples=256,
        )
        n_acqf_calls += 1
        for c in cand.detach().cpu().numpy():
            c = np.asarray(c, dtype=np.float64)
            if _is_feasible(gen, c):
                feasible.append(c)
                if len(feasible) >= q:
                    break
        if len(feasible) >= q:
            break

    n_uniform_tries = 0
    if len(feasible) < q:
        lo, hi = gen.bounds
        while len(feasible) < q and n_uniform_tries < max_uniform_tries:
            n_uniform_tries += 1
            # IMPORTANT: keep float64 — feasibility is determined by exact
            # polygon-intersection checks, and float32 rounding (~1e-7) is
            # enough to push otherwise-feasible barrier curves *just* outside
            # the polyline range, dropping feasibility from ~98% to 0% in
            # 13-D ThreeBrokenLines.
            c = rng.uniform(lo, hi)  # float64
            if _is_feasible(gen, c):
                feasible.append(c)

    return np.stack(feasible[:q]) if feasible else np.zeros((0, len(gen.bounds[0])), dtype=np.float64), {
        "n_acqf_calls": n_acqf_calls,
        "n_uniform_tries": n_uniform_tries,
        "n_feasible_obtained": len(feasible),
    }


def _gp_bo_loop(
    emulator: FEAEmulator,
    gen_cls,
    n_init: int,
    n_bo: int,
    seed: int,
    q: int = 1,
) -> dict:
    gen = gen_cls(REFERENCE_MACHINE, r_stator_end=0.7, offset=0.35)
    rng = np.random.default_rng(seed)
    X_init, _ = _make_initials(gen, n_init, rng)
    Y_init, R_init = [], []
    for x in X_init:
        T, R, _ = _evaluate_emulator(emulator, gen, x)
        Y_init.append(T)
        R_init.append(R)
    X = torch.tensor(X_init, dtype=torch.float32)
    Y = torch.tensor(Y_init, dtype=torch.float32).unsqueeze(-1)
    R = torch.tensor(R_init, dtype=torch.float32).unsqueeze(-1)
    bounds = _bounds_tensor(gen)
    history = [float(Y.max().item())]

    t0 = time.time()
    feas_debug: list[dict] = []
    D = X.shape[1]
    for step in range(1, n_bo + 1):
        # Normalize inputs to the unit cube so SciPy's acquisition optimizer
        # is well-conditioned; without this the GP saw raw params in
        # [0.25, 44] and `optimize_acqf` returned ABNORMAL status (search
        # never moved past initials).
        model = SingleTaskGP(
            X.to(torch.float64), Y.to(torch.float64),
            input_transform=Normalize(d=D, bounds=bounds.to(torch.float64)),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        acq = qLogExpectedImprovement(model=model, best_f=float(Y.max().item()))
        cand_np, dbg = _acquire_feasible(acq, bounds, gen, q, rng)
        feas_debug.append(dbg)
        for c in cand_np:
            T, _R, _ = _evaluate_emulator(emulator, gen, c)
            X = torch.cat([X, torch.tensor(c, dtype=torch.float32).unsqueeze(0)])
            Y = torch.cat([Y, torch.tensor([T], dtype=torch.float32).unsqueeze(0)])
            R = torch.cat([R, torch.tensor([_R], dtype=torch.float32).unsqueeze(0)])
        history.append(float(Y.max().item()))

    return {
        "name": "GP",
        "history": history,
        "X": X.cpu().numpy(),
        "Y": Y.cpu().numpy(),
        "R": R.cpu().numpy(),
        "elapsed_s": time.time() - t0,
        "feasibility_debug": feas_debug,
    }


def _pfn_bo_loop(
    emulator: FEAEmulator,
    gen_cls,
    pfn_checkpoint: Path,
    n_init: int,
    n_bo: int,
    seed: int,
    q: int = 1,
) -> dict:
    gen = gen_cls(REFERENCE_MACHINE, r_stator_end=0.7, offset=0.35)
    rng = np.random.default_rng(seed)
    X_init, _ = _make_initials(gen, n_init, rng)
    Y_init, R_init = [], []
    for x in X_init:
        T, R, _ = _evaluate_emulator(emulator, gen, x)
        Y_init.append(T)
        R_init.append(R)
    X = torch.tensor(X_init, dtype=torch.float32)
    Y = torch.tensor(Y_init, dtype=torch.float32).unsqueeze(-1)
    R = torch.tensor(R_init, dtype=torch.float32).unsqueeze(-1)
    bounds = _bounds_tensor(gen)
    history = [float(Y.max().item())]

    loaded = load_checkpoint(pfn_checkpoint)

    t0 = time.time()
    feas_debug: list[dict] = []
    for step in range(1, n_bo + 1):
        surr = PFNSurrogate.from_loaded_with_real_Y(loaded, X, Y)
        best_f_norm = (float(Y.max().item()) - loaded.y_mean) / loaded.y_std
        acq = qLogExpectedImprovement(model=surr, best_f=best_f_norm)
        cand_np, dbg = _acquire_feasible(acq, bounds, gen, q, rng)
        feas_debug.append(dbg)
        for c in cand_np:
            T, _R, _ = _evaluate_emulator(emulator, gen, c)
            X = torch.cat([X, torch.tensor(c, dtype=torch.float32).unsqueeze(0)])
            Y = torch.cat([Y, torch.tensor([T], dtype=torch.float32).unsqueeze(0)])
            R = torch.cat([R, torch.tensor([_R], dtype=torch.float32).unsqueeze(0)])
        history.append(float(Y.max().item()))

    return {
        "name": "PFN",
        "history": history,
        "X": X.cpu().numpy(),
        "Y": Y.cpu().numpy(),
        "R": R.cpu().numpy(),
        "elapsed_s": time.time() - t0,
        "feasibility_debug": feas_debug,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("generator", choices=list(GENERATORS))
    ap.add_argument("--emulator", type=Path, default=None,
                    help="emulator path (default: emulators/<gen>_fea_emulator.joblib)")
    ap.add_argument("--pfn-checkpoint", type=Path, required=True)
    ap.add_argument("--n-init", type=int, default=50)
    ap.add_argument("--n-bo", type=int, default=50)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--q", type=int, default=1)
    ap.add_argument("--out", type=Path, default=Path("sweeps"))
    args = ap.parse_args()

    gen_cls = GENERATORS[args.generator]
    emulator_path = args.emulator or Path(f"emulators/{args.generator}_fea_emulator.joblib")
    print(f"Generator: {args.generator}")
    print(f"Emulator:  {emulator_path}")
    print(f"PFN:       {args.pfn_checkpoint}")
    print(f"Seeds:     {args.seeds}, n_init={args.n_init}, n_bo={args.n_bo}, q={args.q}")
    emulator = FEAEmulator.load(emulator_path)

    args.out.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, list[dict]] = {"GP": [], "PFN": []}
    for s in args.seeds:
        print(f"\n--- seed={s} ---")
        gp_res = _gp_bo_loop(emulator, gen_cls, args.n_init, args.n_bo, s, args.q)
        print(f"  GP : best={gp_res['history'][-1]:.4f} in {gp_res['elapsed_s']:.1f}s")
        pfn_res = _pfn_bo_loop(emulator, gen_cls, args.pfn_checkpoint,
                               args.n_init, args.n_bo, s, args.q)
        print(f"  PFN: best={pfn_res['history'][-1]:.4f} in {pfn_res['elapsed_s']:.1f}s")
        all_results["GP"].append(gp_res)
        all_results["PFN"].append(pfn_res)

    # Aggregate and save.
    n_evals = args.n_init + args.n_bo + 1
    gp_hist = np.stack([np.asarray(r["history"]) for r in all_results["GP"]])
    pfn_hist = np.stack([np.asarray(r["history"]) for r in all_results["PFN"]])

    out_path = args.out / f"{args.generator}_paired_sweep.npz"
    np.savez_compressed(
        out_path,
        gp_history=gp_hist,
        pfn_history=pfn_hist,
        seeds=np.asarray(args.seeds),
        n_init=args.n_init,
        n_bo=args.n_bo,
    )
    print(f"\nSaved {out_path}")

    # Headline numbers.
    print("\nMean best-Y over seeds:")
    print(f"  GP  final = {gp_hist[:, -1].mean():.4f} ± {gp_hist[:, -1].std():.4f}")
    print(f"  PFN final = {pfn_hist[:, -1].mean():.4f} ± {pfn_hist[:, -1].std():.4f}")

    target = 0.95 * gp_hist[:, -1].mean()
    print(f"\n# evaluations to reach 95% of GP final ({target:.4f}):")
    for name, hist in (("GP", gp_hist), ("PFN", pfn_hist)):
        evals_per_seed = []
        for h in hist:
            idx = np.argmax(h >= target)
            evals_per_seed.append(idx if h[idx] >= target else len(h))
        print(f"  {name}: mean {np.mean(evals_per_seed):.1f}, "
              f"per-seed {evals_per_seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
