"""Phase-3: live-ANSYS DSP-GP Bayesian optimization over the 114-D RadialSpline
SynRM rotor parameterisation, maximising T_mean.

Strategy validated in replications/vanilla_hdbo (E3) and applied on the emulator
(E4): vanilla global GP-BO with the dimensionality-scaled LogNormal lengthscale
prior (Hvarfner et al. 2024) + analytic LogEI. The DSP prior is implemented
EXPLICITLY here so the run does not depend on the BoTorch version on the box.

RadialSpline specifics:
  * the repair decoder makes EVERY box point feasible -> no rejection loop;
  * the central d-axis rib is added by split_barriers (offset = w_rod/sqrt2);
  * every ANSYS evaluation is RECORDED IN FULL so we can learn from the runs.

Recording (per CLAUDE.md §6), under results_radialspline_live/<run>/:
  config.json           -- full run config + generator/bounds/DSP-prior params
  evals.csv             -- one row per eval: idx, T_mean, T_ripple, feasible,
                           best_so_far, t_ansys_s, status
  eval_<NNNN>.npz       -- X_norm, X_raw, barriers (object array), Tor (full
                           torque series), T_mean, T_ripple, feasible, t_ansys
  run.log               -- human-readable top-level log

Resumable: on restart, existing eval_*.npz are reloaded into (train_X, train_Y)
and the loop continues from the next index.

Run (on bayes, isolated venv, ANSYS v242):
  export ANSYSEM_ROOT242=/data/Ansys/v242/Linux64/
  ~/Public/PFN/venv_newparam/bin/python notebooks/run_radialspline_live.py \
      --aedt-project data/SynRM_test.aedt --n-init 20 --n-iters 40 --num-cores 4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import MaternKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import LogNormalPrior

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from machine_design import load_design  # noqa: E402
from machine_design.generators import RadialSplineGenerator  # noqa: E402
from machine_design.geometry import analyze_results  # noqa: E402
from machine_design.lumped.geometry import REFERENCE_MACHINE  # noqa: E402

torch.set_default_dtype(torch.float64)
SQRT2, SQRT3 = math.sqrt(2.0), math.sqrt(3.0)


def dsp_gp(u: torch.Tensor, y: torch.Tensor) -> SingleTaskGP:
    """DSP: sqrt(D)-scaled LogNormal lengthscale prior, sigma_f^2=1 (bare Matern).
    Inputs u already in [0,1]^D (no input transform); outputs standardized."""
    d = u.shape[-1]
    prior = LogNormalPrior(loc=SQRT2 + math.log(d) * 0.5, scale=SQRT3)
    kern = MaternKernel(
        nu=2.5, ard_num_dims=d, lengthscale_prior=prior,
        lengthscale_constraint=GreaterThan(2.5e-2, transform=None, initial_value=prior.mode),
    )
    return SingleTaskGP(u, y, covar_module=kern, outcome_transform=Standardize(m=1))


def log(msg, run_dir):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(run_dir / "run.log", "a") as f:
        f.write(line + "\n")


def evaluate_fea(u_row, gen, design, lo, span, num_cores):
    """u in [0,1]^D -> decode (repair) -> rib -> ANSYS -> record everything.
    Returns dict with X_raw, barriers, Tor, T_mean, T_ripple, feasible, t_ansys, status."""
    X_raw = lo + np.asarray(u_row, float) * span
    gen.set_parameters(X_raw)
    barriers = gen.generate_barriers()
    feasible = bool(gen.feasible_barriers(barriers))
    rec = {"X_raw": X_raw, "feasible": feasible, "Tor": None,
           "T_mean": np.nan, "T_ripple": np.nan, "t_ansys": np.nan, "status": ""}
    try:
        barriers_split = gen.split_barriers(barriers)
        rec["barriers"] = np.array(barriers_split, dtype=object)
        design.add_rotor()
        for b in barriers_split:
            design.add_rotor_barrier(b)
        t0 = time.time()
        Tor = design.compute(num_cores)
        rec["t_ansys"] = time.time() - t0
        design.delete_rotor()
        if Tor is None:
            rec["status"] = "ansys_returned_none"
        else:
            Tor = np.asarray(Tor, float)
            TorAvg, _, TorRippleRms = analyze_results(Tor)
            rec.update(Tor=Tor, T_mean=float(TorAvg), T_ripple=float(TorRippleRms), status="ok")
    except Exception as e:  # log, don't hide (CLAUDE.md §9 no silent fallbacks)
        rec["status"] = f"exception:{type(e).__name__}:{e}"
        try:
            design.delete_rotor()
        except Exception:
            pass
        if "barriers" not in rec:
            rec["barriers"] = np.array(barriers, dtype=object)
    return rec


def save_eval(run_dir, idx, u_row, rec):
    np.savez(
        run_dir / f"eval_{idx:04d}.npz",
        idx=idx, X_norm=np.asarray(u_row, float), X_raw=rec["X_raw"],
        barriers=rec.get("barriers", np.array([], dtype=object)),
        Tor=(rec["Tor"] if rec["Tor"] is not None else np.array([])),
        T_mean=rec["T_mean"], T_ripple=rec["T_ripple"],
        feasible=rec["feasible"], t_ansys=rec["t_ansys"], status=rec["status"],
        allow_pickle=True,
    )


def load_existing(run_dir, D):
    """Reload prior evals for resume -> (U list, Tmean list, next_idx)."""
    U, Y = [], []
    files = sorted(run_dir.glob("eval_*.npz"))
    for fp in files:
        d = np.load(fp, allow_pickle=True)
        U.append(d["X_norm"].astype(float))
        Y.append(float(d["T_mean"]))
    return U, Y, len(files)


FALLBACK_TMEAN = 0.0  # ANSYS-failed eval -> minimal objective (logged, not hidden)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aedt-project", required=True, help="path to SynRM_test.aedt (a COPY, not the pfn worktree's)")
    ap.add_argument("--project-name", default="SynRM_test")
    ap.add_argument("--design-name", default="Design01")
    ap.add_argument("--aedt-version", default="2024.2")
    ap.add_argument("--run-name", default="OneLambda_dsp_live")
    ap.add_argument("--n-init", type=int, default=20)
    ap.add_argument("--n-iters", type=int, default=40)
    ap.add_argument("--warmstart-npz", default=None,
                    help="re-encoded RadialSpline designs (X_rs,keep,T) to warm-start init")
    ap.add_argument("--n-warmstart", type=int, default=0,
                    help="number of warm-start designs spanning Hackl T; replaces Sobol init")
    ap.add_argument("--num-cores", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run_dir = Path("results_radialspline_live") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    gen = RadialSplineGenerator(REFERENCE_MACHINE)
    lo_t, hi_t = gen.bounds
    lo, span = lo_t.astype(float), (hi_t - lo_t).astype(float)
    D = lo.shape[0]

    json.dump({
        "run_name": args.run_name, "D": D, "N": gen.N, "K": gen.K,
        "aedt_version": args.aedt_version, "aedt_project": args.aedt_project,
        "n_init": args.n_init, "n_iters": args.n_iters, "num_cores": args.num_cores,
        "seed": args.seed, "dsp_prior": {"loc": SQRT2 + math.log(D) / 2, "scale": SQRT3},
        "structural": {"t_bridge": gen.t_bridge, "w_rod": gen.w_rod, "t_rib": gen.t_rib,
                       "t_shaft": gen.t_shaft, "min_air": gen.min_air},
        "bounds_lo": lo.tolist(), "bounds_hi": hi_t.astype(float).tolist(),
    }, open(run_dir / "config.json", "w"), indent=2)

    log(f"Phase-3 live ANSYS DSP-GP-BO | D={D} | run={args.run_name} | AEDT {args.aedt_version}", run_dir)
    design = load_design(args.aedt_project, args.project_name, args.design_name, args.aedt_version)
    log("AEDT project loaded", run_dir)

    U, Y, next_idx = load_existing(run_dir, D)
    if next_idx:
        log(f"resuming: {next_idx} existing evals reloaded", run_dir)

    unit = torch.stack([torch.zeros(D), torch.ones(D)])
    csv_path = run_dir / "evals.csv"
    if not csv_path.exists():
        csv_path.write_text("idx,T_mean,T_ripple,feasible,best_so_far,t_ansys_s,status\n")

    def record(idx, u_row, rec):
        save_eval(run_dir, idx, u_row, rec)
        y = rec["T_mean"] if rec["status"] == "ok" else FALLBACK_TMEAN
        U.append(np.asarray(u_row, float))
        Y.append(float(y))
        best = max(Y)
        with open(csv_path, "a") as f:
            f.write(f"{idx},{rec['T_mean']:.6f},{rec['T_ripple']:.6f},{int(rec['feasible'])},"
                    f"{best:.6f},{rec['t_ansys']:.1f},{rec['status']}\n")
        log(f"eval {idx:04d}: T_mean={rec['T_mean']:.4f} T_ripple={rec['T_ripple']:.4f} "
            f"feasible={rec['feasible']} best={best:.4f} t_ansys={rec['t_ansys']:.0f}s [{rec['status']}]", run_dir)

    # --- build init points: warm-start (re-encoded known-good) or Sobol ---
    if args.n_warmstart > 0 and args.warmstart_npz:
        dd = np.load(args.warmstart_npz, allow_pickle=True)
        Xw, keepw, Tw = dd["X_rs"], dd["keep"], dd["T"]
        Xw, Tw = Xw[keepw.astype(bool)], Tw[keepw.astype(bool)]
        sel = np.argsort(Tw)[np.linspace(0, len(Tw) - 1, args.n_warmstart).astype(int)]  # spread across T
        Uinit = np.clip((Xw[sel] - lo) / span, 0.0, 1.0)
        n_init_designs = args.n_warmstart
        log(f"warm-start: {args.n_warmstart} re-encoded designs spanning Hackl T "
            f"[{float(Tw[sel].min()):.2f},{float(Tw[sel].max()):.2f}] N·m", run_dir)
    else:
        Uinit = draw_sobol_samples(bounds=unit, n=args.n_init, q=1, seed=args.seed).squeeze(1).numpy()
        n_init_designs = args.n_init

    # --- evaluate remaining init points (resume-aware) ---
    for i in range(next_idx, n_init_designs):
        record(i, Uinit[i], evaluate_fea(Uinit[i], gen, design, lo, span, args.num_cores))

    # --- BO iterations ---
    total = n_init_designs + args.n_iters
    while len(U) < total:
        Ut = torch.tensor(np.stack(U))
        Yt = torch.tensor(Y).unsqueeze(-1)
        gp = dsp_gp(Ut, Yt)
        fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
        acqf = LogExpectedImprovement(gp, best_f=Yt.max())
        cand, _ = optimize_acqf(acqf, bounds=unit, q=1, num_restarts=8, raw_samples=512)
        u = cand.squeeze(0).numpy()
        idx = len(U)
        log(f"--- BO iter {idx - n_init_designs + 1}/{args.n_iters} (n={len(U)}) proposing eval {idx} ---", run_dir)
        record(idx, u, evaluate_fea(u, gen, design, lo, span, args.num_cores))

    log(f"=== done: {len(U)} evals, best T_mean = {max(Y):.4f} ===", run_dir)
    design.close_project()


if __name__ == "__main__":
    main()
