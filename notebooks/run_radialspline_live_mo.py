"""Phase-3 MULTI-OBJECTIVE (E6): live-ANSYS DSP-GP-BO over the 114-D RadialSpline
space, optimising the (T_mean ↑, T_ripple ↓) Pareto front with qLogEHVI, warm-
started from a POOLED set of re-encoded Hackl designs (all three parameterisations)
for design variability.

Objectives (maximised): f = (T_mean, -T_ripple/100)  -- the ICEM convention.
Surrogate: ModelListGP of two independent DSP-prior GPs (sqrt(D) LogNormal
lengthscale prior, Hvarfner 2024; explicit, version-independent).
Repair decoder => every box point feasible (no rejection loop).

Warm-start: the pooled Pareto front + a gen-stratified diverse fill, so the GP
sees the best known trade-offs across OneLambda/SixLambdas/ThreeBrokenLines.

Recording (CLAUDE.md §6): per-eval npz (geometry, full torque series, both
objectives), evals.csv (incl. hypervolume), run.log. Resumable.

Run (bayes, isolated venv, v242):
  export ANSYSEM_ROOT242=/data/Ansys/v242/Linux64/
  ~/Public/PFN/venv_newparam/bin/python notebooks/run_radialspline_live_mo.py \
    --aedt-project data/SynRM_test.aedt --aedt-version 2024.2 \
    --pooled-npz notebooks/RadialSpline_reencoded_pooled.npz \
    --n-warmstart 60 --n-iters 140 --num-cores 4
Local machinery test (no ANSYS):  add  --mock
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import ModelListGP
from botorch.optim import optimize_acqf
from gpytorch.mlls import SumMarginalLogLikelihood
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_radialspline_live import dsp_gp, evaluate_fea, log, save_eval  # noqa: E402

from machine_design.generators import RadialSplineGenerator  # noqa: E402
from machine_design.lumped.geometry import REFERENCE_MACHINE  # noqa: E402

torch.set_default_dtype(torch.float64)
MO_FALLBACK = (0.0, 100.0)  # failed ANSYS -> worst (T_mean, T_ripple)


def to_obj(t_mean, t_ripple):
    """(T_mean, T_ripple) -> maximisation objectives (T_mean, -T_ripple/100)."""
    return np.array([t_mean, -t_ripple / 100.0])


def select_warmstart(Tm, Tr, gid, n, seed):
    """Pooled Pareto front + gen-stratified diverse fill -> chosen indices."""
    f = np.column_stack([Tm, -Tr / 100.0])
    nd = is_non_dominated(torch.tensor(f)).numpy()
    chosen = list(np.where(nd)[0])
    rng = np.random.default_rng(seed)
    if len(chosen) > n:
        chosen = list(rng.choice(chosen, n, replace=False))
    else:
        remaining = n - len(chosen)
        pool = np.setdiff1d(np.arange(len(Tm)), chosen)
        per_gen = remaining // 3
        for g in (0, 1, 2):
            cand = pool[gid[pool] == g]
            k = min(per_gen, len(cand))
            if k:
                chosen += list(rng.choice(cand, k, replace=False))
        # top up any shortfall
        pool2 = np.setdiff1d(np.arange(len(Tm)), chosen)
        short = n - len(chosen)
        if short > 0 and len(pool2):
            chosen += list(rng.choice(pool2, min(short, len(pool2)), replace=False))
    return np.array(sorted(chosen))


def mock_objective_factory(Xpool, Tm, Tr):
    """Nearest-neighbour lookup in the pooled re-encoded set (local test only)."""
    def f(X_canon):
        j = int(np.argmin(np.linalg.norm(Xpool - X_canon[None, :], axis=1)))
        return {"T_mean": float(Tm[j]), "T_ripple": float(Tr[j]), "feasible": True,
                "Tor": np.array([]), "t_ansys": 0.0, "status": "mock", "X_raw": X_canon,
                "barriers": np.array([], dtype=object)}
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aedt-project", default="data/SynRM_test.aedt")
    ap.add_argument("--project-name", default="SynRM_test")
    ap.add_argument("--design-name", default="Design01")
    ap.add_argument("--aedt-version", default="2024.2")
    ap.add_argument("--pooled-npz", required=True)
    ap.add_argument("--run-name", default="pooled_mo_live")
    ap.add_argument("--n-warmstart", type=int, default=60)
    ap.add_argument("--n-iters", type=int, default=140)
    ap.add_argument("--num-cores", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mock", action="store_true", help="local test: NN-lookup objective, no ANSYS")
    args = ap.parse_args()

    run_dir = Path("results_radialspline_live") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    gen = RadialSplineGenerator(REFERENCE_MACHINE)
    lo_t, hi_t = gen.bounds
    lo, span = lo_t.astype(float), (hi_t - lo_t).astype(float)
    D = lo.shape[0]

    d = np.load(args.pooled_npz, allow_pickle=True)
    Xpool, Tm, Tr, gid = d["X_rs"], d["T_mean"], d["T_ripple"], d["gen_id"]
    ws_idx = select_warmstart(Tm, Tr, gid, args.n_warmstart, args.seed)
    ng = [int((gid[ws_idx] == g).sum()) for g in (0, 1, 2)]
    Uws = np.clip((Xpool[ws_idx] - lo) / span, 0.0, 1.0)

    # reference point for HV (dominated by all designs): worst objective minus margin
    fpool = np.column_stack([Tm, -Tr / 100.0])
    ref = torch.tensor([float(fpool[:, 0].min()) - 0.1, float(fpool[:, 1].min()) - 0.1])

    json.dump({"run_name": args.run_name, "D": D, "objectives": "(T_mean, -T_ripple/100)",
               "n_warmstart": args.n_warmstart, "warmstart_per_gen": {"OneLambda": ng[0],
               "SixLambdas": ng[1], "ThreeBrokenLines": ng[2]}, "n_iters": args.n_iters,
               "ref_point": ref.tolist(), "aedt_version": args.aedt_version, "mock": args.mock,
               "seed": args.seed}, open(run_dir / "config.json", "w"), indent=2)
    log(f"MO Phase-3 | D={D} | warm-start {args.n_warmstart} (1λ={ng[0]},6λ={ng[1]},3BL={ng[2]}) "
        f"| ref={ref.tolist()} | mock={args.mock}", run_dir)

    if args.mock:
        objective = mock_objective_factory(Xpool, Tm, Tr)
        design = None
    else:
        from machine_design import load_design
        design = load_design(args.aedt_project, args.project_name, args.design_name, args.aedt_version)
        log("AEDT project loaded", run_dir)

    # resume
    U, Ymean, Yrip = [], [], []
    existing = sorted(run_dir.glob("eval_*.npz"))
    for fp in existing:
        e = np.load(fp, allow_pickle=True)
        U.append(e["X_norm"].astype(float)); Ymean.append(float(e["T_mean"])); Yrip.append(float(e["T_ripple"]))
    next_idx = len(existing)
    if next_idx:
        log(f"resuming: {next_idx} existing evals", run_dir)

    csv_path = run_dir / "evals.csv"
    if not csv_path.exists():
        csv_path.write_text("idx,phase,T_mean,T_ripple,f1,f2,feasible,hypervolume,t_ansys_s,status\n")
    hv_engine = Hypervolume(ref_point=ref)

    def Yobj():
        return torch.tensor([to_obj(m, r) for m, r in zip(Ymean, Yrip)])

    def record(idx, u_row, rec, phase):
        save_eval(run_dir, idx, u_row, rec)
        m = rec["T_mean"] if rec["status"] in ("ok", "mock") else MO_FALLBACK[0]
        r = rec["T_ripple"] if rec["status"] in ("ok", "mock") else MO_FALLBACK[1]
        U.append(np.asarray(u_row, float)); Ymean.append(float(m)); Yrip.append(float(r))
        Y = Yobj()
        pareto = Y[is_non_dominated(Y)]
        hv = float(hv_engine.compute(pareto)) if len(pareto) else 0.0
        f1, f2 = to_obj(m, r)
        with open(csv_path, "a") as fcsv:
            fcsv.write(f"{idx},{phase},{rec['T_mean']:.6f},{rec['T_ripple']:.6f},{f1:.6f},{f2:.6f},"
                       f"{int(rec['feasible'])},{hv:.6f},{rec['t_ansys']:.1f},{rec['status']}\n")
        log(f"eval {idx:04d} [{phase}]: T_mean={rec['T_mean']:.4f} ripple={rec['T_ripple']:.2f}% "
            f"HV={hv:.4f} t={rec['t_ansys']:.0f}s [{rec['status']}]", run_dir)

    def evaluate(u_row):
        if args.mock:
            gen.set_parameters(lo + np.asarray(u_row) * span)
            X_canon = gen.fit_barriers(gen.generate_barriers())
            return objective(X_canon)
        return evaluate_fea(u_row, gen, design, lo, span, args.num_cores)

    # warm-start
    for i in range(next_idx, len(ws_idx)):
        record(i, Uws[i], evaluate(Uws[i]), "warmstart")

    # MO BO with qLogEHVI
    unit = torch.stack([torch.zeros(D), torch.ones(D)])
    total = len(ws_idx) + args.n_iters
    while len(U) < total:
        Ut = torch.tensor(np.stack(U)); Y = Yobj()
        models = [dsp_gp(Ut, Y[:, j: j + 1]) for j in (0, 1)]
        ml = ModelListGP(*models)
        fit_gpytorch_mll(SumMarginalLogLikelihood(ml.likelihood, ml))
        part = NondominatedPartitioning(ref_point=ref, Y=Y[is_non_dominated(Y)])
        acq = qLogExpectedHypervolumeImprovement(model=ml, ref_point=ref.tolist(), partitioning=part)
        cand, _ = optimize_acqf(acq, bounds=unit, q=1, num_restarts=8, raw_samples=512)
        idx = len(U)
        log(f"--- MO iter {idx - len(ws_idx) + 1}/{args.n_iters} (n={len(U)}) ---", run_dir)
        record(idx, cand.squeeze(0).numpy(), evaluate(cand.squeeze(0).numpy()), "bo")

    Yf = Yobj()
    log(f"=== done: {len(U)} evals, final HV={float(hv_engine.compute(Yf[is_non_dominated(Yf)])):.4f} ===", run_dir)
    if design is not None:
        design.close_project()


if __name__ == "__main__":
    main()
