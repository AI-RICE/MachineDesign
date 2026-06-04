"""Step-5 MULTI-OBJECTIVE live-ANSYS BO over the unified Bézier-superset space
(D=108), optimising (T_mean ↑, T_ripple ↓) with qLogEHVI, warm-started from the
POOLED re-encoded Hackl designs (all three parameterisations).

Differs from run_radialspline_live_mo.py in two ways that matter:
  1. Bézier has NO repair decoder — feasibility is a HARD CONSTRAINT enforced by
     `optimize_acqf_feasible` (candidates from perturbing feasible anchors; the
     cheap geometry validator filters; no second GP).
  2. FEA runs at the CONVERGED mesh (rotor 0.5 + airgap 0.5, Step 0), not the old
     3 mm default — so v2 numbers carry no mesh confound.

Objectives (maximised): f = (T_mean, -T_ripple/100). Surrogate: ModelListGP of two
DSP-prior GPs (sqrt(D) LogNormal lengthscale, Hvarfner 2024). Recording per
CLAUDE.md §6 (per-eval npz, evals.csv incl. HV, run.log). Resumable.

Run (bayes, isolated venv, v242):
  export ANSYSEM_ROOT242=/data/Ansys/v242/Linux64/
  ~/Public/PFN/venv_newparam/bin/python notebooks/run_bezier_live_mo.py \
    --pooled-npz notebooks/Bezier_reencoded_pooled.npz \
    --n-warmstart 50 --n-iters 200 --num-cores 4
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
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated
from gpytorch.mlls import SumMarginalLogLikelihood

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from run_radialspline_live import dsp_gp, log, save_eval  # noqa: E402
from run_radialspline_live_mo import select_warmstart, to_obj  # noqa: E402

from machine_design.bezier_bo import optimize_acqf_feasible, to_unit, warmstart_box  # noqa: E402
from machine_design.bezier_generator import BezierSupersetGenerator  # noqa: E402
from machine_design.geometry import analyze_results  # noqa: E402
from machine_design.lumped.geometry import REFERENCE_MACHINE  # noqa: E402

torch.set_default_dtype(torch.float64)
MO_FALLBACK = (0.0, 100.0)  # failed/infeasible -> worst (T_mean, T_ripple)


def evaluate_fea_converged(u_row, gen, design, lo, span, num_cores, mesh, airgap):
    """u in [0,1]^D -> decode -> validator -> ANSYS at the CONVERGED mesh.
    Returns the same dict schema as run_radialspline_live.evaluate_fea."""
    X_raw = lo + np.asarray(u_row, float) * span
    gen.set_parameters(X_raw)
    barriers = gen.generate_barriers()
    feasible = bool(gen.feasible_barriers(barriers))
    rec = {"X_raw": X_raw, "feasible": feasible, "Tor": None, "T_mean": np.nan,
           "T_ripple": np.nan, "t_ansys": np.nan, "status": "",
           "barriers": np.array(barriers, dtype=object)}
    if not feasible:  # acqf should never propose this; log if it does, don't hide
        rec["status"] = "infeasible"
        return rec
    try:
        bsplit = gen.split_barriers(barriers)
        rec["barriers"] = np.array(bsplit, dtype=object)
        design.add_rotor()
        for b in bsplit:
            design.add_rotor_barrier(b)
        t0 = time.time()
        Tor = design.compute(num_cores, mesh_length=mesh, airgap_mesh=airgap)
        rec["t_ansys"] = time.time() - t0
        design.delete_rotor()
        if Tor is None:
            rec["status"] = "ansys_returned_none"
        else:
            Tor = np.asarray(Tor, float)
            TorAvg, _, TorRippleRms = analyze_results(Tor)
            rec.update(Tor=Tor, T_mean=float(TorAvg), T_ripple=float(TorRippleRms), status="ok")
    except Exception as e:  # CLAUDE.md §9: log, don't hide
        rec["status"] = f"exception:{type(e).__name__}:{e}"
        try:
            design.delete_rotor()
        except Exception:
            pass
    return rec


def mock_objective(Upool, Tm, Tr):
    def f(u, gen, design, lo, span, *a):
        j = int(np.argmin(np.linalg.norm(Upool - u[None, :], axis=1)))
        return {"X_raw": lo + u * span, "feasible": True, "T_mean": float(Tm[j]),
                "T_ripple": float(Tr[j]), "t_ansys": 0.0, "status": "mock",
                "Tor": np.array([]), "barriers": np.array([], dtype=object)}
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aedt-project", default="data/SynRM_test.aedt")
    ap.add_argument("--project-name", default="SynRM_test")
    ap.add_argument("--design-name", default="Design01")
    ap.add_argument("--aedt-version", default="2024.2")
    ap.add_argument("--pooled-npz", default=os.path.join(HERE, "Bezier_reencoded_pooled.npz"))
    ap.add_argument("--run-name", default="bezier_mo_live")
    ap.add_argument("--n-warmstart", type=int, default=50)
    ap.add_argument("--n-iters", type=int, default=200)
    ap.add_argument("--num-cores", type=int, default=4)
    ap.add_argument("--mesh", type=float, default=0.5)
    ap.add_argument("--airgap", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ref-point", type=float, nargs=2, default=None,
                    help="fixed HV ref (f1 f2); pin to the bar's for comparability")
    ap.add_argument("--n-cand", type=int, default=2048, help="feasible candidate pool per acqf step")
    ap.add_argument("--mock", action="store_true", help="local test: NN-lookup objective, no ANSYS")
    args = ap.parse_args()

    run_dir = Path("results_bezier_live") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    d = np.load(args.pooled_npz, allow_pickle=True)
    Xpool, Tm, Tr, gid = d["X_bz"], d["T_mean"], d["T_ripple"], d["gen_id"]
    M, D = int(d["M"]), int(d["D"])
    gen = BezierSupersetGenerator(REFERENCE_MACHINE, M=M)
    lo, span = warmstart_box(Xpool, gen)
    Upool = to_unit(Xpool, lo, span)

    ws_idx = select_warmstart(Tm, Tr, gid, args.n_warmstart, args.seed)
    ng = [int((gid[ws_idx] == g).sum()) for g in (0, 1, 2)]
    if args.ref_point is not None:
        ref = torch.tensor([float(args.ref_point[0]), float(args.ref_point[1])])
    else:
        fpool = np.column_stack([Tm, -Tr / 100.0])
        ref = torch.tensor([float(fpool[:, 0].min()) - 0.1, float(fpool[:, 1].min()) - 0.1])

    json.dump({"run_name": args.run_name, "D": D, "M": M, "mesh": [args.mesh, args.airgap],
               "objectives": "(T_mean, -T_ripple/100)", "n_warmstart": args.n_warmstart,
               "warmstart_per_gen": {"OneLambda": ng[0], "SixLambdas": ng[1],
               "ThreeBrokenLines": ng[2]}, "n_iters": args.n_iters, "ref_point": ref.tolist(),
               "aedt_version": args.aedt_version, "mock": args.mock, "seed": args.seed},
              open(run_dir / "config.json", "w"), indent=2)
    log(f"Bézier MO Step-5 | D={D} mesh=({args.mesh},{args.airgap}) | warm-start "
        f"{args.n_warmstart} (1λ={ng[0]},6λ={ng[1]},3BL={ng[2]}) | ref={ref.tolist()} "
        f"| mock={args.mock}", run_dir)

    if args.mock:
        evaluate = mock_objective(Upool, Tm, Tr)
        design = None
    else:
        from machine_design import load_design
        design = load_design(args.aedt_project, args.project_name, args.design_name, args.aedt_version)
        log("AEDT project loaded", run_dir)
        evaluate = evaluate_fea_converged

    # resume
    U, Ym, Yr = [], [], []
    existing = sorted(run_dir.glob("eval_*.npz"))
    for fp in existing:
        e = np.load(fp, allow_pickle=True)
        U.append(e["X_norm"].astype(float)); Ym.append(float(e["T_mean"])); Yr.append(float(e["T_ripple"]))
    next_idx = len(existing)
    if next_idx:
        log(f"resuming: {next_idx} existing evals", run_dir)

    csv_path = run_dir / "evals.csv"
    if not csv_path.exists():
        csv_path.write_text("idx,phase,T_mean,T_ripple,f1,f2,feasible,hypervolume,t_ansys_s,status\n")
    hv_engine = Hypervolume(ref_point=ref)

    def Yobj():
        return torch.tensor(np.array([to_obj(m, r) for m, r in zip(Ym, Yr)]))

    def record(idx, u_row, rec, phase):
        save_eval(run_dir, idx, u_row, rec)
        ok = rec["status"] in ("ok", "mock")
        m = rec["T_mean"] if ok else MO_FALLBACK[0]
        r = rec["T_ripple"] if ok else MO_FALLBACK[1]
        U.append(np.asarray(u_row, float)); Ym.append(float(m)); Yr.append(float(r))
        Y = Yobj(); P = Y[is_non_dominated(Y)]
        hv = float(hv_engine.compute(P)) if len(P) else 0.0
        f1, f2 = to_obj(m, r)
        with open(csv_path, "a") as fcsv:
            fcsv.write(f"{idx},{phase},{rec['T_mean']:.6f},{rec['T_ripple']:.6f},{f1:.6f},{f2:.6f},"
                       f"{int(rec['feasible'])},{hv:.6f},{rec.get('t_ansys',0) or 0:.1f},{rec['status']}\n")
        log(f"eval {idx:04d} [{phase}]: T_mean={rec['T_mean']:.4f} ripple={rec['T_ripple']:.2f}% "
            f"HV={hv:.4f} t={rec.get('t_ansys',0) or 0:.0f}s [{rec['status']}]", run_dir)
        return hv

    def do_eval(u):
        if args.mock:
            return evaluate(np.asarray(u, float), gen, design, lo, span)
        return evaluate_fea_converged(u, gen, design, lo, span, args.num_cores, args.mesh, args.airgap)

    # warm-start
    Uws = Upool[ws_idx]
    for i in range(next_idx, len(ws_idx)):
        record(i, Uws[i], do_eval(Uws[i]), "warmstart")

    # MO BO with qLogEHVI + feasibility-constrained acqf
    total = len(ws_idx) + args.n_iters
    while len(U) < total:
        Ut = torch.tensor(np.stack(U)); Y = Yobj()
        ml = ModelListGP(*[dsp_gp(Ut, Y[:, j:j + 1]) for j in (0, 1)])
        fit_gpytorch_mll(SumMarginalLogLikelihood(ml.likelihood, ml))
        part = NondominatedPartitioning(ref_point=ref, Y=Y[is_non_dominated(Y)])
        acq = qLogExpectedHypervolumeImprovement(model=ml, ref_point=ref.tolist(), partitioning=part)
        # anchors = all feasible evaluated designs (fall back to all if none flagged)
        anchors = np.stack(U)
        cand, val = optimize_acqf_feasible(acq, gen, lo, span, anchors=anchors, rng=rng, n_cand=args.n_cand)
        idx = len(U)
        log(f"--- MO iter {idx - len(ws_idx) + 1}/{args.n_iters} (n={len(U)}) acq={val[0]:.3e} ---", run_dir)
        record(idx, cand[0], do_eval(cand[0]), "bo")

    Yf = Yobj()
    log(f"=== done: {len(U)} evals, final HV={float(hv_engine.compute(Yf[is_non_dominated(Yf)])):.4f} ===", run_dir)
    if design is not None:
        design.close_project()


if __name__ == "__main__":
    main()
