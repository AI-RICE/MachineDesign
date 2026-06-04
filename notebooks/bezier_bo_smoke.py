"""Step-4 smoke test (no FEA): wire the Bézier-superset MO-BO loop end-to-end on a
MOCK objective (nearest-neighbour lookup in the pooled re-encoded set) to prove:
  (1) the data-driven warm-start box + DSP-GP ModelListGP + qLogEHVI runs,
  (2) `optimize_acqf_feasible` proposes ONLY feasible candidates (the validator
      is enforced as a hard constraint, no second GP),
  (3) hypervolume is non-decreasing across iterations.

  .venv/bin/python notebooks/bezier_bo_smoke.py --n-warmstart 30 --n-iters 15
"""

import argparse
import os
import sys

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

from run_radialspline_live import dsp_gp  # noqa: E402

from machine_design.bezier_bo import (  # noqa: E402
    decode_feasible,
    optimize_acqf_feasible,
    to_unit,
    warmstart_box,
)
from machine_design.bezier_generator import BezierSupersetGenerator  # noqa: E402
from machine_design.lumped.geometry import REFERENCE_MACHINE  # noqa: E402

torch.set_default_dtype(torch.float64)


def to_obj(tm, tr):
    return np.array([tm, -tr / 100.0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pooled-npz", default=os.path.join(HERE, "Bezier_reencoded_pooled.npz"))
    ap.add_argument("--n-warmstart", type=int, default=30)
    ap.add_argument("--n-iters", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    d = np.load(args.pooled_npz, allow_pickle=True)
    Xpool, Tm, Tr, gid = d["X_bz"], d["T_mean"], d["T_ripple"], d["gen_id"]
    M, D = int(d["M"]), int(d["D"])
    gen = BezierSupersetGenerator(REFERENCE_MACHINE, M=M)
    lo, span = warmstart_box(Xpool, gen)
    Upool = to_unit(Xpool, lo, span)
    print(f"pooled N={len(Xpool)}  D={D}  (1λ={int((gid==0).sum())},6λ={int((gid==1).sum())},"
          f"3BL={int((gid==2).sum())})")

    # warm-start: Pareto front + random fill
    f = np.column_stack([Tm, -Tr / 100.0])
    nd = np.where(is_non_dominated(torch.tensor(f)).numpy())[0]
    chosen = list(nd)
    if len(chosen) < args.n_warmstart:
        pool = np.setdiff1d(np.arange(len(Tm)), chosen)
        chosen += list(rng.choice(pool, args.n_warmstart - len(chosen), replace=False))
    chosen = np.array(sorted(rng.choice(chosen, min(args.n_warmstart, len(chosen)), replace=False)))

    # MOCK objective: nearest pooled design in unit space
    def mock(u):
        j = int(np.argmin(np.linalg.norm(Upool - u[None, :], axis=1)))
        return float(Tm[j]), float(Tr[j])

    fpool = np.column_stack([Tm, -Tr / 100.0])
    ref = torch.tensor([float(fpool[:, 0].min()) - 0.1, float(fpool[:, 1].min()) - 0.1])
    hv_engine = Hypervolume(ref_point=ref)

    U = list(Upool[chosen])
    Ym = list(Tm[chosen]); Yr = list(Tr[chosen])

    def Yobj():
        return torch.tensor([to_obj(m, r) for m, r in zip(Ym, Yr)])

    def hv_now():
        Y = Yobj(); P = Y[is_non_dominated(Y)]
        return float(hv_engine.compute(P)) if len(P) else 0.0

    n_infeasible = 0
    hv0 = hv_now()
    print(f"warm-start {len(U)} designs, HV={hv0:.4f}")
    hv_prev = hv0
    for it in range(args.n_iters):
        Ut = torch.tensor(np.stack(U)); Y = Yobj()
        ml = ModelListGP(*[dsp_gp(Ut, Y[:, j:j + 1]) for j in (0, 1)])
        fit_gpytorch_mll(SumMarginalLogLikelihood(ml.likelihood, ml))
        part = NondominatedPartitioning(ref_point=ref, Y=Y[is_non_dominated(Y)])
        acq = qLogExpectedHypervolumeImprovement(model=ml, ref_point=ref.tolist(), partitioning=part)
        cand, val = optimize_acqf_feasible(acq, gen, lo, span, anchors=np.stack(U), rng=rng)
        u = cand[0]
        feas = decode_feasible(gen, u, lo, span)  # must be True by construction
        n_infeasible += (not feas)
        tm, tr = mock(u)
        U.append(u); Ym.append(tm); Yr.append(tr)
        hv = hv_now()
        print(f"iter {it+1:02d}: acq={val[0]:.3e} feasible={feas} T_mean={tm:.3f} ripple={tr:.2f}% "
              f"HV={hv:.4f} {'+' if hv >= hv_prev - 1e-9 else 'DROP'}")
        hv_prev = hv

    print(f"\n=== smoke result ===")
    print(f"proposed candidates feasible: {args.n_iters - n_infeasible}/{args.n_iters} "
          f"({'PASS' if n_infeasible == 0 else 'FAIL'})")
    print(f"HV warm-start {hv0:.4f} -> final {hv_now():.4f} "
          f"({'improved' if hv_now() >= hv0 else 'no improvement (mock ceiling)'})")


if __name__ == "__main__":
    main()
