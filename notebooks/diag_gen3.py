"""Post-run sample-sufficiency diagnostic for gen-3 (no FEA).

Answers the question the user asked: "was n-paths enough?" for the pathwise-Thompson outer.
At the FINAL fitted joint GP it sweeps the number of coherent sample paths S and measures
two things, bootstrapped:
  (A) HV(S): the hypervolume of the AGGREGATED Thompson Pareto front (union of S paths'
      non-dominated geometries) vs a fixed reference. As S grows this rises then PLATEAUS;
      the plateau S is the sufficient sample count. If the run's --n-paths sits on the flat
      part, it was enough.
  (B) per-path HV coefficient-of-variation (the same live metric gen3 logs each round),
      reported per S -- a second, decision-level read on Monte-Carlo noise.
Usage:  python notebooks/diag_gen3.py [run_dir]
"""
import sys

import gen2
import gen3
import h0h1_par as P
import h0h1_study as H
import numpy as np
import torch
from botorch.sampling.pathwise import draw_matheron_paths

OUT = sys.argv[1] if len(sys.argv) > 1 else "results/gen3"
z = np.load(f"{OUT}/gen3.npz", allow_pickle=True)
Xg, Xi, T, R = [np.array(z[k]) for k in ("Xg", "Xi", "T", "R")]
demands = [20.0, 35.0, 30.0]
NG, NI = 256, 512
S_GRID = [16, 32, 64, 128, 256, 512, 1024]
BOOT = 6
print(f"[diag3] {OUT}: {len(T)} pooled points, dim {Xg.shape[1] + Xi.shape[1]}", flush=True)


def dq_of(u):
    return gen2.ICUR_LB + u * (gen2.ICUR_UB - gen2.ICUR_LB)


# fit final GP + build a fixed geometry/current candidate grid (shared across all S)
m = gen2.fit_joint(np.hstack([Xg, Xi]), T, gen2.ptp_of(R, T))
gd = P.open_isolated_design("d3", 91, "2024.2", slots=60, phases=5)
gen = P.make_generator(gd, False); lb, ub = H.geom_bounds_arrays(gen)


def feasible_g():
    for _ in range(2000):
        gn = np.asarray(H.rand_feasible_geom_norm(gen, lb, ub), float)
        if H.build_barriers(gen, gn, lb, ub) is not None:
            return gn
    return np.random.rand(gen2.DIM_G)


np.random.seed(0); torch.manual_seed(0)
Gcand = np.array([feasible_g() for _ in range(NG)])
Icand_u = torch.quasirandom.SobolEngine(gen2.DIM_I, scramble=True, seed=7).draw(NI).numpy()
Icand_dq = np.array([dq_of(u) for u in Icand_u])
ipk_cand = np.array([gen2.ipk_of(d) for d in Icand_dq])
loss_cand = np.sum(Icand_dq ** 2, axis=1)
Xq = torch.tensor(np.hstack([np.repeat(Gcand, NI, axis=0), np.tile(Icand_u, (NG, 1))]))

# draw the largest batch ONCE; subsample for smaller S (nested, fair)
Smax = max(S_GRID)
with torch.no_grad():
    pT = draw_matheron_paths(m.models[0], sample_shape=torch.Size([Smax]))
    pP = draw_matheron_paths(m.models[1], sample_shape=torch.Size([Smax]))
    Tall = pT(Xq).numpy().reshape(Smax, NG, NI)
    Pall = np.clip(pP(Xq).numpy(), 0.0, None).reshape(Smax, NG, NI)

# per-path objective pair + non-dominated mask, computed once for all Smax paths
per_path = []
for s in range(Smax):
    cl, mr, _ = gen3.paretoA_objectives(Tall[s], Pall[s], ipk_cand, loss_cand, demands)
    obj = np.stack([cl, mr], 1)
    from botorch.utils.multi_objective.pareto import is_non_dominated
    nd = is_non_dominated(torch.tensor(-obj, dtype=torch.double)).numpy()
    per_path.append(obj[nd])

# fixed reference from ALL paths' Pareto points (so HV is comparable across S)
allpts = np.vstack(per_path)
ref = allpts.max(0) * 1.05

print("\n=== (A) HV(S): aggregated-front hypervolume vs #paths (plateau => sufficient) ===")
print("   S    HV_mean   HV_sd    Δ%vs-prev   per-path HV_cov")
prev = None
for S in S_GRID:
    aggr_hv, pp_cov = [], []
    for _b in range(BOOT):
        idx = np.random.choice(Smax, S, replace=False)
        union = np.vstack([per_path[i] for i in idx])
        aggr_hv.append(gen3.hv_of(union, ref))
        pph = np.array([gen3.hv_of(per_path[i], ref) for i in idx])
        pp_cov.append(pph.std() / pph.mean() if pph.mean() > 0 else np.nan)
    hm, hs = float(np.mean(aggr_hv)), float(np.std(aggr_hv))
    d = "" if prev is None else f"{100 * (hm - prev) / prev:+.2f}%"
    print(f"  {S:5d}  {hm:8.3g}  {hs:7.2g}   {d:>9}   {np.nanmean(pp_cov):.3f}")
    prev = hm

print("\n[diag3] Read: the smallest S whose Δ%vs-prev is within noise (HV_sd) is sufficient;")
print("[diag3] if the run's --n-paths is at/above it, sampling was adequate.", flush=True)
