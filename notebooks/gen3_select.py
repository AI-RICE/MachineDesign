"""Pure GP + pathwise-Thompson SELECTION step for gen3, isolated in its own process.

Imports ONLY numpy/torch/botorch/gpytorch -- NO gen2, NO h0h1_*, NO PyAEDT. This is
deliberate: gen3's main process holds an open AEDT/gRPC session (for geometry generation),
and running torch's Matheron/autograd machinery in that same process segfaults it (torch
vs ANSYS native BLAS/OpenMP). So main hands the GP data + geometry candidates to THIS module
as a subprocess; here we fit the joint GP, draw coherent Matheron sample paths, run the
option-A Pareto inner + HV-greedy batch, and write the picks + varlog + confirmed front back.

CLI:  python gen3_select.py <job.npz> <out.json>
job.npz keys: Xg,Xi,T,R (pool); Gcand,Icand_u,ipk_cand,loss_cand; demands; i_max,lam,n_paths,
q,seed. out.json: {picks:[[gi, [icand_idx per demand]]...], varlog:{...}, front:[[cl,ptp,pct]]}.
"""
import json
import math
import sys

import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms import Standardize
from botorch.sampling.pathwise import draw_matheron_paths
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import SumMarginalLogLikelihood
from gpytorch.priors import LogNormalPrior

SQRT2, SQRT3 = math.sqrt(2.0), math.sqrt(3.0)


def dsp_gp(X, Y):  # identical kernel/prior to gen2.dsp_gp (kept pure here)
    d = X.shape[-1]
    prior = LogNormalPrior(loc=SQRT2 + math.log(d) * 0.5, scale=SQRT3)
    kern = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=d, lengthscale_prior=prior,
                                    lengthscale_constraint=GreaterThan(2.5e-2, transform=None,
                                                                       initial_value=prior.mode)))
    return SingleTaskGP(X, Y, covar_module=kern, outcome_transform=Standardize(m=1))


def fit_joint(XI, T, PTP):
    X = torch.tensor(np.asarray(XI))
    gt = dsp_gp(X, torch.tensor(np.nan_to_num(T, nan=0.0)).unsqueeze(-1))
    gp = dsp_gp(X, torch.tensor(np.nan_to_num(PTP, nan=1.0e3)).unsqueeze(-1))
    m = ModelListGP(gt, gp)
    fit_gpytorch_mll(SumMarginalLogLikelihood(m.likelihood, m))
    return m


def ptp_of(R, T):
    return np.asarray(R, float) * np.asarray(T, float) / 100.0


def paretoA_objectives(Ts, PTPs, ipk_cand, loss_cand, demands, i_max, lam):
    """Ts,PTPs [G,I]. Returns (cycle_loss[G], max_ripple[G], feas[G]). Ripple READ OFF at the
    per-demand loss-optimal current (option A), not constrained."""
    pen_ipk = lam * np.clip(ipk_cand - i_max, 0.0, None)
    G = Ts.shape[0]
    Fk = np.empty((len(demands), G)); Rk = np.empty((len(demands), G))
    feask = np.ones((len(demands), G), bool); ar = np.arange(G)
    for k, Tk in enumerate(demands):
        pen = loss_cand[None, :] + pen_ipk[None, :] + lam * np.clip(Tk - Ts, 0.0, None)
        j = np.argmin(pen, axis=1)
        Fk[k] = pen[ar, j]; Rk[k] = PTPs[ar, j]
        feask[k] = (Ts[ar, j] >= Tk) & (ipk_cand[j] <= i_max)
    return Fk.mean(0), Rk.max(0), feask.all(0)


def hv_of(obj, ref):
    Y = torch.tensor(-obj, dtype=torch.double); rp = torch.tensor(-ref, dtype=torch.double)
    nd = is_non_dominated(Y)
    return float(Hypervolume(ref_point=rp).compute(Y[nd])) if nd.any() else 0.0


def greedy_hv_batch(cand_obj, ref, q):
    rp = torch.tensor(-ref, dtype=torch.double); hv = Hypervolume(ref_point=rp)
    chosen, cur, remaining = [], 0.0, list(range(len(cand_obj)))
    while remaining and len(chosen) < q:
        best_gain, best_i = -1.0, None
        for i in remaining:
            sel = torch.tensor(-cand_obj[chosen + [i]], dtype=torch.double)
            nd = is_non_dominated(sel)
            gain = (float(hv.compute(sel[nd])) if nd.any() else 0.0) - cur
            if gain > best_gain:
                best_gain, best_i = gain, i
        if best_i is None or best_gain <= 0.0:
            break
        chosen.append(best_i); remaining.remove(best_i); cur += best_gain
    return chosen


def confirmed_front(Xg, Xi, T, R, dq_of, demands, i_max):
    """Non-dominated (cycle_loss, max_ptp[Nm], ripple%_at_worst) over FEA-confirmed geometries."""
    Xg = np.array(Xg); key = {}
    for i in range(len(T)):
        key.setdefault(tuple(np.round(Xg[i], 6)), []).append(i)
    obj, pct = [], []
    for _g, ids in key.items():
        dq = np.array([dq_of(Xi[i]) for i in ids])
        tt = np.array([T[i] for i in ids]); rr = np.array([R[i] for i in ids])
        ptp = ptp_of(rr, tt); ll = np.sum(dq ** 2, axis=1)
        ip = np.array([peak_current(dq[j]) for j in range(len(dq))])
        base = ip <= i_max; cl, mp, mpct, ok = 0.0, 0.0, 0.0, True
        for Tk in demands:
            fe = base & (tt >= Tk)
            if not fe.any():
                ok = False; break
            j = int(np.where(fe)[0][np.argmin(ll[fe])]); cl += ll[j] / len(demands)
            if ptp[j] > mp:
                mp, mpct = ptp[j], rr[j]
        if ok:
            obj.append((cl, mp)); pct.append(mpct)
    if not obj:
        return []
    O = np.array(obj); pct = np.array(pct)
    nd = is_non_dominated(torch.tensor(-O, dtype=torch.double)).numpy()
    return [[float(O[i, 0]), float(O[i, 1]), float(pct[i])] for i in np.where(nd)[0]]


# peak-current waveform (copied pure from h0h1_par.peak_current_from_dq; theta grid passed in)
_THETA = None


def peak_current(dq):
    Id1, Iq1, Id3, Iq3 = dq
    Im1, a1 = math.hypot(Id1, Iq1), math.atan2(Iq1, Id1)
    Im3, a3 = math.hypot(Id3, Iq3), math.atan2(Iq3, Id3)
    return float(np.max(np.abs(Im1 * np.cos(_THETA + a1) + Im3 * np.cos(3.0 * _THETA + a3))))


def main():
    job, outp = sys.argv[1], sys.argv[2]
    z = np.load(job, allow_pickle=True)
    Xg, Xi, T, R = z["Xg"], z["Xi"], z["T"], z["R"]
    Gcand, Icand_u = z["Gcand"], z["Icand_u"]
    ipk_cand, loss_cand = z["ipk_cand"], z["loss_cand"]
    icur_lb, icur_ub = z["icur_lb"], z["icur_ub"]
    demands = list(z["demands"]); i_max = float(z["i_max"]); lam = float(z["lam"])
    n_paths, q, seed = int(z["n_paths"]), int(z["q"]), int(z["seed"])
    global _THETA
    _THETA = z["theta"]
    NG, NI = len(Gcand), len(Icand_u)
    np.random.seed(seed); torch.manual_seed(seed)

    def dq_of(u):
        return icur_lb + np.asarray(u) * (icur_ub - icur_lb)

    if n_paths == 0:                     # front-only mode (final report; no GP/Matheron)
        front = confirmed_front(Xg, Xi, T, R, dq_of, demands, i_max)
        json.dump(dict(picks=[], varlog={}, front=front), open(outp, "w"))
        return

    m = fit_joint(np.hstack([Xg, Xi]), T, ptp_of(R, T))
    GX = np.repeat(Gcand, NI, axis=0); IX = np.tile(Icand_u, (NG, 1))
    Xq = torch.tensor(np.hstack([GX, IX]))

    def eval_paths_chunked(path, Xq, block=512):
        """Evaluate S sample paths at N points WITHOUT materialising the [S,N,features]
        intermediate (which is ~270 GB at S=256, N=131072, F~1024 and OOM-kills the box).
        Chunk over points: each block holds only [S, block, F]. block=512 (one geometry's
        currents) -> ~1 GB/block regardless of the total grid size."""
        N = Xq.shape[0]
        out = np.empty((n_paths, N), dtype=np.float64)
        for s in range(0, N, block):
            e = min(s + block, N)
            out[:, s:e] = path(Xq[s:e]).numpy()
        return out

    with torch.no_grad():
        pT = draw_matheron_paths(m.models[0], sample_shape=torch.Size([n_paths]))
        pP = draw_matheron_paths(m.models[1], sample_shape=torch.Size([n_paths]))
        Tsmp = eval_paths_chunked(pT, Xq).reshape(n_paths, NG, NI)
        Psmp = np.clip(eval_paths_chunked(pP, Xq), 0.0, None).reshape(n_paths, NG, NI)

    cand_gi, cand_obj, path_hv, path_argmin = [], [], [], []
    for s in range(n_paths):
        cl, mr, _ = paretoA_objectives(Tsmp[s], Psmp[s], ipk_cand, loss_cand, demands, i_max, lam)
        obj = np.stack([cl, mr], 1); path_argmin.append(int(np.argmin(cl)))
        nd = is_non_dominated(torch.tensor(-obj, dtype=torch.double)).numpy()
        for gi in np.where(nd)[0]:
            cand_gi.append(int(gi)); cand_obj.append(obj[gi])
        path_hv.append(obj[nd])
    cand_obj = np.array(cand_obj); ref = cand_obj.max(0) * 1.05
    hvs = np.array([hv_of(o, ref) for o in path_hv])
    hv_mean, hv_sd = float(hvs.mean()), float(hvs.std())
    agree = float(np.mean(np.array(path_argmin) == np.bincount(path_argmin).argmax()))
    varlog = dict(n_paths=n_paths, hv_mean=hv_mean, hv_sd=hv_sd,
                  hv_cov=(hv_sd / hv_mean if hv_mean > 0 else None),
                  top_geom_agreement=agree, n_pareto_cand=int(len(cand_obj)))

    pick = greedy_hv_batch(cand_obj, ref, q)
    picked_gi = list(dict.fromkeys(cand_gi[i] for i in pick))
    # posterior MEAN only for the PICKED geometries (a handful). Evaluating m.posterior over the
    # FULL Gcand x Icand grid makes GPyTorch materialise a joint [N,N] covariance (~137 GB at
    # N=131072) and OOMs -- but we only need the picked geometries' per-demand currents, so
    # restrict Xq to them (<=q*NI points, ~130 MB).
    Gpick = np.array([Gcand[gi] for gi in picked_gi])
    Xq_pick = torch.tensor(np.hstack([np.repeat(Gpick, NI, axis=0), np.tile(Icand_u, (len(picked_gi), 1))]))
    with torch.no_grad():
        mu = m.posterior(Xq_pick).mean.numpy()
    muT = mu[:, 0].reshape(len(picked_gi), NI)
    muP = np.clip(mu[:, 1].reshape(len(picked_gi), NI), 0.0, None)  # noqa: F841
    pen_ipk = lam * np.clip(ipk_cand - i_max, 0.0, None)
    picks = []
    for k_idx, gi in enumerate(picked_gi):
        idxs = []
        for Tk in demands:
            pen = loss_cand + pen_ipk + lam * np.clip(Tk - muT[k_idx], 0.0, None)
            idxs.append(int(np.argmin(pen)))
        picks.append([gi, idxs])

    front = confirmed_front(Xg, Xi, T, R, dq_of, demands, i_max)
    json.dump(dict(picks=picks, varlog=varlog, front=front), open(outp, "w"))


if __name__ == "__main__":
    main()
