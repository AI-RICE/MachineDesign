"""Pure GP + pathwise-Thompson SELECTION step for gen3, isolated in its own process.

Imports ONLY numpy/torch/botorch/gpytorch -- NO gen2, NO h0h1_*, NO PyAEDT. gen3's main
process holds an open AEDT/gRPC session, and torch's Matheron/autograd machinery segfaults
in that same process; so main hands GP data + geometry candidates here as a subprocess.

Joint GP over (g,i) with SIX outputs: T, ptp (abs ripple Nm), and the four flux linkages
Fd1,Fq1,Fd3,Fq3. Option-A inner minimises loss s.t. sampled T>=T_k, Ipk<=I_MAX AND the
VOLTAGE bound V_pk(omega_k) <= V_MAX at each operating point's electrical speed (Lew-aware);
ripple is read off at the loss-optimal current and used as the outer Pareto objective.

CLI:  python gen3_select.py <job.npz> <out.json>
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
_THETA = None
_VGRID = np.linspace(0.0, 2.0 * math.pi, 181, endpoint=False)  # theta grid for exact V-peak


def dsp_gp(X, Y):  # identical kernel/prior to gen2.dsp_gp
    d = X.shape[-1]
    prior = LogNormalPrior(loc=SQRT2 + math.log(d) * 0.5, scale=SQRT3)
    kern = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=d, lengthscale_prior=prior,
                                    lengthscale_constraint=GreaterThan(2.5e-2, transform=None,
                                                                       initial_value=prior.mode)))
    return SingleTaskGP(X, Y, covar_module=kern, outcome_transform=Standardize(m=1))


def fit_joint(XI, T, PTP, FL):
    """6-output joint GP: [T, ptp, Fd1, Fq1, Fd3, Fq3]."""
    X = torch.tensor(np.asarray(XI))
    cols = [np.nan_to_num(T, nan=0.0), np.nan_to_num(PTP, nan=1.0e3)] + \
           [np.nan_to_num(FL[:, j], nan=0.0) for j in range(4)]
    models = [dsp_gp(X, torch.tensor(c).unsqueeze(-1)) for c in cols]
    m = ModelListGP(*models)
    fit_gpytorch_mll(SumMarginalLogLikelihood(m.likelihood, m))
    return m


def ptp_of(R, T):
    return np.asarray(R, float) * np.asarray(T, float) / 100.0


def voltage_bound(fd1, fq1, fd3, fq3, dq, omega, R, Lew):
    """EXACT peak phase voltage: max over theta of the combined 1st+3rd waveform (not the
    |V1|+|V3| triangle bound, which over-rejects near-limit designs). dq voltage of harmonic h:
    V_d = R*Id - h*w*(psi_q + Lew*Iq); V_q = R*Iq + h*w*(psi_d + Lew*Id). fd*/fq* and dq[k]
    broadcast together; the running max over _VGRID avoids a [n_theta,G,NI] tensor."""
    Id1, Iq1, Id3, Iq3 = dq
    Vd1 = R * Id1 - omega * (fq1 + Lew * Iq1); Vq1 = R * Iq1 + omega * (fd1 + Lew * Id1)
    Vd3 = R * Id3 - 3.0 * omega * (fq3 + Lew * Iq3); Vq3 = R * Iq3 + 3.0 * omega * (fd3 + Lew * Id3)
    Vm1, p1 = np.hypot(Vd1, Vq1), np.arctan2(Vq1, Vd1)
    Vm3, p3 = np.hypot(Vd3, Vq3), np.arctan2(Vq3, Vd3)
    mx = np.zeros_like(np.asarray(Vm1, dtype=float))
    for th in _VGRID:
        np.maximum(mx, np.abs(Vm1 * np.cos(th + p1) + Vm3 * np.cos(3.0 * th + p3)), out=mx)
    return mx


def paretoA_objectives(surf, ipk_cand, loss_cand, dq_cand, demands, omegas, i_max, lam,
                       R, Lew, v_max):
    """surf = (Ts, PTPs, Fd1, Fq1, Fd3, Fq3), each [G, NI]. Returns (cycle_loss[G],
    max_ripple[G], feas[G]). Per demand k: min penalised loss over currents s.t. T>=Tk,
    Ipk<=Imax, V_pk(omega_k)<=Vmax; ripple read off there."""
    Ts, PTPs, Fd1, Fq1, Fd3, Fq3 = surf
    pen_ipk = lam * np.clip(ipk_cand - i_max, 0.0, None)
    G = Ts.shape[0]; K = len(demands); ar = np.arange(G)
    Fk = np.empty((K, G)); Rk = np.empty((K, G)); feask = np.ones((K, G), bool)
    for k, (Tk, wk) in enumerate(zip(demands, omegas)):
        dqc = [dq_cand[:, j][None, :] for j in range(4)]                  # each [1, NI] -> [G,NI]
        Vpk = voltage_bound(Fd1, Fq1, Fd3, Fq3, dqc, wk, R, Lew)          # [G, NI]
        pen = (loss_cand[None, :] + pen_ipk[None, :] + lam * np.clip(Tk - Ts, 0.0, None)
               + lam * np.clip(Vpk - v_max, 0.0, None))
        j = np.argmin(pen, axis=1)
        Fk[k] = pen[ar, j]; Rk[k] = PTPs[ar, j]
        feask[k] = (Ts[ar, j] >= Tk) & (ipk_cand[j] <= i_max) & (Vpk[ar, j] <= v_max)
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


def confirmed_front(Xg, Xi, T, R, FL, dq_of, demands, omegas, i_max, Rs, Lew, v_max):
    """Non-dominated (cycle_loss, max_ptp[Nm], ripple%) over FEA-confirmed geometries, with
    the voltage bound enforced per operating point using the ACTUAL FEA flux."""
    Xg = np.array(Xg); FL = np.array(FL); key = {}
    for i in range(len(T)):
        key.setdefault(tuple(np.round(Xg[i], 6)), []).append(i)
    obj, pct = [], []
    for _g, ids in key.items():
        dq = np.array([dq_of(Xi[i]) for i in ids])
        tt = np.array([T[i] for i in ids]); rr = np.array([R[i] for i in ids])
        fl = FL[ids]; ptp = ptp_of(rr, tt); ll = np.sum(dq ** 2, axis=1)
        ip = np.array([peak_current(dq[j]) for j in range(len(dq))])
        cl, mp, mpct, ok = 0.0, 0.0, 0.0, True
        for Tk, wk in zip(demands, omegas):
            vpk = voltage_bound(fl[:, 0], fl[:, 1], fl[:, 2], fl[:, 3],
                                [dq[:, 0], dq[:, 1], dq[:, 2], dq[:, 3]], wk, Rs, Lew)
            fe = (ip <= i_max) & (tt >= Tk) & (vpk <= v_max)
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


def peak_current(dq):
    Id1, Iq1, Id3, Iq3 = dq
    Im1, a1 = math.hypot(Id1, Iq1), math.atan2(Iq1, Id1)
    Im3, a3 = math.hypot(Id3, Iq3), math.atan2(Iq3, Id3)
    return float(np.max(np.abs(Im1 * np.cos(_THETA + a1) + Im3 * np.cos(3.0 * _THETA + a3))))


def main():
    job, outp = sys.argv[1], sys.argv[2]
    z = np.load(job, allow_pickle=True)
    Xg, Xi, T, R, FL = z["Xg"], z["Xi"], z["T"], z["R"], z["FL"]
    Gcand, Icand_u, Icand_dq = z["Gcand"], z["Icand_u"], z["Icand_dq"]
    ipk_cand, loss_cand = z["ipk_cand"], z["loss_cand"]
    icur_lb, icur_ub = z["icur_lb"], z["icur_ub"]
    demands = list(z["demands"]); omegas = list(z["omegas"])
    i_max = float(z["i_max"]); lam = float(z["lam"])
    Rs = float(z["r_stator"]); Lew = float(z["lew"]); v_max = float(z["v_max"])
    n_paths, q, seed = int(z["n_paths"]), int(z["q"]), int(z["seed"])
    global _THETA
    _THETA = z["theta"]
    NG, NI = len(Gcand), len(Icand_u)
    np.random.seed(seed); torch.manual_seed(seed)

    def dq_of(u):
        return icur_lb + np.asarray(u) * (icur_ub - icur_lb)

    if n_paths == 0:                     # front-only mode (final report; no GP/Matheron)
        front = confirmed_front(Xg, Xi, T, R, FL, dq_of, demands, omegas, i_max, Rs, Lew, v_max)
        json.dump(dict(picks=[], varlog={}, front=front), open(outp, "w"))
        return

    m = fit_joint(np.hstack([Xg, Xi]), T, ptp_of(R, T), np.asarray(FL))
    GX = np.repeat(Gcand, NI, axis=0); IX = np.tile(Icand_u, (NG, 1))
    Xq = torch.tensor(np.hstack([GX, IX]))

    def eval_paths_chunked(path, block=512):
        out = np.empty((n_paths, Xq.shape[0]), dtype=np.float64)
        for s in range(0, Xq.shape[0], block):
            out[:, s:s + block] = path(Xq[s:s + block]).numpy()
        return out.reshape(n_paths, NG, NI)

    with torch.no_grad():
        paths = [draw_matheron_paths(m.models[i], sample_shape=torch.Size([n_paths])) for i in range(6)]
        S = [eval_paths_chunked(p) for p in paths]           # [T, ptp, Fd1,Fq1,Fd3,Fq3] each [n,NG,NI]
    S[1] = np.clip(S[1], 0.0, None)                            # ptp >= 0

    cand_gi, cand_obj, path_hv, path_argmin = [], [], [], []
    for s in range(n_paths):
        surf = (S[0][s], S[1][s], S[2][s], S[3][s], S[4][s], S[5][s])
        cl, mr, _ = paretoA_objectives(surf, ipk_cand, loss_cand, Icand_dq, demands, omegas,
                                       i_max, lam, Rs, Lew, v_max)
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
    # per picked geometry: per-demand loss-optimal current under the posterior MEAN (T + flux),
    # honouring the same T/Ipk/voltage feasibility used in selection. Posterior mean evaluated
    # ONLY at the picked geometries' rows (never the full grid) to keep memory bounded.
    pen_ipk = lam * np.clip(ipk_cand - i_max, 0.0, None); dqc = [Icand_dq[:, j] for j in range(4)]
    picks = []
    if picked_gi:
        rows = np.concatenate([np.arange(gi * NI, (gi + 1) * NI) for gi in picked_gi])
        with torch.no_grad():
            muA = m.posterior(Xq[rows]).mean.numpy()
        muT = muA[:, 0].reshape(len(picked_gi), NI)
        muF = [muA[:, 2 + j].reshape(len(picked_gi), NI) for j in range(4)]
        for pi, gi in enumerate(picked_gi):
            idxs = []
            for Tk, wk in zip(demands, omegas):
                vpk = voltage_bound(muF[0][pi], muF[1][pi], muF[2][pi], muF[3][pi], dqc, wk, Rs, Lew)
                pen = (loss_cand + pen_ipk + lam * np.clip(Tk - muT[pi], 0.0, None)
                       + lam * np.clip(vpk - v_max, 0.0, None))
                idxs.append(int(np.argmin(pen)))
            picks.append([gi, idxs])

    front = confirmed_front(Xg, Xi, T, R, FL, dq_of, demands, omegas, i_max, Rs, Lew, v_max)
    json.dump(dict(picks=picks, varlog=varlog, front=front), open(outp, "w"))


if __name__ == "__main__":
    main()
