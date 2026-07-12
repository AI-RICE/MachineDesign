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
_NCGRID = np.arange(55.0, 161.0, 5.0)                          # candidate winding counts (turns-free)


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
                       R, Lew, v_max, turns_free=False, nc_base=113.0):
    """surf = (Ts, PTPs, Fd1, Fq1, Fd3, Fq3), each [G, NI]. Returns (cycle_loss[G],
    max_ripple[G], feas[G]). Per demand k pick min-loss current with T>=Tk; ripple read there.
    Feasibility:
      fixed turns  -> also require Ipk<=Imax and V_pk(omega_k)<=Vmax at every point.
      turns free   -> Ipk/Vpk are at base turns Nc0; a single winding Nc must satisfy ALL
                      points: Nc in [Nc0*max_k(Ipk_k/Imax), Nc0*min_k(Vmax/Vpk_k)] must be
                      non-empty (V~Nc, I~1/Nc; torque/ripple/loss turns-invariant)."""
    Ts, PTPs, Fd1, Fq1, Fd3, Fq3 = surf
    G = Ts.shape[0]; K = len(demands); ar = np.arange(G)
    Vb = [voltage_bound(Fd1, Fq1, Fd3, Fq3, [dq_cand[:, c][None, :] for c in range(4)], wk, R, Lew)
          for wk in omegas]                                              # base-turns Vpk per demand [G,NI]
    if not turns_free:
        pen_ipk = lam * np.clip(ipk_cand - i_max, 0.0, None)
        Fk = np.empty((K, G)); Rk = np.empty((K, G)); feask = np.ones((K, G), bool)
        for k, Tk in enumerate(demands):
            pen = (loss_cand[None, :] + pen_ipk[None, :] + lam * np.clip(Tk - Ts, 0.0, None)
                   + lam * np.clip(Vb[k] - v_max, 0.0, None))
            j = np.argmin(pen, axis=1)
            Fk[k] = pen[ar, j]; Rk[k] = PTPs[ar, j]
            feask[k] = (Ts[ar, j] >= Tk) & (ipk_cand[j] <= i_max) & (Vb[k][ar, j] <= v_max)
        return Fk.mean(0), Rk.max(0), feask.all(0)
    # turns free: best over the winding-count grid. At Nc the base peak current/voltage rescale
    # by (nc_base/Nc) and (Nc/nc_base); the per-Nc inner properly field-weakens each point.
    best_cl = np.full(G, np.inf); best_mr = np.zeros(G); feas_any = np.zeros(G, bool)
    for Nc in _NCGRID:
        s = nc_base / Nc
        clk = np.zeros(G); mrk = np.zeros(G); feas_all = np.ones(G, bool)
        for k, Tk in enumerate(demands):
            ip_nc = ipk_cand * s; V_nc = Vb[k] / s
            pen = (loss_cand[None, :] + lam * np.clip(Tk - Ts, 0.0, None)
                   + lam * np.clip(ip_nc[None, :] - i_max, 0.0, None) + lam * np.clip(V_nc - v_max, 0.0, None))
            j = np.argmin(pen, axis=1)
            clk += pen[ar, j] / K; mrk = np.maximum(mrk, PTPs[ar, j])
            feas_all &= (Ts[ar, j] >= Tk) & (ip_nc[j] <= i_max) & (V_nc[ar, j] <= v_max)
        better = clk < best_cl
        best_mr = np.where(better, mrk, best_mr); best_cl = np.where(better, clk, best_cl)
        feas_any |= feas_all
    return best_cl, best_mr, feas_any


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


def _front_ab(front_lr):
    """(loss,ripple) list -> non-dominated front as (a ascending loss, b descending ripple)."""
    if len(front_lr) == 0:
        return np.empty(0), np.empty(0)
    O = np.asarray(front_lr, float)
    nd = is_non_dominated(torch.tensor(-O, dtype=torch.double)).numpy()
    O = O[nd]; O = O[np.argsort(O[:, 0])]
    return O[:, 0].copy(), O[:, 1].copy()


def hvi_2d(ql, qr, a, b, rL, rR):
    """Vectorized 2-D hypervolume improvement (MINIMIZATION) of query points (ql,qr) over an
    incumbent front (a ascending loss, b descending ripple), reference (rL,rR). Exact:
    HVI = area{ x in [ql,rL], y in [qr,rR] NOT already dominated by any incumbent }. This handles
    a query that itself DOMINATES incumbent points (it then subsumes, not adds, their area).
    Computed as the query box minus the incumbent-dominated staircase clipped into that box."""
    ql = np.asarray(ql, float); qr = np.asarray(qr, float)
    inside = (ql < rL) & (qr < rR)
    box = np.maximum(0.0, rL - ql) * np.maximum(0.0, rR - qr)
    if len(a) == 0:
        return np.where(inside, box, 0.0)
    cx = np.maximum(a[:, None], ql[None, :])                  # [n,G] incumbent loss clipped into box
    cy = np.maximum(b[:, None], qr[None, :])                  # [n,G] incumbent ripple clipped into box
    cx_next = np.empty_like(cx); cx_next[:-1] = cx[1:]; cx_next[-1] = rL
    width = np.clip(np.minimum(cx_next, rL) - cx, 0.0, None)  # a asc -> cx non-decreasing (valid strips)
    height = np.clip(rR - cy, 0.0, None)                      # b desc -> cy non-increasing
    covered = np.sum(width * height, axis=0)                  # area of box already dominated
    return np.where(inside, np.maximum(0.0, box - covered), 0.0)


def constrained_ehvi_batch(CL, MR, FE, inc_front, q):
    """Gardner-style constrained acquisition, Thompson/MC form. CL,MR,FE are [n_paths, G] arrays
    of the per-path derived cycle-loss, max-ripple and FEASIBILITY of each candidate geometry.
    alpha(g) = mean_s[ 1{FE_s(g)} * HVI(CL_s(g),MR_s(g) | incumbent feasible front) ]
             = E[ HVI * feasibility-indicator ]  = EHVI(g) * P(feasible)(g)  (Gardner 2014),
    estimated on coherent posterior sample PATHS rather than a plug-in mean. Greedy q-batch with
    an expected-feasible-objective fantasy so picks are diverse. The incumbent front is the
    FEA-confirmed feasible Pareto front, so the acquisition *enriches that front*."""
    n_paths, G = CL.shape
    p_feas = FE.mean(0)
    ls, rs = [], []
    if len(inc_front):
        I = np.asarray(inc_front, float); ls.append(I[:, 0]); rs.append(I[:, 1])
    fin = FE & np.isfinite(CL) & np.isfinite(MR)
    if fin.any():
        ls.append(CL[fin]); rs.append(MR[fin])
    rL = (np.concatenate(ls).max() if ls else 1.0) * 1.05
    rR = (np.concatenate(rs).max() if rs else 1.0) * 1.05
    front = [tuple(x) for x in (np.asarray(inc_front, float)[:, :2] if len(inc_front) else [])]
    picked, ehvi_hist, ehvi_full = [], [], None
    for it in range(q):
        a, b = _front_ab(front)
        acc = np.zeros(G)
        for s in range(n_paths):
            acc += np.where(FE[s], hvi_2d(CL[s], MR[s], a, b, rL, rR), 0.0)
        ehvi = acc / n_paths
        if it == 0:
            ehvi_full = ehvi.copy()
        if picked:
            ehvi[picked] = -1.0
        g = int(np.argmax(ehvi))
        if ehvi[g] <= 1e-12:
            break
        picked.append(g); ehvi_hist.append(float(ehvi[g]))
        fm = FE[:, g]
        if fm.any():                                          # fantasy: expected feasible objective
            front.append((float(CL[fm, g].mean()), float(MR[fm, g].mean())))
    return picked, dict(ref=[float(rL), float(rR)], ehvi=ehvi_hist,
                        p_feasible=[float(p_feas[g]) for g in picked],
                        ehvi_full=(ehvi_full if ehvi_full is not None else np.zeros(G)))


def confirmed_front(Xg, Xi, T, R, FL, dq_of, demands, omegas, i_max, Rs, Lew, v_max,
                    turns_free=False, nc_base=113.0):
    """Non-dominated (cycle_loss, max_ptp[Nm], ripple%[, Nc_lo, Nc_hi]) over FEA-confirmed
    geometries, voltage bound per point from the ACTUAL FEA flux. turns_free: feasibility is a
    non-empty common-winding window [Nc0*max(ipk/Imax), Nc0*min(Vmax/Vpk)] rather than fixed
    turns; the window is reported so each finalist's required turns count is known."""
    Xg = np.array(Xg); FL = np.array(FL); key = {}
    for i in range(len(T)):
        key.setdefault(tuple(np.round(Xg[i], 6)), []).append(i)
    K = len(demands); obj, pct, ncw = [], [], []
    for _g, ids in key.items():
        dq = np.array([dq_of(Xi[i]) for i in ids])
        tt = np.array([T[i] for i in ids]); rr = np.array([R[i] for i in ids])
        fl = FL[ids]; ptp = ptp_of(rr, tt); ll = np.sum(dq ** 2, axis=1)
        ip = np.array([peak_current(dq[j]) for j in range(len(dq))])
        vb = [voltage_bound(fl[:, 0], fl[:, 1], fl[:, 2], fl[:, 3],
                            [dq[:, 0], dq[:, 1], dq[:, 2], dq[:, 3]], wk, Rs, Lew) for wk in omegas]
        if not turns_free:
            cl, mp, mpct, ok = 0.0, 0.0, 0.0, True
            for k, Tk in enumerate(demands):
                fe = (ip <= i_max) & (tt >= Tk) & (vb[k] <= v_max)
                if not fe.any():
                    ok = False; break
                j = int(np.where(fe)[0][np.argmin(ll[fe])]); cl += ll[j] / K
                if ptp[j] > mp:
                    mp, mpct = ptp[j], rr[j]
            if ok:
                obj.append((cl, mp)); pct.append(mpct); ncw.append((nc_base, nc_base))
            continue
        # turns free: best feasible over the winding-count grid (field-weakens per Nc)
        best = None
        for Nc in _NCGRID:
            s = nc_base / Nc; cl, mp, mpct, ok = 0.0, 0.0, 0.0, True
            for k, Tk in enumerate(demands):
                fe = (ip * s <= i_max) & (tt >= Tk) & (vb[k] / s <= v_max)
                if not fe.any():
                    ok = False; break
                j = int(np.where(fe)[0][np.argmin(ll[fe])]); cl += ll[j] / K
                if ptp[j] > mp:
                    mp, mpct = ptp[j], rr[j]
            if ok and (best is None or cl < best[0]):
                best = (cl, mp, mpct, Nc)
        if best is not None:
            obj.append((best[0], best[1])); pct.append(best[2]); ncw.append((best[3], best[3]))
    if not obj:
        return []
    O = np.array(obj); pct = np.array(pct)
    nd = is_non_dominated(torch.tensor(-O, dtype=torch.double)).numpy()
    return [[float(O[i, 0]), float(O[i, 1]), float(pct[i]), round(ncw[i][0], 1), round(ncw[i][1], 1)]
            for i in np.where(nd)[0]]


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
    turns_free = bool(int(z["turns_free"])) if "turns_free" in z else False
    nc_base = float(z["nc_base"]) if "nc_base" in z else 113.0
    global _THETA
    _THETA = z["theta"]
    NG, NI = len(Gcand), len(Icand_u)
    np.random.seed(seed); torch.manual_seed(seed)

    def dq_of(u):
        return icur_lb + np.asarray(u) * (icur_ub - icur_lb)

    if n_paths == 0:                     # front-only mode (final report; no GP/Matheron)
        front = confirmed_front(Xg, Xi, T, R, FL, dq_of, demands, omegas, i_max, Rs, Lew, v_max,
                                turns_free=turns_free, nc_base=nc_base)
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

    # ---- Gardner-style constrained EHVI (P(feasible)-weighted), Thompson/MC form ----
    # Per path, derive (cycle-loss, max-ripple, FEASIBLE) for EVERY candidate geometry. The
    # feasibility mask (torque, Ipk, voltage; turns-free: a valid winding exists) now GATES the
    # acquisition rather than being discarded: alpha(g) = mean_s[1{feas}*HVI] = EHVI*P(feasible).
    CL = np.empty((n_paths, NG)); MR = np.empty((n_paths, NG)); FE = np.zeros((n_paths, NG), bool)
    for s in range(n_paths):
        surf = (S[0][s], S[1][s], S[2][s], S[3][s], S[4][s], S[5][s])
        cl, mr, fe = paretoA_objectives(surf, ipk_cand, loss_cand, Icand_dq, demands, omegas,
                                        i_max, lam, Rs, Lew, v_max, turns_free=turns_free, nc_base=nc_base)
        CL[s] = cl; MR[s] = mr; FE[s] = fe
    inc = confirmed_front(Xg, Xi, T, R, FL, dq_of, demands, omegas, i_max, Rs, Lew, v_max,
                          turns_free=turns_free, nc_base=nc_base)
    inc_front = [(row[0], row[1]) for row in inc]
    picked_gi, acq = constrained_ehvi_batch(CL, MR, FE, inc_front, q)
    varlog = dict(n_paths=n_paths, method="constrained_ehvi_gardner",
                  inc_front_size=len(inc_front), mean_feasible_frac=float(FE.mean()),
                  n_positive_ehvi=int(np.sum(acq["ehvi_full"] > 1e-12)),
                  ehvi_picks=acq["ehvi"], p_feasible_picks=acq["p_feasible"], ref=acq["ref"])
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
                if turns_free:                        # turns absorbs I/V; choose min-loss meeting T
                    pen = loss_cand + lam * np.clip(Tk - muT[pi], 0.0, None)
                else:
                    pen = (loss_cand + pen_ipk + lam * np.clip(Tk - muT[pi], 0.0, None)
                           + lam * np.clip(vpk - v_max, 0.0, None))
                idxs.append(int(np.argmin(pen)))
            picks.append([gi, idxs])

    front = confirmed_front(Xg, Xi, T, R, FL, dq_of, demands, omegas, i_max, Rs, Lew, v_max,
                            turns_free=turns_free, nc_base=nc_base)
    json.dump(dict(picks=picks, varlog=varlog, front=front), open(outp, "w"))


if __name__ == "__main__":
    main()
