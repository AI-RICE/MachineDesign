"""Offline acquisition diagnostics for gen2 (no FEA). Loads a run's gen2.npz + the
fitted joint GP and reports the surrogate-validation health check
(docs/surrogate-validation.md): (1) CV calibration of the modelled outputs, with an
old-ripple%-vs-new-ptp contrast; (2) inner current-grid convergence across sizes;
(3) predicted-vs-realized F. Usage: python notebooks/diag_gen2.py [run_dir]
"""
import sys
import numpy as np
import torch

import gen2

OUT = sys.argv[1] if len(sys.argv) > 1 else "results/gen2_v2"
z = np.load(f"{OUT}/gen2.npz", allow_pickle=True)
Xg = np.array(z["Xg"]); Xi = np.array(z["Xi"]); T = np.array(z["T"]); R = np.array(z["R"])
X = np.hstack([Xg, Xi]); n = len(T)
PTP = gen2.ptp_of(R, T)
demands = [20.0, 35.0, 30.0]
print(f"[diag] {OUT}: {n} pooled points, dim {X.shape[1]}", flush=True)


def dq_of(u):
    return gen2.ICUR_LB + u * (gen2.ICUR_UB - gen2.ICUR_LB)


def cv(Y, tag):
    Kf = 8; idx = np.arange(n); np.random.RandomState(0).shuffle(idx)
    res, zz = [], []
    for f in np.array_split(idx, Kf):
        tr = np.setdiff1d(idx, f)
        g = gen2.dsp_gp(torch.tensor(X[tr]), torch.tensor(Y[tr]).unsqueeze(-1))
        from botorch.fit import fit_gpytorch_mll
        from gpytorch.mlls import ExactMarginalLogLikelihood
        fit_gpytorch_mll(ExactMarginalLogLikelihood(g.likelihood, g))
        with torch.no_grad():
            p = g.posterior(torch.tensor(X[f])); mu = p.mean.numpy().ravel(); sd = np.sqrt(p.variance.numpy().ravel())
        res += list(mu - Y[f]); zz += list((mu - Y[f]) / sd)
    res, zz = np.array(res), np.array(zz)
    print("  %-14s RMSE %8.2f  signal-sd %8.2f  ratio %.2f  cover@2sig %.2f"
          % (tag, np.sqrt(np.mean(res**2)), Y.std(), np.sqrt(np.mean(res**2)) / Y.std(), np.mean(np.abs(zz) < 2)))


print("\n=== (1) CV calibration (RMSE/signal-sd < 1 and cover ~0.95 = healthy) ===")
cv(T, "T [Nm]")
cv(R, "ripple%% (OLD)")   # the broken target
cv(PTP, "ptp [Nm] (NEW)")  # the fix

# ---- (2) inner current-grid convergence ----
m = gen2.fit_joint(X, T, PTP)
gsamp = Xg[np.random.RandomState(0).choice(len(Xg), 8, replace=False)]
print("\n=== (2) inner-grid convergence (mean penalized F over 8 geoms; should flatten) ===")
prev = None
for nc in [64, 256, 512, 1024, 4096]:
    Iu = torch.quasirandom.SobolEngine(4, scramble=True, seed=1).draw(nc).numpy()
    dqc = np.array([dq_of(u) for u in Iu]); ipk = np.array([gen2.ipk_of(d) for d in dqc]); loss = np.sum(dqc**2, axis=1)
    Fs = []
    for g in gsamp:
        Xq = torch.tensor(np.hstack([np.tile(g, (nc, 1)), Iu]))
        with torch.no_grad():
            mu = m.posterior(Xq).mean.numpy()
        Fs.append(gen2.inner_from_surface(mu[:, 0], np.clip(mu[:, 1], 0, None), Iu, ipk, loss, demands)[0])
    mF = float(np.mean(Fs))
    d = "" if prev is None else "  (Δ vs prev %.2f)" % (prev - mF)
    print("  n_icand %5d: mean F %.2f%s" % (nc, mF, d)); prev = mF

# ---- (3) predicted vs realized F ----
I0 = torch.quasirandom.SobolEngine(4, scramble=True, seed=7).draw(512).numpy()
dq0 = np.array([dq_of(u) for u in I0]); ipk0 = np.array([gen2.ipk_of(d) for d in dq0]); loss0 = np.sum(dq0**2, axis=1)
key = {}
for i in range(n):
    key.setdefault(tuple(np.round(Xg[i], 6)), []).append(i)
preds, real = [], []
for g, ids in key.items():
    gg = np.array(g)
    dq = np.array([dq_of(Xi[i]) for i in ids]); tt = T[list(ids)]; rr = R[list(ids)]
    ll = np.sum(dq**2, axis=1); ip = np.array([gen2.ipk_of(dq[j]) for j in range(len(dq))])
    base = (rr <= gen2.R_MAX) & (ip <= gen2.I_MAX)
    Fk = [float(np.min(ll[base & (tt >= Tk)])) if (base & (tt >= Tk)).any() else gen2.BIG_LOSS for Tk in demands]
    real.append(float(np.mean(Fk)))
    Xq = torch.tensor(np.hstack([np.tile(gg, (512, 1)), I0]))
    with torch.no_grad():
        mu = m.posterior(Xq).mean.numpy()
    preds.append(gen2.inner_from_surface(mu[:, 0], np.clip(mu[:, 1], 0, None), I0, ipk0, loss0, demands)[0])
preds, real = np.array(preds), np.array(real); fin = real < gen2.BIG_LOSS
print("\n=== (3) predicted vs realized F (feasible geoms; corr high = outer trustworthy) ===")
if fin.sum() > 2:
    print("  n feasible %d  corr(pred,real) %.3f" % (int(fin.sum()), np.corrcoef(preds[fin], real[fin])[0, 1]))
print("\n[diag] DONE", flush=True)
