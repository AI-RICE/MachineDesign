"""RadialSpline latent-space viability gate (§11.2 / PARAMETERISATION.md §13 test 6).

The real go/no-go for the new parameterisation under latent-space BO:

  1. re-encode existing FEA designs (Hackl geometry) into RadialSpline X (114-D),
  2. train an MLP-VAE on a BROAD RadialSpline prior  --  GEOMETRY ONLY (leak-free),
  3. encode the FEA designs to the VAE latent,
  4. fit GP+ARD on (latent, T_mean) with a train/test split; compare to baselines.

HYGIENE (PARAMETERISATION.md §13): FEA torques are used only to *measure* this
gate. The VAE is selected on reconstruction (geometry) only — torques never tune it.

Baselines for context:
  * GP on native Hackl X (low-D)      -- information ceiling (faithful coords)
  * GP on raw RadialSpline X (114-D)  -- curse-of-dim floor
  * GBM on RadialSpline X (114-D)     -- nonparametric reference (§6.7)
  * GP on VAE latent                  -- the viability number

Run:
  .venv/bin/python notebooks/radialspline_latent_gate.py \
      --results-root ../MachineDesign/results --generator OneLambda
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from machine_design.fea_emulator import load_fea_designs  # noqa: E402
from machine_design.generators import (  # noqa: E402
    HacklGenerator_3BrokenLines,
    HacklGenerator_OneLambda,
    HacklGenerator_SixLambdas,
    RadialSplineGenerator,
)
from machine_design.lumped.geometry import REFERENCE_MACHINE  # noqa: E402

HACKL = {
    "OneLambda": HacklGenerator_OneLambda,
    "SixLambdas": HacklGenerator_SixLambdas,
    "ThreeBrokenLines": HacklGenerator_3BrokenLines,
}
HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------- re-encode
def reencode_fea(loaded, hackl, rs):
    """Hackl FEA X -> barriers -> RadialSpline X. Returns (X_rs, keep_mask)."""
    X_rs, keep = [], []
    for x in loaded.X:
        hackl.set_parameters(hackl.X_to_params(np.asarray(x, float)))
        bars = hackl.generate_barriers()
        ok = hackl.feasible_barriers(bars)
        keep.append(ok)
        X_rs.append(rs.fit_barriers(bars) if ok else np.zeros(rs.bounds[0].shape[0]))
    return np.asarray(X_rs), np.asarray(keep, bool)


# ---------------------------------------------------------------- VAE
class VAE(nn.Module):
    def __init__(self, D, latent, hidden=(256, 128)):
        super().__init__()
        h1, h2 = hidden
        self.enc = nn.Sequential(nn.Linear(D, h1), nn.ReLU(), nn.Linear(h1, h2), nn.ReLU())
        self.mu = nn.Linear(h2, latent)
        self.lv = nn.Linear(h2, latent)
        self.dec = nn.Sequential(
            nn.Linear(latent, h2), nn.ReLU(), nn.Linear(h2, h1), nn.ReLU(), nn.Linear(h1, D), nn.Sigmoid()
        )

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.lv(h)

    def forward(self, x):
        mu, lv = self.encode(x)
        z = mu + torch.randn_like(mu) * (0.5 * lv).exp()
        return self.dec(z), mu, lv


def train_vae(Xn, latent, epochs, beta, seed, batch=256, lr=1e-3, val_frac=0.15):
    torch.manual_seed(seed)
    D = Xn.shape[1]
    X = torch.from_numpy(Xn.astype(np.float32))
    n_val = int(len(X) * val_frac)
    perm = torch.randperm(len(X))
    val, tr = X[perm[:n_val]], X[perm[n_val:]]
    vae = VAE(D, latent)
    opt = torch.optim.Adam(vae.parameters(), lr=lr)
    for ep in range(epochs):
        vae.train()
        for i in range(0, len(tr), batch):
            xb = tr[i : i + batch]
            recon, mu, lv = vae(xb)
            rec = ((recon - xb) ** 2).sum(1).mean()
            kld = (-0.5 * (1 + lv - mu**2 - lv.exp()).sum(1)).mean()
            loss = rec + beta * kld
            opt.zero_grad()
            loss.backward()
            opt.step()
    vae.eval()
    with torch.no_grad():
        for name, S in (("train", tr), ("val", val)):
            recon, mu, lv = vae(S)
            rmse = ((recon - S) ** 2).mean().sqrt().item()
            kld_dim = (-0.5 * (1 + lv - mu**2 - lv.exp())).mean(0)  # per-dim KL
            if name == "val":
                val_rmse, val_kld = rmse, kld_dim.cpu().numpy()
            else:
                tr_rmse = rmse
    active = int((val_kld > 0.01).sum())
    return vae, tr_rmse, val_rmse, active, val_kld


# ---------------------------------------------------------------- GP / metrics
def fit_predict_gp(Xtr, ytr, Xte, bounds):
    bt = torch.from_numpy(bounds.astype(np.float64))
    ym, ys = ytr.mean(), ytr.std() + 1e-9
    Xt = torch.from_numpy(Xtr.astype(np.float64))
    yt = torch.from_numpy(((ytr - ym) / ys).astype(np.float64)).unsqueeze(-1)
    gp = SingleTaskGP(Xt, yt, input_transform=Normalize(d=Xt.shape[1], bounds=bt))
    fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
    gp.eval()
    with torch.no_grad():
        pred = gp.posterior(torch.from_numpy(Xte.astype(np.float64))).mean.squeeze(-1).cpu().numpy()
    return pred * ys + ym


def metrics(y_true, y_pred):
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    rho = float(spearmanr(y_true, y_pred).statistic)
    return rmse, rho


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="../MachineDesign/results")
    ap.add_argument("--generator", default="OneLambda", choices=list(HACKL))
    ap.add_argument("--latent", type=int, default=12)
    ap.add_argument("--n-prior", type=int, default=30000)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--n-test", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    rs = RadialSplineGenerator(REFERENCE_MACHINE)
    lo, hi = rs.bounds
    span = hi - lo

    # --- 1. load + re-encode FEA designs (cached) -----------------------
    cache = os.path.join(HERE, f"RadialSpline_reencoded_{args.generator}.npz")
    loaded = load_fea_designs(args.generator, results_root=args.results_root, constrained=None)
    if os.path.exists(cache):
        d = np.load(cache)
        X_rs, keep, T = d["X_rs"], d["keep"], d["T"]
    else:
        hackl = HACKL[args.generator](REFERENCE_MACHINE, r_stator_end=0.7, offset=0.35)
        X_rs, keep = reencode_fea(loaded, hackl, rs)
        T = loaded.T_mean
        np.savez(cache, X_rs=X_rs, keep=keep, T=T)
    X_hk = loaded.X[keep]
    X_rs = X_rs[keep]
    T = T[keep]
    print(f"[{args.generator}] FEA designs feasible/total: {keep.sum()}/{len(keep)}; D_hackl={X_hk.shape[1]}")

    # --- 2. broad RadialSpline prior, normalise to [0,1]^114 ------------
    Xp = np.stack([rs.random_X(rng) for _ in range(args.n_prior)])
    Xp_n = (Xp - lo) / span
    Xfea_n = np.clip((X_rs - lo) / span, 0, 1)

    # --- 3. train VAE (geometry only) ----------------------------------
    vae, tr_rmse, val_rmse, active, _ = train_vae(Xp_n, args.latent, args.epochs, args.beta, args.seed)
    print(f"VAE: recon RMSE train={tr_rmse:.4f} val={val_rmse:.4f} | active latent dims {active}/{args.latent}")

    with torch.no_grad():
        Z, _ = vae.encode(torch.from_numpy(Xfea_n.astype(np.float32)))
    Z = Z.cpu().numpy()
    z_bounds = np.stack([Z.min(0), Z.max(0)])

    # --- 4. GP/GBM viability with train/test split ---------------------
    idx = rng.permutation(len(T))
    te = idx[: args.n_test]
    pool = idx[args.n_test :]
    yte = T[te]
    print(f"\nheld-out test n={len(te)}, train pool n={len(pool)}")
    print(f"{'n_tr':>5} | {'GP latent':>16} | {'GP Hackl(ceil)':>16} | {'GP raw114':>16} | {'GBM raw114':>16}")
    print(f"{'':>5} | {'RMSE   rho':>16} | {'RMSE   rho':>16} | {'RMSE   rho':>16} | {'RMSE   rho':>16}")
    for n_tr in (32, 64, 128, 256):
        if n_tr > len(pool):
            break
        tr = pool[:n_tr]
        row = []
        # GP latent
        row.append(metrics(yte, fit_predict_gp(Z[tr], T[tr], Z[te], z_bounds)))
        # GP native Hackl (ceiling)
        hb_lo, hb_hi = HACKL[args.generator](REFERENCE_MACHINE, r_stator_end=0.7, offset=0.35).bounds
        row.append(metrics(yte, fit_predict_gp(X_hk[tr], T[tr], X_hk[te], np.stack([hb_lo, hb_hi]))))
        # GP raw 114-D
        row.append(metrics(yte, fit_predict_gp(X_rs[tr], T[tr], X_rs[te], np.stack([lo, hi]))))
        # GBM raw 114-D
        gbm = GradientBoostingRegressor(n_estimators=300, max_depth=4, random_state=args.seed).fit(X_rs[tr], T[tr])
        row.append(metrics(yte, gbm.predict(X_rs[te])))
        cells = " | ".join(f"{r:.3f} {rho:+.2f}".rjust(16) for (r, rho) in row)
        print(f"{n_tr:>5} | {cells}")


if __name__ == "__main__":
    main()
