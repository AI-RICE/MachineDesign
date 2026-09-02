"""Seed the turns-free run from the union of the 400V + 500V pools, and write a front-only
turns-free test job to sanity-check gen3_select before spending FEA. Run on bayes."""
import math
import os

import gen2  # ipk_of
import h0h1_study as H  # _THETA
import numpy as np

POOLS = ["results/gen3_500w_v", "results/gen3_500w_v500"]
OUT = "results/gen3_500w_tf"
LB = np.array([0.0, 0.0, -0.39, -0.39]); UB = np.array([1.3, 1.3, 0.39, 0.39])

Xg, Xi, T, R, FL = [], [], [], [], []
for d in POOLS:
    z = np.load(f"{d}/gen3.npz", allow_pickle=True)
    Xg += list(z["Xg"]); Xi += list(z["Xi"]); T += list(z["T"]); R += list(z["R"]); FL += list(z["FL"])
Xg, Xi, T, R, FL = map(np.array, (Xg, Xi, T, R, FL))
os.makedirs(OUT, exist_ok=True)
np.savez(f"{OUT}/gen3.npz", Xg=Xg, Xi=Xi, T=T, R=R, FL=FL)
print(f"[prep_tf] seeded {OUT}/gen3.npz with {len(T)} points, "
      f"{len({tuple(np.round(g,6)) for g in Xg})} geometries")

# front-only turns-free test job (n_paths=0 -> confirmed_front path only)
Icu = np.random.RandomState(0).rand(8, 4)          # dummy candidate pool (unused when n_paths=0)
Idq = LB + Icu * (UB - LB)
ipk = np.array([gen2.ipk_of(d) for d in Idq]); loss = np.sum(Idq ** 2, 1)
np.savez(f"{OUT}/seljob_tftest.npz", Xg=Xg, Xi=Xi, T=T, R=R, FL=FL,
         Gcand=Xg[:8], Icand_u=Icu, Icand_dq=Idq, ipk_cand=ipk, loss_cand=loss,
         icur_lb=LB, icur_ub=UB, demands=np.array([4.0, 8.0, 2.5]),
         omegas=np.array([2 * math.pi * f for f in (25.0, 16.0, 63.0)]),
         theta=np.asarray(H._THETA, float), i_max=1.3, lam=20.0, r_stator=19.0, lew=2.4e-3,
         v_max=400.0, turns_free=1, nc_base=113.0, n_paths=0, q=8, seed=0)
print(f"[prep_tf] wrote {OUT}/seljob_tftest.npz (front-only, turns_free=1)")
