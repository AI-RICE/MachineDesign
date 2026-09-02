"""Convert the 63Hz OAT grid into a warm-start checkpoint for a local joint BO.

Maps each grid point (geom_norm, dq) into the joint-stage trust-region u-space
(center = voltage P0, tr=0.15) and writes results_volt_local/joint.npz so
`h0h1_par.py --stage joint` resumes from it and runs more cEI iterations -- now
seeded with a real FEASIBLE neighborhood instead of cold-starting at the TR edges.
Drops geometry-degenerate points (barrier intersection -> sentinel ripple) so the
constraint GPs train on real measurements.
"""
import json
import os

import h0h1_par as P
import numpy as np

TR = 0.15

g = np.load("results_grid/grid_results.npz", allow_pickle=True)
G, DQ, T, R, V, I = g["G"], g["DQ"], g["T"], g["R"], g["V"], g["I"]
s2 = json.load(open("results_volt/stage2_best.json"))
gc = np.array(s2["geom_norm"])
cc = np.clip(P.cur_to_norm(np.array(s2["dq"])), 0.0, 1.0)
span = P.ICUR_UB - P.ICUR_LB

keep = R < 900   # drop geometry-infeasible (barrier intersection) points
U = []
for k in np.where(keep)[0]:
    curnorm = (DQ[k] - P.ICUR_LB) / span
    ug = 0.5 + (G[k] - gc) / (2 * TR)
    uc = 0.5 + (curnorm - cc) / (2 * TR)
    U.append(np.clip(np.concatenate([ug, uc]), 0.0, 1.0))
U = np.array(U)
T2, R2, V2, I2 = T[keep], R[keep], V[keep], I[keep]

os.makedirs("results_volt_local", exist_ok=True)
np.savez("results_volt_local/joint.npz", U=U, T=T2, R=R2, V=V2, Ipk=I2)
json.dump(s2, open("results_volt_local/stage2_best.json", "w"))
feas = (T2 > 0) & (R2 <= 5) & (V2 <= 800) & (I2 <= 10.001)
print(f"seeded {len(U)} grid points ({int(feas.sum())} feasible) into results_volt_local/joint.npz "
      f"(dropped {int((~keep).sum())} geom-degenerate); best feasible T={float(np.max(T2[feas])) if feas.any() else float('nan'):.3f}")
