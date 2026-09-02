"""Seed a current-only BO on the FIXED dq1-geometry G* (63.1 Hz) to find the TRUE
max-feasible torque there -> the honest T_seq. If it stays << 42, the +11% needs
the moved geometry => coupling (H1) confirmed; if it reaches ~42, separable.

Warm-starts with current points already evaluated on G*: the grid's current-
direction setpoints (c0..c3 +/-) + P0, plus c_joint@G* from the attribution test
(T=41.68 but ripple 6.39 -> infeasible). Writes results_volt_g2/stage2.npz.
"""
import json
import os

import h0h1_par as P
import numpy as np

g = np.load("results_grid/grid_results.npz", allow_pickle=True)
labels, G, DQ, T, R, V, I = g["labels"], g["G"], g["DQ"], g["T"], g["R"], g["V"], g["I"]
# current-direction points + P0 are all on G* (geometry unchanged)
mask = np.array([str(l) == "P0" or str(l).startswith("c") for l in labels])
DQs, Ts, Rs, Vs, Is = DQ[mask], T[mask], R[mask], V[mask], I[mask]
# add c_joint evaluated at G* (from attribution): infeasible (ripple 6.39)
cj = np.array(json.load(open("results_volt_local/joint_best.json"))["dq"])
DQs = np.vstack([DQs, cj])
Ts = np.append(Ts, 41.684); Rs = np.append(Rs, 6.39); Vs = np.append(Vs, 790.0); Is = np.append(Is, 9.01)

U = np.clip(np.array([P.cur_to_norm(dq) for dq in DQs]), 0.0, 1.0)
os.makedirs("results_volt_g2", exist_ok=True)
np.savez("results_volt_g2/stage2.npz", U=U, T=Ts, R=Rs, V=Vs, Ipk=Is)
s1 = json.load(open("results_volt/stage1_best.json"))
json.dump(s1, open("results_volt_g2/stage1_best.json", "w"))
feas = (Ts > 0) & (Rs <= 5) & (Vs <= 800) & (Is <= 10.001)
print(f"seeded {len(U)} current pts on G* ({int(feas.sum())} feasible); "
      f"best feasible T={float(np.max(Ts[feas])) if feas.any() else 0:.2f}  (T_joint=42.4)")
