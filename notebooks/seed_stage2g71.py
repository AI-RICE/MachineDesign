"""Seed a current-only BO on the FIXED 71.2 Hz dq1-geometry G* to get the honest
max-FEASIBLE torque there (the true T_seq). Warm-starts with two known points on
G*: the stage-1 dq1-only optimum and c_joint@G* (from attribution, the high-torque
but likely ripple-infeasible joint setpoint). h0h1_par stage2 then tops up the
init and runs the constrained-EI search. Writes results_volt71_g2/.
"""
import json
import os

import h0h1_par as P
import numpy as np

s1 = json.load(open("results_volt71/stage1_best.json"))
at = json.load(open("attrib71.json"))
cj = np.array(at["c_joint"], float)
gstar = at["G_star"]

DQ = np.vstack([np.array(s1["dq"], float), cj])
T = np.array([float(s1["T"]), float(gstar["T"])])
R = np.array([float(s1["ripple"]), float(gstar["rip"])])
V = np.array([float(s1["Vpk"]), float(gstar["vpk"])])
Ipk = np.array([float(s1["Ipk"]), float(gstar["ipk"])])

U = np.clip(np.array([P.cur_to_norm(dq) for dq in DQ]), 0.0, 1.0)
os.makedirs("results_volt71_g2", exist_ok=True)
np.savez("results_volt71_g2/stage2.npz", U=U, T=T, R=R, V=V, Ipk=Ipk)
json.dump(s1, open("results_volt71_g2/stage1_best.json", "w"))   # provides G* geom_norm
feas = (T > 0) & (R <= 5) & (V <= 800) & (Ipk <= 10.001)
print(f"seeded {len(U)} pts on 71.2Hz G* ({int(feas.sum())} feasible); "
      f"best feasible T={float(np.max(T[feas])) if feas.any() else 0:.2f}  (T_joint={at['T_joint']:.2f})")
