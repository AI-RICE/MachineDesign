"""Attribution test for a local-joint gain (voltage regime). Two deterministic FEA:
  (1) (G_joint, c_joint)  -> reconfirm T_joint (rules out a recording/extraction bug).
  (2) (G*,     c_joint)  -> the DECIDER: the joint's optimal currents at the UN-moved
      dq1-geometry. If ~T_joint and FEASIBLE -> geometry move unnecessary (SEPARABLE).
      If << T_joint or INFEASIBLE -> the moved geometry is needed (geometry x dq3
      COUPLING, H1).

Parametrized so it serves any study (default: the 71.2 Hz / R=0.19 run). Saves the
result to --out json so the max-feasible-on-G* re-converge can seed c_joint@G*.
"""
import argparse
import json
import os

import numpy as np

import h0h1_par as P
from machine_design import analyze_results
from machine_design.design2 import Design2

ap = argparse.ArgumentParser()
ap.add_argument("--stage1", default="results_volt71/stage1_best.json")
ap.add_argument("--joint", default="results_volt71/joint_best.json")
ap.add_argument("--fhz", type=float, default=71.2)
ap.add_argument("--tag", default="attrib71")
ap.add_argument("--out", default="attrib71.json")
args = ap.parse_args()

s1 = json.load(open(args.stage1))
sj = json.load(open(args.joint))
Gstar = np.array(s1["geom_norm"])
Gjoint = np.array(sj["geom_joint"])
cj = [float(x) for x in sj["dq"]]
T_seq = float(s1.get("T", 0.0))
T_joint = float(sj.get("T_joint", 0.0))

path = os.path.join(os.getcwd(), "data", f"{args.tag}.aedt")
mk = dict(version="2024.2", non_graphical=True, new_desktop=True, close_on_exit=True)
design = Design2.load(path, **mk) if os.path.exists(path) else Design2.create(args.tag, "Design01", path, **mk)
P.set_speed(design, args.fhz)
gen = P.make_generator(design, wide=True)
lb, ub = P.H.geom_bounds_arrays(gen)


def fea(geom, dq):
    barriers = P.H.build_barriers(gen, geom, lb, ub)
    assert barriers is not None, "geometry infeasible"
    design.add_rotor()
    for b in barriers:
        design.add_rotor_barrier(b)
    res = design.compute(*[float(x) for x in dq], NUM_CORES=4)
    m = res["means"]
    T, _, rip = analyze_results(np.asarray(res["Tor"], float))
    vpk = P.combined_voltage_peak(m["V_d1"], m["V_q1"], m["V_d3"], m["V_q3"])
    design.delete_rotor()
    return float(T), float(rip), float(vpk), float(P.peak_current_from_dq(*dq))


print(f"f={args.fhz} Hz, c_joint = {cj}", flush=True)
T1, r1, v1, i1 = fea(Gjoint, cj)
print(f"(1) (G_joint, c_joint) reconfirm:  T={T1:.3f}  rip={r1:.2f}  Vpk={v1:.0f}  Ipk={i1:.2f}   (expect ~{T_joint:.2f})", flush=True)
T2, r2, v2, i2 = fea(Gstar, cj)
print(f"(2) (G*,      c_joint) ATTRIBUTION: T={T2:.3f}  rip={r2:.2f}  Vpk={v2:.0f}  Ipk={i2:.2f}", flush=True)
feas2 = (r2 <= 5.0) and (v2 <= 800.0) and (i2 <= 10.001)
print(f"--- T_seq(P0)={T_seq:.2f}, T_joint={T_joint:.2f} ; T(G*,c_joint)={T2:.2f} feasible={feas2} (rip{r2:.2f} V{v2:.0f} I{i2:.2f}) ---", flush=True)
if feas2 and T2 > 0.97 * T_joint:
    verdict = "c_joint feasible at G* and ~T_joint -> geometry move unnecessary -> SEPARABLE (H0)"
else:
    why = []
    if r2 > 5.0: why.append(f"ripple {r2:.2f}>5")
    if v2 > 800.0: why.append(f"V {v2:.0f}>800")
    if i2 > 10.001: why.append(f"Ipk {i2:.2f}>10")
    if T2 <= 0.97 * T_joint and feas2: why.append(f"T {T2:.1f}<<{T_joint:.1f}")
    verdict = f"c_joint NOT a feasible ~T_joint point at G* ({'; '.join(why)}) -> moved geometry needed -> COUPLING (H1)"
print(f"VERDICT: {verdict}", flush=True)

json.dump({"fhz": args.fhz, "c_joint": cj, "T_seq": T_seq, "T_joint": T_joint,
           "G_joint": {"T": T1, "rip": r1, "vpk": v1, "ipk": i1},
           "G_star": {"T": T2, "rip": r2, "vpk": v2, "ipk": i2, "feasible": feas2}},
          open(args.out, "w"), indent=2)
print(f"saved {args.out}", flush=True)
design.save_project()
design.close_project()
