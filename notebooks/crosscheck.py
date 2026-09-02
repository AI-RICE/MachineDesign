"""2x2 geometry x current crosscheck for the two 5-phase 60-slot designs.

Decouples geometry from current to attribute what dq3 actually does:
  rows  = rotor geometry   {G_dq1 (5f dq1-only opt), G_joint (5f dq1+dq3 opt)}
  cols  = current setpoint {I_dq1 (dq3=0),            I_joint (dq3 freed)}
Diagonal reproduces the two optima; off-diagonal runs each geometry with the
OTHER's current. If I_joint lowers ripple on BOTH geometries -> dq3 is a genuine
ripple lever; if it only helps on G_joint -> geometry-coupled.

wide=False (matches the runs). Uses the REAL Design2_60 (correct rotor radii), so
it ALSO dumps the correctly-decoded barriers -> xcheck.json for re-plotting geom3.
4 FEA solves (2 rotors x 2 currents). 50 Hz, T-target 20, r-max 5.
"""
import json
import os

import h0h1_par as P
import h0h1_study as H
import numpy as np

DQ1 = json.load(open("results/minloss60_5f_dq1/dq1_best.json"))
JNT = json.load(open("results/minloss60_5f_joint/joint_best.json"))

GEOMS = [("G_dq1", np.asarray(DQ1["geom_norm"], float)),
         ("G_joint", np.asarray(JNT["geom_norm"], float))]
CURS = [("I_dq1", [float(x) for x in DQ1["dq"]]),
        ("I_joint", [float(x) for x in JNT["dq"]])]

print("currents:")
for cn, dq in CURS:
    print(f"  {cn}: dq={['%.3f' % x for x in dq]}  |I|={np.sqrt(sum(x*x for x in dq)):.3f}  "
          f"Ipk={P.peak_current_from_dq(*dq):.3f}")

for ext in ("", ".lock"):
    p = f"data/xcheck_w0.aedt{ext}"
    if os.path.exists(p):
        os.remove(p)
design = P.open_isolated_design("xcheck", 0, "2024.2", slots=60, phases=5)
P.set_speed(design, 50.0)
gen = P.make_generator(design, False)          # wide=False, matches the runs
lb, ub = H.geom_bounds_arrays(gen)

rows = {}
dump = {"rotor_r_min": float(design.rotor_r_min), "rotor_r_max": float(design.rotor_r_max),
        "geoms": {}, "results": {}}
for gname, gn in GEOMS:
    barriers = H.build_barriers(gen, gn, lb, ub)
    assert barriers is not None, f"{gname} infeasible at wide=False"
    dump["geoms"][gname] = [np.asarray(b, float)[:, :2].tolist() for b in barriers]
    design.add_rotor()
    for b in barriers:
        design.add_rotor_barrier(b)
    for cname, dq in CURS:
        res = design.compute(*dq, NUM_CORES=4)
        T, _, rip = H.analyze_results(np.asarray(res["Tor"], float))
        loss = float(sum(x * x for x in dq))
        rows[(gname, cname)] = (T, rip, loss)
        dump["results"][f"{gname}|{cname}"] = {"T": float(T), "ripple": float(rip), "loss": loss}
        print(f"[{gname} x {cname}]  T={T:7.3f}  rip={rip:6.2f}%  loss={loss:5.2f}", flush=True)
    design.delete_rotor()
design.close_project()

json.dump(dump, open("xcheck.json", "w"))
print("\n=== 2x2: T_mean [Nm] / ripple [%] (loss is current-only) ===")
print(f"{'':10s} {'I_dq1':>18s} {'I_joint':>18s}")
for gname, _ in GEOMS:
    cells = []
    for cname, _ in CURS:
        T, rip, _ = rows[(gname, cname)]
        cells.append(f"{T:6.2f} Nm / {rip:4.2f}%")
    print(f"{gname:10s} {cells[0]:>18s} {cells[1]:>18s}")
print("saved xcheck.json")
