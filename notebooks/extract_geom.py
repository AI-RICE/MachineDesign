"""Extract rotor flux-barrier polylines (one pole sector) for the 3x2 regime grid,
WITHOUT FEA (mock generator). Rows = regimes (no-limit / current-limited /
voltage-limited); columns = G* (dq1-only) and joint (dq1+dq3). Each cell's label
is built from its own *_best.json so the numbers are always accurate. Writes
geom_shapes6.json for plot_geom.py.
"""
import json

import numpy as np

import h0h1_par as P
import h0h1_study as H


class _D:
    rotor_r_min, rotor_r_max = 12.5, 39.275


gen = P.make_generator(_D(), True)
lb, ub = H.geom_bounds_arrays(gen)


def label_nolim(d, gstar):
    role = "$G^\\star$: min-loss $dq1$" if gstar else "joint: min-loss $dq1{+}dq3$"
    note = f"$|I|$={d['Irms_equiv']:.1f} A  ($T$={d['T']:.1f} Nm, rip {d['ripple']:.1f}%)"
    return role, note


def label_maxT(d, gstar, regime):
    if gstar:
        role = "$G^\\star$: max-$T$ $dq1$"
        note = f"$T$={d['T']:.1f} Nm (rip {d['ripple']:.1f}%)"
    else:
        role = "joint: max-$T$ $dq1{+}dq3$"
        note = f"$T$={d['T_joint']:.1f} Nm  ($+{d['dT_pct']:.0f}\\%$, rip {d['ripple']:.1f}%)"
    return role, note


# (key, file, geom_json_key, regime_title, labeller, is_gstar)
CASES = [
    ("nolim_gstar", "results_nolimit/dq1_best.json",    "geom_norm",  "no limit (min loss @ 20 Nm)", label_nolim, True),
    ("nolim_joint", "results_nolimit/joint_best.json",  "geom_norm",  "no limit (min loss @ 20 Nm)", label_nolim, False),
    ("curr_gstar",  "results_volt50/stage1_best.json",  "geom_norm",  "current-limited (50 Hz)",     label_maxT,  True),
    ("curr_joint",  "results_volt50/joint_best.json",   "geom_joint", "current-limited (50 Hz)",     label_maxT,  False),
    ("volt_gstar",  "results_volt71/stage1_best.json",  "geom_norm",  "voltage-limited (71.2 Hz)",   label_maxT,  True),
    ("volt_joint",  "results_volt71/joint_best.json",   "geom_joint", "voltage-limited (71.2 Hz)",   label_maxT,  False),
]

out = {"rotor_r_min": float(_D.rotor_r_min), "rotor_r_max": float(_D.rotor_r_max), "cases": {}}
for key, fn, gk, regime, labeller, gstar in CASES:
    d = json.load(open(fn))
    gn = np.asarray(d[gk], float)
    bars = H.build_barriers(gen, gn, lb, ub)
    assert bars is not None, f"{key} infeasible"
    role, note = (labeller(d, gstar) if labeller is label_nolim else labeller(d, gstar, regime))
    out["cases"][key] = {"barriers": [np.asarray(b, float)[:, :2].tolist() for b in bars],
                         "regime": regime, "role": role, "note": note}
    print(f"{key}: {len(bars)} barriers | {regime} | {role} | {note}", flush=True)

json.dump(out, open("geom_shapes6.json", "w"))
print("saved geom_shapes6.json")
