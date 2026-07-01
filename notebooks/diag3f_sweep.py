"""Diagnostic: is the 3-phase 60-slot machine sound? Sweep the current angle at
fixed |I| and report torque + ripple. A clean MTPA-like peak (sizable torque, low
ripple at some angle) => sound, the min-loss BO will find it. Garbage at all angles
=> real winding/alignment bug in the (never-before-run) 3-phase path.
"""
import os

import numpy as np

from machine_design import HacklGenerator_OneLambda, analyze_results
from machine_design.design import Design


class Design_60(Design):
    def set_geom_params(self):
        super().set_geom_params(); self.geom_params["SlotNumber"] = "60"

    def set_slot_params(self):
        super().set_slot_params()
        self.slot_params.update(Bs0="1.5mm", Bs1="2.0mm", Bs2="2.85mm", Rs="1.0mm", SetAngle="6deg")

    def set_winds_params(self):
        super().set_winds_params(); self.wind_params["CoilPitch"] = "15"

    def set_variables(self, Id, Iq):
        self.m2d["Im"] = f"{float(np.hypot(Id, Iq))}A"
        self.m2d["epsI"] = f"{float(np.arctan2(Iq, Id))}rad"

    def extract_results(self, solutions):
        return {"Tor": np.asarray(solutions.data_real("Moving1.Torque"), float), "means": {}}


path = os.path.join(os.getcwd(), "data", "m3_60d.aedt")
for ext in ("", ".lock"):
    if os.path.exists(path + ext):
        os.remove(path + ext)
mk = dict(version="2024.2", non_graphical=True, new_desktop=True, close_on_exit=True)
d = Design_60.create("m3_60d", "Design01", path, **mk)
d.m2d["f"] = "50Hz"; d.m2d["RotSpeed"] = f"{60.0 * 50 / 2}rpm"; d.m2d["Nper"] = "1"
np.random.seed(0)
gen = HacklGenerator_OneLambda(d, 0.7, offset=0.35)
while True:
    gen.set_parameters(gen.random_parameters())
    barriers = gen.split_barriers(gen.generate_barriers())
    if gen.feasible_barriers(barriers):
        break
d.add_rotor()
for b in barriers:
    d.add_rotor_barrier(b)

IM = 10.0
print("angle[deg]  T_mean[Nm]  ripple[%]", flush=True)
for a_deg in (10, 30, 45, 60, 80, 110, 135):
    a = np.deg2rad(a_deg)
    res = d.compute(IM * np.cos(a), IM * np.sin(a), NUM_CORES=4)
    T, _, rip = analyze_results(np.asarray(res["Tor"], float))
    print(f"  {a_deg:3d}      {T:8.3f}    {rip:8.2f}", flush=True)
d.close_project()
print("DONE", flush=True)
