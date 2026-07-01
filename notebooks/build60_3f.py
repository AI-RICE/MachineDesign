"""De-risk the 3-phase 60-slot counterpart: build Design_60 (3-phase, SAME 60-slot
stator + rotor as Design2_60) and solve one point. Confirms the generic winding
works for m=3 at 60 slots (q=5) and the 3-phase compute/extract interface.

Drives the 3-phase machine via (Id,Iq) -> Im,epsI; returns the {Tor} dict so the
eval machinery can consume it later. Rotor: OneLambda seed 0, 50 Hz.
"""
import os

import numpy as np

from machine_design import HacklGenerator_OneLambda, analyze_results
from machine_design.design import Design


class Design_60(Design):
    """3-phase machine on the common 60-slot stator (same slot params as Design2_60)."""

    def set_geom_params(self):
        super().set_geom_params()
        self.geom_params["SlotNumber"] = "60"

    def set_slot_params(self):
        super().set_slot_params()
        self.slot_params.update(Bs0="1.5mm", Bs1="2.0mm", Bs2="2.85mm",
                                Rs="1.0mm", SetAngle="6deg")

    def set_winds_params(self):
        super().set_winds_params()
        self.wind_params["CoilPitch"] = "15"

    def set_variables(self, Id, Iq):
        self.m2d["Im"] = f"{float(np.hypot(Id, Iq))}A"
        self.m2d["epsI"] = f"{float(np.arctan2(Iq, Id))}rad"

    def extract_results(self, solutions):
        tor = np.asarray(solutions.data_real("Moving1.Torque"), float)
        return {"Tor": tor, "means": {}}


path = os.path.join(os.getcwd(), "data", "m3_60.aedt")
os.makedirs(os.path.dirname(path), exist_ok=True)
for ext in ("", ".lock"):
    if os.path.exists(path + ext):
        os.remove(path + ext)

mk = dict(version="2024.2", non_graphical=True, new_desktop=True, close_on_exit=True)
print("creating 60-slot 3-phase stator (generic winding, m=3, Q=60, q=5)...", flush=True)
d = Design_60.create("m3_60", "Design01", path, **mk)
print("stator + winding built OK", flush=True)
d.m2d["f"] = "50Hz"
d.m2d["RotSpeed"] = f"{60.0 * 50 / 2}rpm"
d.m2d["Nper"] = "1"

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
print("rotor built; solving (Id=Iq=7.07, Im=10)...", flush=True)

res = d.compute(7.0711, 7.0711, NUM_CORES=4)
Tor = np.asarray(res["Tor"], float)
T, _, rip = analyze_results(Tor)
print(f"[60slot-3ph] BUILT+SOLVED  T_mean={T:.4f} Nm  ripple={rip:.3f}%  n={Tor.size}", flush=True)
d.save_project()
d.close_project()
print("DONE", flush=True)
