"""First 60-slot build: a 5-phase machine on a common 60-slot stator, using the
parametric winding generator (winding.py auto-produces the 60-slot/5-phase layout,
q=3). Confirms the generator + a resized 60-slot stator build, mesh and solve.

Slot widths scaled ~x0.66 from the 40-slot 5-phase stator to fit the 60-slot bore
pitch (~4.1 mm at the gap). Rotor: OneLambda seed 0 (same as demo_horizon), 50 Hz.
"""
import os

import numpy as np

from machine_design import HacklGenerator_OneLambda, analyze_results
from machine_design.design2 import Design2


class Design2_60(Design2):
    """5-phase machine on a 60-slot common stator."""
    def set_geom_params(self):
        super().set_geom_params()
        self.geom_params["SlotNumber"] = "60"

    def set_slot_params(self):
        super().set_slot_params()
        self.slot_params.update(Bs0="1.5mm", Bs1="2.0mm", Bs2="2.85mm",
                                Rs="1.0mm", SetAngle="6deg")

    def set_winds_params(self):
        super().set_winds_params()
        self.wind_params["CoilPitch"] = "15"   # full pitch over one pole (60/4)


path = os.path.join(os.getcwd(), "data", "m5_60.aedt")
os.makedirs(os.path.dirname(path), exist_ok=True)
for ext in ("", ".lock"):
    p = path + ext
    if os.path.exists(p):
        os.remove(p)

mk = dict(version="2024.2", non_graphical=True, new_desktop=True, close_on_exit=True)
print("creating 60-slot 5-phase stator (generic winding, m=5, Q=60, q=3)...", flush=True)
d = Design2_60.create("m5_60", "Design01", path, **mk)
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
print("rotor built; solving (Id1=Iq1=7.07, dq3=0)...", flush=True)

res = d.compute(7.0711, 7.0711, 0.0, 0.0, NUM_CORES=4)
Tor = np.asarray(res["Tor"], float)
T, _, rip = analyze_results(Tor)
print(f"[60slot-5ph] BUILT+SOLVED  T_mean={T:.4f} Nm  ripple={rip:.3f}%  n={Tor.size}", flush=True)
d.save_project()
d.close_project()
print("DONE", flush=True)
