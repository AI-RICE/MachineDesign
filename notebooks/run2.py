"""5-phase Design2 single-design evaluation in ANSYS Maxwell 2D, WITH barriers.

The stock ``load_design`` / ``run.py`` only build the 3-phase base ``Design``.
This script instantiates the 5-phase ``Design2`` (dq1/dq3 windings + the 5-phase
Clarke/Park transforms), generates a feasible flux-barrier rotor with a
``BarrierGenerator`` (as ``run.py`` does for 3-phase), applies a dq1 current
setpoint at 45 deg, and solves a single transient. Prints mean torque / ripple.

Run on bayes:
    export ANSYSEM_ROOT242=/data/Ansys/v242/Linux64/
    ./venv_5f/bin/python notebooks/run2.py
"""

import os

import numpy as np

from machine_design import HacklGenerator_OneLambda, analyze_results
from machine_design.design2 import Design2

aedt_version = "2024.2"
num_cores = 4
project_name = "SynRM5f_barriers"
design_name = "Design01"

# Barrier generator geometry (same family/values as the 3-phase run.py).
r_stator_end = 0.7
offset = 0.7 / 2
seed = 0  # reproducible barrier geometry

# dq1 current setpoint (A) at 45 deg => Im1 = 10 A; dq3 left at zero.
Id1, Iq1, Id3, Iq3 = 7.0711, 7.0711, 0.0, 0.0

path_data = os.path.join(os.getcwd(), "data")
path_results = "results"
for p in (path_data, path_results):
    os.makedirs(p, exist_ok=True)
file_name_aedt = f"{path_data}/{project_name}.aedt"

kw = dict(version=aedt_version, non_graphical=True, new_desktop=False, close_on_exit=True)
if os.path.exists(file_name_aedt):
    print(f"[run2] loading existing project {file_name_aedt}")
    design = Design2.load(file_name_aedt, **kw)
else:
    print(f"[run2] creating new 5-phase project {file_name_aedt}")
    design = Design2.create(project_name, design_name, file_name_aedt, **kw)

# Generate a feasible flux-barrier rotor.
print(f"[run2] generating feasible barriers (seed={seed}, r_min={design.rotor_r_min}, r_max={design.rotor_r_max})")
np.random.seed(seed)
generator = HacklGenerator_OneLambda(design, r_stator_end, offset=offset)
while True:
    params = generator.random_parameters()
    generator.set_parameters(params)
    barriers = generator.generate_barriers()
    barriers = generator.split_barriers(barriers)
    if generator.feasible_barriers(barriers):
        break
print(f"[run2] feasible barrier set with {len(barriers)} polyline(s)")

print("[run2] building rotor + barriers")
design.add_rotor()
for barrier in barriers:
    design.add_rotor_barrier(barrier)

print(f"[run2] solving: Id1={Id1} Iq1={Iq1} Id3={Id3} Iq3={Iq3} A, cores={num_cores}")
Tor = design.compute(Id1, Iq1, Id3, Iq3, NUM_CORES=num_cores)

if Tor is None:
    print("[run2] compute() returned None — torque not extracted")
else:
    Tor = np.asarray(Tor, dtype=float)
    T_mean, T_se, T_ripple = analyze_results(Tor)
    print(f"[run2] SOLVED  T_mean={T_mean:.4f} Nm  ripple={T_ripple:.3f} %  n_pts={Tor.size}")

design.save_project()
design.close_project()
print("[run2] done")
