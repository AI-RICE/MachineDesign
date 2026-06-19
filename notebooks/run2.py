"""5-phase Design2 single-design evaluation in ANSYS Maxwell 2D, WITH barriers.

The stock ``load_design`` / ``run.py`` only build the 3-phase base ``Design``.
This instantiates the 5-phase ``Design2`` (dq1/dq3 windings + the 5-phase
Clarke/Park transforms), generates a feasible flux-barrier rotor with a
``BarrierGenerator``, applies a dq1/dq3 current setpoint, and solves a single
transient. Prints mean torque / ripple.

The simulated window (``--nper``, in electrical periods) is a knob: Design2
defaults to 1/10 = 1/(2*m), one ideal torque-ripple period that assumes the
m-phase symmetry holds. Use ``--nper 1`` for a full period that does not rely on
that assumption (robust to dq1<->dq3 saturation cross-coupling / FEA asymmetry).

Run on bayes:
    export ANSYSEM_ROOT242=/data/Ansys/v242/Linux64/
    ./venv_5f/bin/python notebooks/run2.py --nper 1 --id3 4 --iq3 2 --project SynRM5f_inj_full
"""

import argparse
import os

import numpy as np

from machine_design import HacklGenerator_OneLambda, analyze_results
from machine_design.design2 import Design2

p = argparse.ArgumentParser()
p.add_argument("--project", default="SynRM5f_barriers")
p.add_argument("--nper", default="1", help="simulated window in electrical periods (e.g. '1' or '1/10')")
p.add_argument("--id1", type=float, default=7.0711)
p.add_argument("--iq1", type=float, default=7.0711)
p.add_argument("--id3", type=float, default=0.0)
p.add_argument("--iq3", type=float, default=0.0)
p.add_argument("--seed", type=int, default=0, help="RNG seed for the barrier geometry")
p.add_argument("--num-cores", type=int, default=4)
p.add_argument("--aedt-version", default="2024.2")
args = p.parse_args()

design_name = "Design01"
r_stator_end = 0.7
offset = 0.7 / 2

path_data = os.path.join(os.getcwd(), "data")
os.makedirs(path_data, exist_ok=True)
file_name_aedt = f"{path_data}/{args.project}.aedt"

kw = dict(version=args.aedt_version, non_graphical=True, new_desktop=False, close_on_exit=True)
if os.path.exists(file_name_aedt):
    print(f"[run2] loading existing project {file_name_aedt}")
    design = Design2.load(file_name_aedt, **kw)
else:
    print(f"[run2] creating new 5-phase project {file_name_aedt}")
    design = Design2.create(args.project, design_name, file_name_aedt, **kw)

# Override the simulated window length (StopTime = "Nper/f" tracks this var).
design.m2d["Nper"] = args.nper
print(f"[run2] Nper = {args.nper} electrical period(s) (StopTime = Nper/f)")

# Generate a feasible flux-barrier rotor.
print(f"[run2] barriers: seed={args.seed}, r_min={design.rotor_r_min}, r_max={design.rotor_r_max}")
np.random.seed(args.seed)
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

print(f"[run2] solving: Id1={args.id1} Iq1={args.iq1} Id3={args.id3} Iq3={args.iq3} A, cores={args.num_cores}")
Tor = design.compute(args.id1, args.iq1, args.id3, args.iq3, NUM_CORES=args.num_cores)

if Tor is None:
    print("[run2] compute() returned None — torque not extracted")
else:
    Tor = np.asarray(Tor, dtype=float)
    T_mean, T_se, T_ripple = analyze_results(Tor)
    print(f"[run2] SOLVED  nper={args.nper}  T_mean={T_mean:.4f} Nm  ripple={T_ripple:.3f} %  n_pts={Tor.size}")

design.save_project()
design.close_project()
print("[run2] done")
