import os

import motors.motor2 as _m2
from machine_design.designs import load_design
from motors.motor2 import Computation2, Geometry2

print("Confirming which motor2.py this run actually uses:")
print(f"  {_m2.__file__}\n")

aedt_version = "2025.1"
project_name = "Diagnose_fr_test"
design_name = "Design01"
num_cores = 4

CURRENT_SETPOINT = (0.0, 1.3, 1e-6, 0.0)

path_data = os.path.join(os.getcwd(), "data")
os.makedirs(path_data, exist_ok=True)
file_name_aedt = os.path.join(path_data, f"{project_name}.aedt")

geometry = Geometry2()
computation = Computation2(geometry)

computation.output_vars["diag_coeff_50"] = "Im1*2*pi*50"
computation.output_vars["diag_coeff_f"] = "Im1*2*pi*f"
#computation.output_vars["diag_coeff_fr"] = "Im1*2*pi*fr"
computation.solution_expressions = ["dIA_dt", "diag_coeff_50", "diag_coeff_f"]

design = load_design(
    file_name_aedt,
    project_name,
    design_name,
    aedt_version,
    geometry,
    computation,
    non_graphical=True,
    new_desktop=True,
    close_on_exit=False,
)

design.add_rotor()

Id1, Iq1, Id3, Iq3 = CURRENT_SETPOINT
out = design.compute(Id1, Iq1, Id3, Iq3, NUM_CORES=num_cores)

design.delete_rotor()

if out is None:
    print("compute() returned None")
else:
    for expr, val in out.items():
        print(f"  {expr:16s} = {val}")

design.close_project()
