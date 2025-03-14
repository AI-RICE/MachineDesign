import os
import numpy as np
from ansys.aedt.core import Maxwell2d
from design import Design

project_name = "SynRM_test"
design_name = "Design01"
path = os.path.join(os.getcwd(), 'data')
os.makedirs(path, exist_ok=True)
file_design = 'results/design.png'

# Define constants
AEDT_VERSION = "2024.2"
NUM_CORES = 4
NG_MODE = True  #non-graphical mode
CLS_EXIT = True #close on exit

m2d = Maxwell2d(project=project_name, design=design_name, version=AEDT_VERSION,
                non_graphical=NG_MODE, new_desktop=False,
                close_on_exit=CLS_EXIT, student_version=False, solution_type="TransientXY", )
m2d.save_project(f"{path}/{project_name}.aedt")

barrier_points = [
    [14, 14, 0],
    [34, 2, 0],
    [36, 2, 0],
    [20, 20, 0],
    [2, 36, 0],
    [2, 34, 0],
    [14, 14, 0],
]

design = Design(m2d)
design.create_stator()
design.add_rotor()
design.add_rotor_holes(barrier_points)
design.save_design(file_design)
Tor = design.compute(NUM_CORES)
results = design.analyze_results(Tor)
design.print_results(*results)
design.delete_rotor()
design.close_session()
