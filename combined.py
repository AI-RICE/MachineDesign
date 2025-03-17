import os
from ansys.aedt.core import Maxwell2d
from design import Design

project_name = "SynRM_test"
design_name = "Design01"
path = os.path.join(os.getcwd(), 'data')
os.makedirs(path, exist_ok=True)
file_name_aedt = f'{path}/SynRM_test.aedt'

# Define constants
AEDT_VERSION = "2024.2"
NUM_CORES = 4
NG_MODE = True  #non-graphical mode
CLS_EXIT = True #close on exit

barrier_points = [
    [14, 14],
    [34, 2],
    [36, 2],
    [20, 20],
    [2, 36],
    [2, 34],
    [14, 14],
]

if not os.path.exists(file_name_aedt):
    m2d = Maxwell2d(project=project_name, design=design_name, version=AEDT_VERSION,
        non_graphical=NG_MODE, new_desktop=False,
        close_on_exit=CLS_EXIT, student_version=False, solution_type="TransientXY")
    design = Design(m2d)
    design.save_project(file_name_aedt)
    design.create_stator()
    design.save_project()
else:
    design = Design()
    design.load_stator(file_name_aedt, version=AEDT_VERSION, non_graphical=NG_MODE,
        new_desktop=False, close_on_exit=CLS_EXIT, student_version=False)

design.add_rotor()
design.add_rotor_holes(barrier_points)
Tor = design.compute(NUM_CORES)
results = design.analyze_results(Tor)
design.print_results(*results)
design.delete_rotor()
design.close_project()
