import os
import numpy as np
from ansys.aedt.core import Maxwell2d
from design import Design

project_name = "SynRM_test"
design_name = "Design01"
path = os.path.join(os.getcwd(), 'data')
os.makedirs(path, exist_ok=True)
os.makedirs('results', exist_ok=True)
file_name_aedt = f'{path}/{project_name}.aedt'

# Define constants
AEDT_VERSION = "2024.2"
NUM_CORES = 4
NG_MODE = True  #non-graphical mode
CLS_EXIT = True #close on exit

if not os.path.exists(file_name_aedt):
    m2d = Maxwell2d(project=project_name, design=design_name, version=AEDT_VERSION,
        non_graphical=NG_MODE, new_desktop=False,
        close_on_exit=CLS_EXIT, student_version=False, solution_type="TransientXY")
    design = Design(m2d)
    design.create_stator()
    design.save_project(file_name_aedt)
else:
    design = Design()
    design.load_stator(file_name_aedt, version=AEDT_VERSION, non_graphical=NG_MODE,
        new_desktop=False, close_on_exit=CLS_EXIT, student_version=False)

n_holes = 1
r_min = design.rotor_r_min
r_max = design.rotor_r_max
segment_type = design.m2d.modeler.polyline_segment("Spline", num_points=7)
#segment_type = None
for i in range(10):
    mid_r1 = (r_min+1+(r_max-r_min)/3*np.random.random()) / np.sqrt(2)
    mid_r2 = (r_max-2-(r_max-r_min)/3*np.random.random()) / np.sqrt(2)
    side_x = 30+6*np.random.random()
    side_y = 1+1*np.random.random()
    side_w = 1+1*np.random.random()

    barrier_points = [
        [mid_r1, mid_r1],
        [side_x, side_y],
        [side_x+side_w, side_y],
        [mid_r2, mid_r2],
        [side_y, side_x+side_w],
        [side_y, side_x],
        [mid_r1, mid_r1],
    ]
    file_design = f'results/design_{i}.png'

    design.add_rotor()
    design.add_rotor_holes(barrier_points, n=n_holes, segment_type=segment_type)

    design.save_design(file_design)
    #Tor = design.compute(NUM_CORES)
    #results = design.analyze_results(Tor)
    #design.print_results(*results)
    design.delete_rotor()

design.close_project()
