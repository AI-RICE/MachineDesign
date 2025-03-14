import os
import numpy as np
from ansys.aedt.core import Maxwell2d
from design import Design

project_name = "SynRM_test"
design_name = "Design01"
path = os.path.join(os.getcwd(), 'data')
os.makedirs(path, exist_ok=True)
file_design = 'results/desing.png'

# Define constants
AEDT_VERSION = "2024.2"
NUM_CORES = 4
NG_MODE = True  #non-graphical mode
CLS_EXIT = True #close on exit

m2d = Maxwell2d(project=project_name, design=design_name, version=AEDT_VERSION,
                non_graphical=NG_MODE, new_desktop=False,
                close_on_exit=CLS_EXIT, student_version=False, solution_type="TransientXY", )
m2d.save_project(f"{path}/{project_name}.aedt")

design = Design(m2d)
design.create_stator()
Tor = design.create_rotor(NUM_CORES)
design.analyze_results(Tor)
design.save_design(file_design)
design.delete_rotor()
design.close_session()
