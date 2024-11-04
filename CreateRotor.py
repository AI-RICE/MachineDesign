import time
import numpy as np

# Define constants
AEDT_VERSION = "2024.1"
NUM_CORES = 4
NG_MODE = False  #non-graphical mode
CLS_EXIT = True #close on exit

#materials
Fe = "Cogent Power - M350-50A, B-H at 50Hz"

# main definitions
geom_params = {
    "DiaStatorGap": "79mm",
    "Airgap": "0.225mm",
    "DiaShaft": "25mm",
}
# model parameters
PolePairs = 2
f = 50 #[Hz]
RotSpeed = 60*f/PolePairs #[rpm]
mod_params = {
    "Poles": f"2*{PolePairs}",
    "SymmetryFactor": "Poles",
}
setup_name = "Setup1"

from ansys.aedt.core import Desktop, Maxwell2d

desktop = Desktop(specified_version=AEDT_VERSION, new_desktop=False, close_on_exit=False)
desktop.load_project(r"d:\DATA\Software\PyAnsys\SynRM_test.aedt")
m2d=Maxwell2d()
m2d.set_active_design("Design01")
modeler = m2d.modeler

# Rotor quarter
rot_points = [
    ["DiaShaft/2*cos(360deg/SymmetryFactor)", "DiaShaft/2*sin(360deg/SymmetryFactor)", "0mm"],
    ["DiaShaft/2*cos(360deg/(2*SymmetryFactor))", "DiaShaft/2*sin(360deg/(2*SymmetryFactor))", "0mm"],
    ["DiaShaft/2", "0mm", "0mm"],
    ["DiaStatorGap/2-Airgap", "0mm", "0mm"],
    ["(DiaStatorGap/2-Airgap)*cos(360deg/(2*SymmetryFactor))", "(DiaStatorGap/2-Airgap)*sin(360deg/(2*SymmetryFactor))", "0mm"],
    ["(DiaStatorGap/2-Airgap)*cos(360deg/SymmetryFactor)", "(DiaStatorGap/2-Airgap)*sin(360deg/SymmetryFactor)", "0mm"],
]

# Create Rotor
rotor_id = modeler.create_polyline(
    points=rot_points, segment_type=[ "Arc","Line", "Arc"], cover_surface=True, name="Rotor"
)
#Rotor properties
rotor_id.material_name = Fe
rotor_id.color = (192, 192, 192)  # rgb
rotor_id.transparency = 0.0
#Rotor geometry modification
barrier_points = [
    ["14", "14", "0"],
    ["34","2","0"],
    ["36","2","0"],
    ["20","20","0"],
    ["2", "36", "0"],
    ["2", "34", "0"],
    ["14", "14", "0"],
]
barrier_id = modeler.create_polyline(
    points=barrier_points, segment_type=modeler.polyline_segment("Spline", num_points=7), cover_surface=True, name="Barrier"
)
barr_subtr = rotor_id.subtract(barrier_id)
modeler.delete(barrier_id)

m2d.mesh.assign_length_mesh(
    assignment=rotor_id,
    inside_selection=True,
    maximum_length=3,
    maximum_elements=None,
    name="rotor",
)
#core loss rotor
m2d.set_core_losses("Rotor", core_loss_on_field=False)

# Analyze
m2d.save_project()
m2d.analyze_setup(setup_name, use_auto_settings=False, cores=NUM_CORES)

solutions = m2d.post.get_solution_data(
    expressions="Moving1.Torque", primary_sweep_variable="Time"
)
Tor = solutions.data_magnitude()
TorAvg = np.mean(Tor[:-1])
TorAvgAC = np.mean(np.abs(Tor[:-1]-TorAvg))
TorRmsAC = np.sqrt(np.mean(np.square(Tor[:-1]-TorAvg)))
TorRippleAvg = TorAvgAC/TorAvg*100
TorRippleRms = TorRmsAC/TorAvg*100

print("\nTorque mean value: {:.2f} Nm".format(TorAvg))
# print("\nTorque ripple mean value: {:.2f} Nm".format(TorAvgAC))
print("\nTorque ripple rms value: {:.2f} Nm".format(TorRmsAC))
# print("\nTorque ripple relative value: {:.2f} %".format(TorRippleAvg))
print("\nTorque ripple relative value: {:.2f} %\n".format(TorRippleRms))

input("Press Enter to continue...")

modeler.delete(rotor_id)

m2d.save_project()
m2d.close_desktop()