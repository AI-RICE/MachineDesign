# export ANSYSEM_ROOT241=/data/AnsysEM/v241/Linux64

import os

import numpy as np
import pandas as pd

from machine_design.config import load_config
from machine_design.designs import load_design
from machine_design.optimization import analyze_results
from motors.template_3f_36s import Computation, Geometry

config = load_config()
aedt_version = config["aedt_version"]
num_cores = config["num_cores"]
n_designs = 1
current_setpoint = (1.5, 1.5)
plot_design = True

project_name = "SynRM_test"
design_name = "Design01"
path_data = os.path.join(os.getcwd(), "data")
path_results = "results"
for path in [path_data, path_results]:
    os.makedirs(path, exist_ok=True)
file_name_aedt = f"{path_data}/{project_name}.aedt"

geometry = Geometry()
computation = Computation(geometry)
design = load_design(file_name_aedt, project_name, design_name, aedt_version, geometry, computation)

metadata = pd.DataFrame()
for i in range(0, n_designs):
    # Generate the geometry
    design.add_rotor()

    # Compute the torque
    outputs = design.compute(*current_setpoint, NUM_CORES=num_cores)
    if outputs is None:
        TorAvg, TorRippleRms = np.nan, np.nan
    else:
        TorAvg, _, TorRippleRms = analyze_results(outputs["Moving1.Torque"])

    # Potentially save the design
    if plot_design:
        title = f"Torque mean value: {np.round(TorAvg, 2)} Nm, ripple relative value: {np.round(TorRippleRms, 2)} %"
        file_name = f"{path_results}/design_PMCore_{i}"
        design.m2d.plot(show=False, output_file=os.path.abspath(f"{file_name}.png"), view="xy", plot_air_objects=False)
        print(f"{file_name}: {title}")

    # Delete the rotor
    design.delete_rotor()

    metadata_new = {
        "method": "PMCore",
        "design": i,
        "T": TorAvg,
        "ripple": TorRippleRms,
    }
    metadata = pd.concat((metadata, pd.DataFrame([metadata_new])), ignore_index=True)
    metadata.to_csv(f"{path_results}/metadata.csv", index=False)

design.close_project()
