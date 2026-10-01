# export ANSYSEM_ROOT241=/data/AnsysEM/v241/Linux64

import os

import numpy as np
import pandas as pd

from machine_design.config import load_config
from machine_design.designs import load_design
from machine_design.optimization import (
    HacklGenerator_SixLambdas,
    analyze_results,
    plot_barriers,
    save_params,
)

phases=5 
if phases==3:
    from motors.synrm_3f_60s import Computation, Geometry
    current_bounds=np.array([[0.0, 0.0], [5.0, 5.0]])  #Id, Iq bounds for 3f
else:
    from motors.synrm_5f_60s import Computation, Geometry
    current_bounds=np.array([[0.0, 0.0, 0.0, 0.0], [3.0, 3.0, 3.0, 3.0]])  #Id1, Iq1, Id3, Iq3 bounds for 5f

config = load_config()
aedt_version = config["aedt_version"]
num_cores = config["num_cores"]
n_designs = 10
r_stator_end = 0.7
offset = 0.7 / 2
plot_design = True

project_name = f"SynRM_{phases}f_nominal"
design_name = "Design01"
path_data = os.path.join(os.getcwd(), "data")
path_results = f"results_{phases}f"
for path in [path_data, path_results]:
    os.makedirs(path, exist_ok=True)
file_name_aedt = f"{path_data}/{project_name}.aedt"

geometry = Geometry()
computation = Computation(geometry)
design = load_design(file_name_aedt, project_name, design_name, aedt_version, geometry, computation)
generator = HacklGenerator_SixLambdas(design, r_stator_end, offset=offset)

metadata = pd.DataFrame()
for i in range(0, n_designs):
        # Generate a feasible design
    while True:
        barrier_params = generator.random_parameters()
        generator.set_parameters(barrier_params)
        barriers = generator.generate_barriers()
        barriers = generator.split_barriers(barriers)
        feasible = generator.feasible_barriers(barriers)
        if feasible:
            break

    current_setpoint=np.random.uniform(current_bounds[0], current_bounds[1])

    # Generate the geometry
    design.add_rotor()
    for barrier in barriers:
        design.add_rotor_barrier(barrier)

    # Compute the torque
    out = design.compute(*current_setpoint, NUM_CORES=num_cores)
        # Tor = design.compute(num_cores)
    if out is None:
        TorAvg, TorRippleRms = np.nan, np.nan
    else:
        TorAvg, _, TorRippleRms = analyze_results(out["Moving1.Torque"])

    # Delete the rotor
    design.delete_rotor()

    loss=float(np.sum(current_setpoint**2))

    # Potentially save the design
    if plot_design:
        title = f"Torque mean value: {np.round(TorAvg, 2)} Nm, ripple relative value: {np.round(TorRippleRms, 2)} %, loss:{np.round(loss, 2)}"
        file_name = f"{path_results}/design_{generator.name}_{i}"
        plot_barriers(barriers, design, title=title, file_name=f"{file_name}.png")
        save_params((*barrier_params, current_setpoint), f"{file_name}.pkl")

    metadata_new = {
        "method": generator.name,
        "design": i,
        "T": TorAvg,
        "ripple": TorRippleRms,
        "loss": loss,
        }
    metadata = pd.concat((metadata, pd.DataFrame([metadata_new])), ignore_index=True)
    metadata.to_csv(f"{path_results}/metadata.csv", index=False)

design.close_project()
