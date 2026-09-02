# export ANSYSEM_ROOT241=/data/AnsysEM/v241/Linux64

import os

import numpy as np
import pandas as pd

from machine_design.designs import load_design
from machine_design.optimization import (
    HacklGenerator_3BrokenLines,
    HacklGenerator_OneLambda,
    HacklGenerator_SixLambdas,
    analyze_results,
    plot_barriers,
    save_params,
)
from motors.motor1 import Computation, Geometry

aedt_version = "2024.1"
n_designs = 50
r_stator_end = 0.7
offset = 0.7 / 2
num_cores = 4
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
generators = [
    HacklGenerator_OneLambda(design, r_stator_end, offset=offset),
    HacklGenerator_SixLambdas(design, r_stator_end, offset=offset),
    HacklGenerator_3BrokenLines(design, r_stator_end, offset=offset),
]

metadata = pd.DataFrame()
for i in range(0, n_designs):
    for generator in generators:
        # Generate a feasible design
        while True:
            params = generator.random_parameters()
            generator.set_parameters(params)
            barriers = generator.generate_barriers()
            barriers = generator.split_barriers(barriers)
            feasible = generator.feasible_barriers(barriers)
            if feasible:
                break

        # Generate the geometry
        design.add_rotor()
        for barrier in barriers:
            design.add_rotor_barrier(barrier)

        # Compute the torque
        Tor = design.compute(NUM_CORES=num_cores)
        # Tor = design.compute(num_cores)
        if Tor is None:
            TorAvg, TorRippleRms = np.nan, np.nan
        else:
            TorAvg, _, TorRippleRms = analyze_results(Tor)

        # Delete the rotor
        design.delete_rotor()

        # Potentially save the design
        if plot_design:
            title = f"Torque mean value: {np.round(TorAvg, 2)} Nm, ripple relative value: {np.round(TorRippleRms, 2)} %"
            file_name = f"{path_results}/design_{generator.name}_{i}"
            plot_barriers(barriers, design, title=title, file_name=f"{file_name}.png")
            save_params(params, f"{file_name}.pkl")

        metadata_new = {
            "method": generator.name,
            "design": i,
            "T": TorAvg,
            "ripple": TorRippleRms,
        }
        metadata = pd.concat((metadata, pd.DataFrame([metadata_new])), ignore_index=True)
        metadata.to_csv(f"{path_results}/metadata.csv", index=False)

design.close_project()
