# export ANSYSEM_ROOT241=/data/AnsysEM/v241/Linux64

import os

import numpy as np
import pandas as pd

from machine_design import (
    Design,
    HacklGenerator_OneLambda,
    HacklGenerator_SixLambdas,
    HacklGenerator_3BrokenLines,
    analyze_results,
    plot_barriers,
    save_params,
)

project_name = "SynRM_test"
design_name = "Design01"
path_data = os.path.join(os.getcwd(), "data")
path_results = "results"
for path in [path_data, path_results]:
    os.makedirs(path, exist_ok=True)
file_name_aedt = f"{path_data}/{project_name}.aedt"
plot_design = True
n_designs = 50

# Define constants
AEDT_VERSION = "2024.1"
NUM_CORES = 4
NG_MODE = True  # non-graphical mode
CLS_EXIT = True  # close on exit

if not os.path.exists(file_name_aedt):
    design = Design.create(
        project_name,
        design_name,
        file_name_aedt,
        version=AEDT_VERSION,
        non_graphical=NG_MODE,
        new_desktop=False,
        close_on_exit=CLS_EXIT,
    )
else:
    design = Design.load(
        file_name_aedt,
        version=AEDT_VERSION,
        non_graphical=NG_MODE,
        new_desktop=False,
        close_on_exit=CLS_EXIT,
    )

r_stator_end = 0.7
offset = 0.7 / 2
generators = [
    HacklGenerator_OneLambda(design, r_stator_end, offset=offset),
    HacklGenerator_SixLambdas(design, r_stator_end, offset=offset),
    HacklGenerator_3BrokenLines(design, r_stator_end, offset=offset),
]

metadata = pd.DataFrame()
for i in range(0, n_designs):
    for generator in generators:
        params = generator.random_parameters()
        generator.set_parameters(params)
        barriers = generator.generate_barriers()
        barriers = generator.split_barriers(barriers)

        design.add_rotor()

        for barrier in barriers:
            design.add_rotor_barrier(barrier)

        # Compute the torque
        Tor = design.compute(NUM_CORES)
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
