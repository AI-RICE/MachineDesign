import os
import shutil

import numpy as np
import pandas as pd

from machine_design import (
    HacklGenerator_SixLambdas,
    analyze_results,
    load_design,
)

aedt_version = "2025.1"
mesh_sizes = np.round(np.arange(3, 0.0, -0.05), 2)
# mesh size, step size 0.05 mm, 2 is the converted to mm, e.g., 0.05 mm

r_stator_end = 0.7
offset = 0.7 / 2
num_cores = 4

project_name = "SynRM_test"
design_name = "Design01"

path_data = os.path.join(os.getcwd(), "data")
path_results = os.path.join(os.getcwd(), "mesh_sweep_results")

for p in [path_data, path_results]:
    os.makedirs(p, exist_ok=True)

file_name_aedt = os.path.join(path_data, f"{project_name}.aedt")
file_name_temp = os.path.join(path_data, f"{project_name}_sweep.aedt")

X_best = np.array(
    [
        8.95052421,
        23.01115583,
        33.03271713,
        14.34330278,
        27.92171436,
        37.3,
        0.21260669,
        0.31,
        0.41,
        0.3,
        0.4,
        0.46,
    ],
    dtype=float,
)
# point at the Pareto front

metadata = pd.DataFrame()

for mesh_size in mesh_sizes:
    # Copy clean template so each iteration starts with a fresh mesh state
    shutil.copy(file_name_aedt, file_name_temp)

    design = load_design(
        file_name_temp,
        project_name,
        design_name,
        aedt_version,
        non_graphical=True,
        new_desktop=True,
        close_on_exit=False,
    )
    generator = HacklGenerator_SixLambdas(design, r_stator_end, offset=offset)
    params = generator.X_to_params(X_best)
    generator.set_parameters(params)
    barriers = generator.generate_barriers()
    barriers = generator.split_barriers(barriers)
    if not generator.feasible_barriers(barriers):
        raise RuntimeError("Selected SixLambdas design is not feasible.")

    design.add_rotor()
    for barrier in barriers:
        design.add_rotor_barrier(barrier)

    m2d = design.m2d
    m2d.set_core_losses("Rotor", core_loss_on_field=False)
    id_coils = m2d.modeler.get_objects_w_string(string_name="Coil", case_sensitive=True)

    # set maximum_length for each coil, stator, and rotor.
    m2d.mesh.assign_length_mesh(
        assignment=id_coils, inside_selection=True,
        maximum_length=mesh_size, maximum_elements=None,
    )
    m2d.mesh.assign_length_mesh(
        assignment="Stator", inside_selection=True,
        maximum_length=mesh_size, maximum_elements=None,
    )
    m2d.mesh.assign_length_mesh(
        assignment=design.rotor_id, inside_selection=True,
        maximum_length=mesh_size, maximum_elements=None,
    )

    m2d.analyze_setup(design.setup_name, use_auto_settings=False, cores=num_cores)

    solutions = m2d.post.get_solution_data(
        expressions="Moving1.Torque", primary_sweep_variable="Time"
    )
    try:
        Tor = solutions.data_magnitude()
    except AttributeError:
        Tor = None
    # in case of error, let Tor be None

    if Tor is None:
        TorAvg, TorRippleRms = np.nan, np.nan
    else:
        TorAvg, _, TorRippleRms = analyze_results(Tor)

    metadata = pd.concat(
        (
            metadata,
            pd.DataFrame([{"maximum_length_mm": mesh_size, "T": TorAvg, "ripple": TorRippleRms}]),
        ),
        ignore_index=True,
    )

    # Close AEDT after each iteration; temp file is discarded (overwritten next iteration)
    design.close_project()

csv_path = os.path.join(path_results, "mesh_sweep_results.csv")
metadata.to_csv(csv_path, index=False)
print(f"Results saved to: {csv_path}")
