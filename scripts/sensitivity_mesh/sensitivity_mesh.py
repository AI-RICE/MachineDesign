import os

import numpy as np
import pandas as pd

from machine_design.designs import load_design
from machine_design.optimization import HacklGenerator_3BrokenLines, HacklGenerator_SixLambdas, analyze_results
from motors.motor1 import Computation, Geometry

aedt_version = "2025.1"
design_name = "Design01"
num_cores = 4
r_stator_end = 0.7
offset = 0.7 / 2

mesh_sizes = np.round(np.arange(0.1, 3.05, 0.1), 2)
h_ref = 0.5  # converged value for the regions not being swept

path_data = os.path.join(os.getcwd(), "data")
path_results = os.path.join(os.getcwd(), "results")
os.makedirs(path_data, exist_ok=True)
os.makedirs(path_results, exist_ok=True)

pareto_points = {
    "HacklGenerator_SixLambdas": np.array(
        [8.95052421, 23.01115583, 33.03271713, 14.34330278, 27.92171436, 37.3, 0.21260669, 0.31, 0.41, 0.3, 0.4, 0.46],
        dtype=float,
    ),
    "HacklGenerator_3BrokenLines": np.array(
        [6.0538801, 23.5, 35.74582285, 10.6, 28.5920024, 40.7191182, 0.36646249, 22.65142559, 30.7050295, 38.0, 9.49161503, 2.0, 1.5],
        dtype=float,
    ),
}
generator_classes = {
    "HacklGenerator_SixLambdas": HacklGenerator_SixLambdas,
    "HacklGenerator_3BrokenLines": HacklGenerator_3BrokenLines,
}


def set_region_mesh(m2d, region_assignments, sizes):
    # rotor mesh is handled separately via design.compute(mesh_length=...)
    for op in list(m2d.mesh.meshoperations):
        if op.name in region_assignments:
            op.delete()
    for region, size in sizes.items():
        m2d.mesh.assign_length_mesh(assignment=region_assignments[region], inside_selection=True, maximum_length=size, maximum_elements=None, name=region)


records = []

for method, X_best in pareto_points.items():
    project_name = f"sensitivity_mesh_{method}"
    file_name_aedt = os.path.join(path_data, f"{project_name}.aedt")

    geometry = Geometry()
    computation = Computation(geometry)
    design = load_design(
        file_name_aedt,
        project_name,
        design_name,
        aedt_version,
        geometry,
        computation,
        non_graphical=True,
        new_desktop=True,
        close_on_exit=False,
    )

    generator = generator_classes[method](design, r_stator_end, offset=offset)
    params = generator.X_to_params(X_best)
    generator.set_parameters(params)
    barriers = generator.generate_barriers()
    barriers = generator.split_barriers(barriers)
    if not generator.feasible_barriers(barriers):
        raise RuntimeError(f"Selected {method} design is not feasible.")

    design.add_rotor()
    for barrier in barriers:
        design.add_rotor_barrier(barrier)

    m2d = design.m2d
    id_coils = m2d.modeler.get_objects_w_string(string_name="Coil", case_sensitive=True)
    region_assignments = {
        "coil": id_coils,
        "stator": "Stator",
        "band": "Band",
    }
    all_regions = list(region_assignments) + ["rotor"]

    for swept_region in all_regions:
        for mesh_size in mesh_sizes:
            sizes = {region: h_ref for region in region_assignments}
            mesh_length = h_ref
            if swept_region == "rotor":
                mesh_length = mesh_size
            else:
                sizes[swept_region] = mesh_size
            set_region_mesh(m2d, region_assignments, sizes)

            Tor = design.compute(NUM_CORES=num_cores, mesh_length=mesh_length)
            if Tor is None:
                TorAvg, TorRippleRms = np.nan, np.nan
            else:
                TorAvg, _, TorRippleRms = analyze_results(Tor)

            records.append(
                {
                    "method": method,
                    "region": swept_region,
                    "mesh_size_mm": mesh_size,
                    "TorAvg": TorAvg,
                    "TorRippleRms": TorRippleRms,
                }
            )

    design.delete_rotor()
    design.close_project()

metadata = pd.DataFrame(records)
csv_path = os.path.join(path_results, "sensitivity_mesh_results.csv")
metadata.to_csv(csv_path, index=False)