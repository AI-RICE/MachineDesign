import os

import numpy as np
import pandas as pd

from machine_design.designs import load_design
from machine_design.optimization import (
    HacklGenerator_3BrokenLines,
    HacklGenerator_SixLambdas,
    analyze_results,
)
from motors.motor1 import Computation, Geometry

aedt_version = "2025.1"
design_name = "Design01"
num_cores = 4
r_stator_end = 0.7
offset = 0.7 / 2

path_data = os.path.join(os.getcwd(), "data")
path_results = os.path.join(os.getcwd(), "results")
os.makedirs(path_data, exist_ok=True)
os.makedirs(path_results, exist_ok=True)

point_per_values = list(range(51, 1002, 50))

# Pareto selected points (root 5, with constraints)
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

metadata = pd.DataFrame()

for method, X_best in pareto_points.items():
    project_name = f"sensitivity_torque_{method}"
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

    for point_per in point_per_values:
        design.m2d.variable_manager["PointPer"] = str(point_per)
        Tor = design.compute(NUM_CORES=num_cores)

        if Tor is None:
            n_points, TorAvg, TorRippleRms = 0, np.nan, np.nan
        else:
            n_points = len(Tor)
            TorAvg, _, TorRippleRms = analyze_results(Tor)

        metadata_new = {
            "method": method,
            "point_per": point_per,
            "n_points": n_points,
            "TorAvg": TorAvg,
            "TorRippleRms": TorRippleRms,
        }
        metadata = pd.concat((metadata, pd.DataFrame([metadata_new])), ignore_index=True)

    design.delete_rotor()
    design.close_project()

csv_path = os.path.join(path_results, "sensitivity_torque_results.csv")
metadata.to_csv(csv_path, index=False)