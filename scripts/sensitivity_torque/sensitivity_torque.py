import os
import time

import numpy as np
import pandas as pd

from machine_design.config import load_config
from machine_design.designs import load_design
from machine_design.optimization import (
    HacklGenerator_3BrokenLines,
    HacklGenerator_OneLambda,
    HacklGenerator_SixLambdas,
    analyze_results,
)
from motors.synrm_3f_36s import Computation as Computation1
from motors.synrm_3f_36s import Geometry as Geometry1
from motors.synrm_5f_40s import Computation as Computation2
from motors.synrm_5f_40s import Geometry as Geometry2

config = load_config()
aedt_version = config["aedt_version"]
num_cores = config["num_cores"]
design_name = "Design01"
r_stator_end = 0.7
offset = 0.35
seed = 0

path_data = os.path.join(os.getcwd(), "data")
path_results = os.path.join(os.getcwd(), "results")
os.makedirs(path_data, exist_ok=True)
os.makedirs(path_results, exist_ok=True)

point_per_values = list(range(51, 1002, 50))

# X_best fixed per method; None means a feasible design is drawn once with a fixed seed.
motors = {
    "motor1": (
        Computation1,
        Geometry1,
        (),
        {
            "HacklGenerator_OneLambda": (HacklGenerator_OneLambda, np.array([7.1, 20.4, 33.7, 13.7, 27.1, 40.65, 0.35])),
            "HacklGenerator_SixLambdas": (
                HacklGenerator_SixLambdas,
                np.array([8.95052421, 23.01115583, 33.03271713, 14.34330278, 27.92171436, 37.3, 0.21260669, 0.31, 0.41, 0.3, 0.4, 0.46]),
            ),
            "HacklGenerator_3BrokenLines": (
                HacklGenerator_3BrokenLines,
                np.array([6.0538801, 23.5, 35.74582285, 10.6, 28.5920024, 40.7191182, 0.36646249, 22.65142559, 30.7050295, 38.0, 9.49161503, 2.0, 1.5]),
            ),
        },
    ),
    "motor2": (
        Computation2,
        Geometry2,
        (7.0711, 7.0711, 0.0, 0.0),
        {
            "HacklGenerator_OneLambda": (HacklGenerator_OneLambda, None),
            "HacklGenerator_SixLambdas": (HacklGenerator_SixLambdas, None),
            "HacklGenerator_3BrokenLines": (HacklGenerator_3BrokenLines, None),
        },
    ),
}


def get_feasible_barriers(generator, X_best):
    if X_best is not None:
        params = generator.X_to_params(X_best)
        generator.set_parameters(params)
        barriers = generator.generate_barriers()
        return generator.split_barriers(barriers)

    np.random.seed(seed)
    while True:
        params = generator.random_parameters()
        generator.set_parameters(params)
        barriers = generator.generate_barriers()
        barriers = generator.split_barriers(barriers)
        if generator.feasible_barriers(barriers):
            return barriers


metadata = pd.DataFrame()

for motor_name, (Computation, Geometry, current_setpoint, designs) in motors.items():
    for method, (generator_cls, X_best) in designs.items():
        project_name = f"sensitivity_torque_{motor_name}_{method}"
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

        generator = generator_cls(design, r_stator_end, offset=offset)
        barriers = get_feasible_barriers(generator, X_best)

        if not generator.feasible_barriers(barriers):
            raise RuntimeError(f"Selected {motor_name}/{method} design is not feasible.")

        design.add_rotor()
        for barrier in barriers:
            design.add_rotor_barrier(barrier)

        for point_per in point_per_values:
            design.m2d.variable_manager["PointPer"] = str(point_per)

            start_time = time.time()
            result = design.compute(*current_setpoint, NUM_CORES=num_cores)
            elapsed_seconds = time.time() - start_time
            Tor = result["Moving1.Torque"] if isinstance(result, dict) else result

            if Tor is None:
                n_points, TorAvg, TorRippleRms = 0, np.nan, np.nan
            else:
                n_points = len(Tor)
                TorAvg, _, TorRippleRms = analyze_results(Tor)

            metadata_new = {
                "motor": motor_name,
                "method": method,
                "point_per": point_per,
                "n_points": n_points,
                "TorAvg": TorAvg,
                "TorRippleRms": TorRippleRms,
                "elapsed_seconds": elapsed_seconds,
            }
            metadata = pd.concat((metadata, pd.DataFrame([metadata_new])), ignore_index=True)
            metadata.to_csv(os.path.join(path_results, "sensitivity_torque_results.csv"), index=False)

        design.delete_rotor()
        design.close_project()
