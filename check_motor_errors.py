import os

import numpy as np
import pandas as pd

from machine_design.config import load_config
from machine_design.designs import load_design
from machine_design.optimization.generators import HacklGenerator_OneLambda
from motors.synrm_3f_36s import Computation as Computation1
from motors.synrm_3f_36s import Geometry as Geometry1
from motors.synrm_5f_40s import Computation as Computation2
from motors.synrm_5f_40s import Geometry as Geometry2

aedt_version = load_config()["aedt_version"]
r_stator_end = 0.7
offset = 0.35
seed = 0
num_cores = 4
n_repeats = 30
results_csv = "motor_errors_results.csv"

motors = {
    "motor1": (Computation1, Geometry1, (), os.path.join("tests", "golden", "motor1_baseline", "tor.csv")),
    "motor2": (Computation2, Geometry2, (7.0711, 7.0711, 0.0, 0.0), os.path.join("tests", "golden", "motor2_baseline", "tor.csv")),
}

rows = []
for motor_name, (Computation, Geometry, current_setpoint, golden_path) in motors.items():
    golden_tor = np.loadtxt(golden_path, delimiter=",", skiprows=1)

    geometry = Geometry()
    computation = Computation(geometry)
    design = load_design(
        os.path.join(os.getcwd(), f"{motor_name}_errors_check.aedt"),
        f"{motor_name.capitalize()}ErrorsCheck",
        "Design01",
        aedt_version,
        geometry,
        computation,
        non_graphical=True,
        new_desktop=True,
        close_on_exit=False,
    )

    try:
        design.m2d["Nper"] = "1"

        np.random.seed(seed)
        generator = HacklGenerator_OneLambda(design, r_stator_end, offset=offset)
        while True:
            params = generator.random_parameters()
            generator.set_parameters(params)
            barriers = generator.generate_barriers()
            barriers = generator.split_barriers(barriers)
            if generator.feasible_barriers(barriers):
                break

        for i in range(n_repeats):
            design.add_rotor()
            for barrier in barriers:
                design.add_rotor_barrier(barrier)

            result = design.compute(*current_setpoint, NUM_CORES=num_cores)
            design.delete_rotor()

            Tor = result["Moving1.Torque"] if isinstance(result, dict) else result
            Tor = np.asarray(Tor, dtype=float)

            rel_err = np.abs(Tor - golden_tor) / np.abs(golden_tor) * 100
            row = {"motor": motor_name, "run": i + 1, "max_rel_err_pct": rel_err.max(), "mean_rel_err_pct": rel_err.mean()}
            for j, e in enumerate(rel_err):
                row[f"pt{j}"] = e
            rows.append(row)

            pd.DataFrame(rows).to_csv(results_csv, index=False)
            print(f"{motor_name} run {i + 1}/{n_repeats}: max={rel_err.max():.2f}%, mean={rel_err.mean():.2f}%")
    finally:
        design.close_project()

print(f"saved per-run, per-point errors to {results_csv}")
