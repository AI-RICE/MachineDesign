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
seeds = (42, 43)
num_cores = 4
n_repeats = 5
results_csv = "compare_precompute_results.csv"

motors = {
    "motor1": (Computation1, Geometry1, ()),
    "motor2": (Computation2, Geometry2, (7.0711, 7.0711, 0.0, 0.0)),
}

baselines = pd.read_csv("precompute_baselines.csv")


def make_barriers(design, seed):
    generator = HacklGenerator_OneLambda(design, r_stator_end, offset=offset)
    np.random.seed(seed)
    while True:
        params = generator.random_parameters()
        generator.set_parameters(params)
        barriers = generator.generate_barriers()
        barriers = generator.split_barriers(barriers)
        if generator.feasible_barriers(barriers):
            return barriers


def extract(result):
    return np.asarray(result["Moving1.Torque"] if isinstance(result, dict) else result, dtype=float)


rows = []
for motor_name, (Computation, Geometry, current_setpoint) in motors.items():
    for seed in seeds:
        row = baselines[(baselines["motor"] == motor_name) & (baselines["seed"] == seed)].iloc[0]
        pt_cols = [c for c in baselines.columns if c.startswith("pt")]
        baseline = row[pt_cols].dropna().to_numpy(dtype=float)

        for i in range(n_repeats):
            geometry = Geometry()
            computation = Computation(geometry)
            design = load_design(
                os.path.join(os.getcwd(), f"compare_{motor_name}_seed{seed}_run{i}.aedt"),
                f"Compare{motor_name.capitalize()}Seed{seed}Run{i}",
                "Design01",
                aedt_version,
                geometry,
                computation,
                non_graphical=True,
                new_desktop=True,
                close_on_exit=False,
            )
            try:
                barriers = make_barriers(design, seed)
                design.add_rotor()
                for barrier in barriers:
                    design.add_rotor_barrier(barrier)
                result = design.compute(*current_setpoint, NUM_CORES=num_cores)
                design.delete_rotor()
                live = extract(result)
            finally:
                design.close_project()

            rel_err = np.abs(live - baseline) / np.abs(baseline) * 100
            max_err = np.nanmax(rel_err)
            rows.append({"motor": motor_name, "seed": seed, "run": i + 1, "max_rel_err_pct": max_err})
            pd.DataFrame(rows).to_csv(results_csv, index=False)
            print(f"{motor_name} seed={seed} run {i + 1}/{n_repeats}: max_rel_err={max_err}%")

print(f"saved results to {results_csv}")
