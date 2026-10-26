# export ANSYSEM_ROOT241=/data/AnsysEM/v241/Linux64

import os

import importlib

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

from optimization import current_peak

config = load_config()
aedt_version = config["aedt_version"]
num_cores = config["num_cores"]
n_designs = 10
r_stator_end = 0.7
offset = 0.7 / 2
plot_design = True

for module_name, n_phases, current_bounds, i_max, v_max in [  # i_max and v_max are set temperally to check the code can work.
    ("motors.synrm_3f_60s", 3, np.array([[0.0, 0.0], [2.5, 2.5]]), 3.0, 311.13),
    ("motors.synrm_5f_60s", 5, np.array([[0.0, 0.0, -1.5, -1.5], [1.5, 1.5, 1.5, 1.5]]), 3.0, 311.13),
]:
    module = importlib.import_module(module_name)
    Computation, Geometry = module.Computation, module.Geometry
    geometry = Geometry()
    current_n = current_bounds.shape[1]
    harmonics = [1] if current_n == 2 else [1, 3]

    project_name = f"SynRM_{n_phases}f_nominal"
    design_name = "Design01"
    path_data = os.path.join(os.getcwd(), "data")
    path_results = f"results/results_{n_phases}f"
    for path in [path_data, path_results]:
        os.makedirs(path, exist_ok=True)
    file_name_aedt = f"{path_data}/{project_name}.aedt"

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

        while True:
            current_setpoint = np.random.uniform(current_bounds[0], current_bounds[1])
            peak_current = current_peak(current_setpoint, harmonics)
            if i_max is None or peak_current <= i_max:
                break

        # Generate the geometry
        design.add_rotor()
        for barrier in barriers:
            design.add_rotor_barrier(barrier)

        # Compute the torque
        out = design.compute(*current_setpoint, NUM_CORES=num_cores)
        # Tor = design.compute(num_cores)
        if out is None:
            TorAvg, TorRippleRms, V_peak = np.nan, np.nan, np.nan
        else:
            TorAvg, _, TorRippleRms = analyze_results(out["Moving1.Torque"])
            if len(harmonics) == 1:
                V_components = np.array([np.mean(out["V_d"]), np.mean(out["V_q"])])
            else:
                V_components = np.array([np.mean(out["V_d1"]), np.mean(out["V_q1"]), np.mean(out["V_d3"]), np.mean(out["V_q3"])])
            V_peak = current_peak(V_components, harmonics)

        # Delete the rotor
        design.delete_rotor()

        loss = float(np.sum(current_setpoint**2))

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
            "voltage": V_peak,
        }
        metadata = pd.concat((metadata, pd.DataFrame([metadata_new])), ignore_index=True)
        metadata.to_csv(f"{path_results}/metadata.csv", index=False)

design.close_project()
