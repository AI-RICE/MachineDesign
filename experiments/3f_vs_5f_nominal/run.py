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
n_designs = 20
r_stator_end = 0.7
offset = 0.7 / 2
plot_design = True
current_scale = 0.584  # according to PR79
i_max_3f = 2.0
i_max_5f = current_scale * i_max_3f
current_half_bounds_3f = i_max_3f  # case 1, 3f
current_half_bounds_5f_i3zero = i_max_5f  # case 2, 5f, id3=iq3=0
current_half_bounds_5f = (4 / np.pi) * i_max_5f  # case 3, 5f, free id3 and iq3
v_max = 311.13

for case, module_name, n_phases, current_bounds, i_max, v_max in [  # i_max and v_max are set temperally to check the code can work.
    (1, "motors.synrm_3f_60s", 3, np.array([[-current_half_bounds_3f] * 2, [current_half_bounds_3f] * 2]), i_max_3f, v_max),
    (2, "motors.synrm_5f_60s", 5, np.array([[-current_half_bounds_5f_i3zero] * 2, [current_half_bounds_5f_i3zero] * 2]), i_max_5f, v_max),
    (3, "motors.synrm_5f_60s", 5, np.array([[-current_half_bounds_5f] * 4, [current_half_bounds_5f] * 4]), i_max_5f, v_max),
]:
    module = importlib.import_module(module_name)
    Computation, Geometry = module.Computation, module.Geometry
    geometry = Geometry()
    current_n = current_bounds.shape[1]
    harmonics = [1] if n_phases == 3 else [1, 3]
    harmonics_free = harmonics[: current_n // 2]
    case_name = f"{n_phases}f_case{case}"

    project_name = f"SynRM_{n_phases}f_nominal"
    design_name = "Design01"
    path_data = os.path.join(os.getcwd(), "data")
    path_results = f"results/results_{case_name}"
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

        while True:  # currents constraints
            current_setpoint = np.random.uniform(current_bounds[0], current_bounds[1])
            peak_current = current_peak(current_setpoint, harmonics_free)
            if i_max is None or peak_current <= i_max:
                break

        current_setpoint_full = np.zeros(2 * len(harmonics))
        current_setpoint_full[: len(current_setpoint)] = current_setpoint

        # Generate the geometry
        design.add_rotor()
        for barrier in barriers:
            design.add_rotor_barrier(barrier)

        # Compute the torque
        out = design.compute(*current_setpoint_full, NUM_CORES=num_cores)
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
