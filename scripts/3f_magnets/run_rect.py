# export ANSYSEM_ROOT241=/data/AnsysEM/v241/Linux64

import os

import numpy as np
import pandas as pd

from machine_design.config import load_config
from machine_design.designs import load_design
from machine_design.optimization import (
    RectMagnetGenerator,
    analyze_results,
    plot_barriers,
    save_params,
)
from motors.synrm_3f_36s_mag import Computation, Geometry

config = load_config()
aedt_version = config["aedt_version"]
num_cores = config["num_cores"]
n_designs = 1
r_stator_end = 0.7
offset = 0.7 / 2
current_setpoint = (1.5, 1.5)
plot_design = True

embrace = 0.8
thick_mag = 4.0
bridge = 1.0
template_params = [embrace, thick_mag, bridge]

fill_length = 1.0
fill_thickness = 1.0
random_magnet = False
fill_bounds = (0.5, 1.0)

project_name = "RectMagnet_test"
design_name = "Design01"
path_data = os.path.join(os.getcwd(), "data")
path_results = "results"
for path in [path_data, path_results]:
    os.makedirs(path, exist_ok=True)
file_name_aedt = f"{path_data}/{project_name}.aedt"

geometry = Geometry()
computation = Computation(geometry)
design = load_design(file_name_aedt, project_name, design_name, aedt_version, geometry, computation)
generators = [RectMagnetGenerator(design, r_stator_end, offset=offset, center=template_params, spread=0.1)]

metadata = pd.DataFrame()
for i in range(0, n_designs):
    for generator in generators:
        params = np.array(template_params)
        generator.set_parameters(params)
        barriers = generator.generate_barriers()
        if not generator.feasible_barriers(barriers):
            raise ValueError(f"Barrier is not feasible for parameters {params}")

        fills = np.random.uniform(*fill_bounds, 2) if random_magnet else np.array([fill_length, fill_thickness])
        magnets = generator.generate_magnets(barriers, fill_length=fills[0], fill_thickness=fills[1])

        # Generate the geometry
        design.add_rotor()
        for barrier, magnet in zip(barriers, magnets):
            design.add_rotor_barrier(barrier)
            design.add_rotor_magnet(magnet)

        # Compute the torque
        outputs = design.compute(*current_setpoint, NUM_CORES=num_cores)
        means = {k: np.nan for k in ("Flux_d", "Flux_q", "Ui_d", "Ui_q")}
        if outputs is None:
            TorAvg, TorRippleRms = np.nan, np.nan
        else:
            TorAvg, _, TorRippleRms = analyze_results(outputs["Moving1.Torque"])
            means = {k: float(np.mean(outputs[k])) for k in means}

        if plot_design:
            design.m2d.plot(show=False, output_file=os.path.abspath(f"{path_results}/design_{generator.name}_{i}_ansys.png"), view="xy", plot_air_objects=False)

        # Delete the rotor and the magnets
        design.delete_rotor()
        modeler = design.m2d.modeler
        magnet_names = [name for name in modeler.object_names if name.startswith("Magnet")]
        if magnet_names:
            modeler.delete(magnet_names)
        for cs in [cs for cs in modeler.coordinate_systems if cs.name.startswith("CS_Magnet")]:
            cs.delete()

        # Potentially save the design
        if plot_design:
            title = f"Torque mean value: {np.round(TorAvg, 2)} Nm, ripple relative value: {np.round(TorRippleRms, 2)} %"
            file_name = f"{path_results}/design_{generator.name}_{i}"
            plot_barriers(barriers, design, title=title, file_name=f"{file_name}.png", magnets=magnets)
            save_params({"barrier": params, "fills": fills}, f"{file_name}.pkl")

        metadata_new = {
            "method": generator.name,
            "design": i,
            "T": TorAvg,
            "ripple": TorRippleRms,
            **means,
        }
        metadata = pd.concat((metadata, pd.DataFrame([metadata_new])), ignore_index=True)
        metadata.to_csv(f"{path_results}/metadata_rect.csv", index=False)

design.close_project()
