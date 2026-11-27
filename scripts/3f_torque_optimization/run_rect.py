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
from motors.template_3f_36s import Geometry as TemplateGeometry

config = load_config()
aedt_version = config["aedt_version"]
num_cores = config["num_cores"]
n_designs = 10
r_stator_end = 0.7
offset = 0.7 / 2
current_setpoint = (1.5, 1.5)
plot_design = True
template_rotor = TemplateGeometry().rotor_params
template_params = [float(template_rotor["Embrace"]), float(template_rotor["ThickMag"][:-2]), float(template_rotor["Bridge"][:-2])]
exact_template = True

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
        # Generate a feasible design
        while True:
            params = np.array(template_params) if exact_template and template_params is not None else generator.random_parameters()
            generator.set_parameters(params)
            barriers = generator.generate_barriers()
            # barriers = generator.split_barriers(barriers)
            feasible = generator.feasible_barriers(barriers)
            if feasible:
                break

        magnets = generator.generate_magnets(barriers)

        # Generate the geometry
        design.add_rotor()
        for barrier, magnet in zip(barriers, magnets):
            design.add_rotor_barrier(barrier)
            design.geometry.add_rotor_magnet(design.m2d, magnet, fill_height=1.0, fill_width=1.0)

        # Compute the torque
        outputs = design.compute(*current_setpoint, NUM_CORES=num_cores)
        means = {k: np.nan for k in ("Flux_d", "Flux_q", "Ui_d", "Ui_q")}
        if outputs is None:
            TorAvg, TorRippleRms = np.nan, np.nan
        else:
            TorAvg, _, TorRippleRms = analyze_results(outputs["Moving1.Torque"])
            means = {k: float(np.mean(outputs[k])) for k in means}

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
            save_params(params, f"{file_name}.pkl")

        metadata_new = {
            "method": generator.name,
            "design": i,
            "Embrace": params[0],
            "ThickMag": params[1],
            "Bridge": params[2],
            "y_min": generator.y_min,
            "thickness": generator.thickness,
            "width": generator.width,
            "T": TorAvg,
            "ripple": TorRippleRms,
            **means,
        }
        metadata = pd.concat((metadata, pd.DataFrame([metadata_new])), ignore_index=True)
        metadata.to_csv(f"{path_results}/metadata_rect.csv", index=False)

design.close_project()
