from abc import ABC, abstractmethod

import numpy as np
from ansys.aedt.core import Maxwell2d

from machine_design.generic.winding import assign_slot_currents_to_aedt, compute_winding_layout

from .geometry import GeometryBase
from .reserved_variables import check_no_reserved_variable_names


class ComputationBase(ABC):
    def __init__(self, geometry: GeometryBase) -> None:
        self.geometry = geometry
        self.phases = "ABCDEFGHIJK"[: self.geometry.n_phases]
        self.setup_name = "Setup1"
        self.rotor_mesh = None
        self.set_oper_params()
        self.set_solution_expressions()
        self.set_output_vars()
        self.set_post_params()
        check_no_reserved_variable_names(self.oper_params, self.output_vars)

    @abstractmethod
    def set_oper_params(self): ...

    @abstractmethod
    def assign_stator_coils(self, m2d: Maxwell2d) -> None: ...

    @abstractmethod
    def inductance_computation(self, m2d: Maxwell2d) -> None: ...

    @abstractmethod
    def set_variables(self, m2d: Maxwell2d, *args): ...

    @abstractmethod
    def extract_results(self, solutions): ...

    @staticmethod
    def extract_expression(solutions, expr: str) -> np.ndarray:
        return np.asarray(solutions.data_real(expr, convert_to_SI=True))

    def set_initpos(self) -> None:
        g = self.geometry
        Q = int(g.geom_params["SlotNumber"])
        layout = compute_winding_layout(g.n_phases, Q, g.PolePairs, coil_pitch=int(g.wind_params["CoilPitch"]), layers=int(g.wind_params["Layers"]))
        self.InitPos = (layout.phase_a_axis - 180) / g.PolePairs
        self.oper_params["InitPos"] = f"{self.InitPos}deg"

    def set_solution_expressions(self):
        self.solution_expressions = [
            "Moving1.Position",
            "Moving1.Torque",
            *[f"FluxLinkage(Phase{p})" for p in self.phases],
            *[f"InducedVoltage(Phase{p})" for p in self.phases],
            *[f"InputCurrent(Phase{p})" for p in self.phases],
            *[f"L(Phase{x},Phase{y})" for x in self.phases for y in self.phases],
        ]

    def set_output_vars(self):
        self.output_vars = {}

    def set_post_params(self):
        self.post_params = {
            tuple(f"InducedVoltage(Phase{p})" for p in self.phases): "InducedVoltage",
            ("Moving1.Torque"): "Torque",
            tuple(f"InputCurrent(Phase{p})" for p in self.phases): "Current",
            tuple(f"FluxLinkage(Phase{p})" for p in self.phases): "FluxLinkage",
        }

    def assign_phase_windings(self, m2d: Maxwell2d, phase_currents: list) -> None:
        g = self.geometry
        Q = int(g.geom_params["SlotNumber"])
        n = Q // (2 * g.PolePairs)
        layers = int(g.wind_params["Layers"])
        layout = compute_winding_layout(g.n_phases, Q, g.PolePairs, coil_pitch=int(g.wind_params["CoilPitch"]), layers=layers)
        layer_coils = [[f"Coil{i_layer}" if k == 0 else f"Coil{i_layer}_{k}" for k in range(n)] for i_layer in range(layers)]
        coil_objects = layer_coils[0]
        coil_objects_back = layer_coils[1] if layers == 2 else None
        phase_currents_dict = dict(zip(self.phases, phase_currents))
        assign_slot_currents_to_aedt(m2d, coil_objects, layout, phase_currents_dict, coil_objects_back=coil_objects_back)

    def extract_phase_results(self, solutions):
        # SI units
        flux_phases = np.stack([self.extract_expression(solutions, f"FluxLinkage(Phase{p})") for p in self.phases], axis=-1)
        vind_phases = np.stack([self.extract_expression(solutions, f"InducedVoltage(Phase{p})") for p in self.phases], axis=-1)
        current_phases = np.stack([self.extract_expression(solutions, f"InputCurrent(Phase{p})") for p in self.phases], axis=-1)
        L_raw = [np.stack([self.extract_expression(solutions, f"L(Phase{x},Phase{y})") for y in self.phases], axis=-1) for x in self.phases]
        return flux_phases, vind_phases, current_phases, L_raw

    def push_variables(self, m2d: Maxwell2d) -> None:
        for k, v in self.oper_params.items():
            m2d[k] = v

    def create_setup(self, m2d: Maxwell2d) -> None:
        self.assign_stator_coils(m2d)
        self.inductance_computation(m2d)

        # model depth
        m2d.model_depth = "StackLength"
        # symmetry
        m2d.change_symmetry_multiplier("SymmetryFactor")
        # Calculation setup
        setup = m2d.create_setup(name=self.setup_name)
        setup.props["StopTime"] = "2*pi*Nper/w"
        setup.props["TimeStep"] = "2*pi/(w*(PointPer-1))"
        setup.props["SaveFieldsType"] = "None"
        setup.props["OutputPerObjectCoreLoss"] = False
        setup.props["OutputPerObjectSolidLoss"] = True
        setup.props["OutputError"] = True
        setup.update()
        m2d.validate_simple()

        for k, v in self.output_vars.items():
            m2d.create_output_variable(k, v)

        for k, v in self.post_params.items():
            expressions = list(k) if isinstance(k, tuple) else [k]  # if multiple report, use list(k). Else, use k
            m2d.post.create_report(
                expressions=expressions,
                setup_sweep_name="",
                domain="Sweep",
                variations=None,
                primary_sweep_variable="Time",
                secondary_sweep_variable=None,
                report_category=None,
                plot_type="Rectangular Plot",
                context=None,
                subdesign_id=None,
                polyline_points=1001,
                plot_name=v,
            )

    # TODO: change the other arguments to kwargs
    def compute(self, m2d: Maxwell2d, rotor_id, *args, NUM_CORES: int = 1):
        assert m2d.mesh is not None
        assert m2d.post is not None

        self.rotor_mesh = m2d.mesh.assign_length_mesh(
            assignment=rotor_id,
            inside_selection=True,
            maximum_length=0.5,
            maximum_elements=None,
            name="rotor",
        )
        # core loss rotor
        m2d.set_core_losses("Rotor", core_loss_on_field=False)

        self.set_variables(m2d, *args)

        # Analyze
        m2d.analyze_setup(self.setup_name, use_auto_settings=False, cores=NUM_CORES)

        solutions = m2d.post.get_solution_data(expressions=self.solution_expressions, primary_sweep_variable="Time")
        try:
            result = self.extract_results(solutions)
        except AttributeError:
            result = None

        # Delete solution data to prevent the saving size to explode
        m2d.odesign.DeleteFullVariation("All", False)

        return result

    def delete_rotor_mesh(self) -> None:
        if self.rotor_mesh is not None:
            self.rotor_mesh.delete()
            self.rotor_mesh = None
