from abc import ABC, abstractmethod

from ansys.aedt.core import Maxwell2d

from .design_geometry import GeometryBase


class ComputationBase(ABC):
    def __init__(self, geometry: GeometryBase) -> None:
        self.geometry = geometry
        self.setup_name = "Setup1"
        self.set_oper_params()
        self.set_solution_expressions()
        self.set_output_vars()
        self.set_post_params()

    @abstractmethod
    def set_oper_params(self): ...

    @abstractmethod
    def set_solution_expressions(self): ...

    @abstractmethod
    def set_output_vars(self): ...

    @abstractmethod
    def set_post_params(self): ...

    @abstractmethod
    def assign_stator_coils(self, m2d: Maxwell2d) -> None: ...

    @abstractmethod
    def inductance_computation(self, m2d: Maxwell2d) -> None: ...

    @abstractmethod
    def set_variables(self, m2d: Maxwell2d, *args): ...

    @abstractmethod
    def extract_results(self, solutions): ...

    def assign_motion(self, m2d: Maxwell2d, band_name: str = "Band") -> None:
        m2d.assign_rotate_motion(
            assignment=band_name,
            coordinate_system="Global",
            axis="Z",
            positive_movement=True,
            start_position="InitPos",
            angular_velocity="RotSpeed",
            has_rotation_limits=False,
        )

    def create_setup(self, m2d: Maxwell2d) -> None:
        for k, v in self.oper_params.items():
            m2d[k] = v

        self.assign_stator_coils(m2d)
        self.inductance_computation(m2d)

        # model depth
        m2d.model_depth = "StackLength"
        # symmetry
        m2d.change_symmetry_multiplier("SymmetryFactor")
        # Calculation setup
        setup = m2d.create_setup(name=self.setup_name)
        setup.props["StopTime"] = "Nper/f"
        setup.props["TimeStep"] = "1/(f*(PointPer-1))"
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

        m2d.mesh.assign_length_mesh(
            assignment=rotor_id,
            inside_selection=True,
            maximum_length=3,
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


class Computation(ComputationBase):
    def set_oper_params(self):
        f = 50  # [Hz]
        RotSpeed = 60 * f / self.geometry.PolePairs  # [rpm]
        self.oper_params = {
            "Im": "1.5*sqrt(2)A",
            "epsI": "pi/4",  # current angle
            "InitPos": "-30deg",
            "f": f"{f}Hz",
            "RotSpeed": f"{RotSpeed}rpm",
            "Nper": "1/6",  # number of included periods
            "PointPer": "101",  # number of time points per period
        }

    def set_solution_expressions(self):
        self.solution_expressions = "Moving1.Torque"

    def set_output_vars(self):
        self.output_vars = {
            "pos": "(Moving1.Position -InitPos) * Poles/2",
            "cos0": "cos(pos)",
            "cos1": "cos(pos-2*PI/3)",
            "cos2": "cos(pos-4*PI/3)",
            "sin0": "sin(pos)",
            "sin1": "sin(pos-2*PI/3)",
            "sin2": "sin(pos-4*PI/3)",
            "Lad": "L(PhaseA,PhaseA)*cos0 + L(PhaseA,PhaseB)*cos1 + L(PhaseA,PhaseC)*cos2",
            "Laq": "L(PhaseA,PhaseA)*sin0 + L(PhaseA,PhaseB)*sin1 + L(PhaseA,PhaseC)*sin2",
            "Lbd": "L(PhaseB,PhaseA)*cos0 + L(PhaseB,PhaseB)*cos1 + L(PhaseB,PhaseC)*cos2",
            "Lbq": "L(PhaseB,PhaseA)*sin0 + L(PhaseB,PhaseB)*sin1 + L(PhaseB,PhaseC)*sin2",
            "Lcd": "L(PhaseC,PhaseA)*cos0 + L(PhaseC,PhaseB)*cos1 + L(PhaseC,PhaseC)*cos2",
            "Lcq": "L(PhaseC,PhaseA)*sin0 + L(PhaseC,PhaseB)*sin1 + L(PhaseC,PhaseC)*sin2",
            "L_d": "(Lad*cos0 + Lbd*cos1 + Lcd*cos2) * 2/3",
            "L_q": "(Laq*sin0 + Lbq*sin1 + Lcq*sin2) * 2/3",
            "Flux_d": "(FluxLinkage(PhaseA)*cos0+FluxLinkage(PhaseB)*cos1+FluxLinkage(PhaseC)*cos2)*2/3",
            "Flux_q": "-(FluxLinkage(PhaseA)*sin0+FluxLinkage(PhaseB)*sin1+FluxLinkage(PhaseC)*sin2)*2/3",
            "Ui_d": "(InducedVoltage(PhaseA)*cos0+InducedVoltage(PhaseB)*cos1+InducedVoltage(PhaseC)*cos2)*2/3",
            "Ui_q": "-(InducedVoltage(PhaseA)*sin0+InducedVoltage(PhaseB)*sin1+InducedVoltage(PhaseC)*sin2)*2/3",
            "I_d": "(InputCurrent(PhaseA)*cos0 + InputCurrent(PhaseB)*cos1 + InputCurrent(PhaseC)*cos2)*2/3",
            "I_q": "-(InputCurrent(PhaseA)*sin0 + InputCurrent(PhaseB)*sin1 + InputCurrent(PhaseC)*sin2)*2/3",
            "Irms": "sqrt(I_d^2+I_q^2)/sqrt(2)",
        }

    def set_post_params(self):
        self.post_params = {  # reports
            ("InducedVoltage(PhaseA)", "InducedVoltage(PhaseB)", "InducedVoltage(PhaseC)"): "InducedVoltage",
            ("Moving1.Torque"): "Torque",
            ("InputCurrent(PhaseA)", "InputCurrent(PhaseB)", "InputCurrent(PhaseC)"): "Current",
            (
                "FluxLinkage(PhaseA)",
                "FluxLinkage(PhaseB)",
                "FluxLinkage(PhaseC)",
            ): "FluxLinkage",
            ("I_d", "I_q"): "Current_dq",
            ("Flux_d", "Flux_q"): "FluxLinkage_dq",
            ("Ui_d", "Ui_q"): "InducedVoltage_dq",
            ("L_d", "L_q"): "Inductance_dq",
        }

    def assign_stator_coils(self, m2d: Maxwell2d) -> None:
        # Excitations
        I_A = "Im * cos(2*pi*f*time+epsI)"
        I_B = "Im * cos(2*pi*f*time-120deg+epsI)"
        I_C = "Im * cos(2*pi*f*time-240deg+epsI)"

        # Define phase windings
        m2d.assign_coil(
            assignment=["Coil"],
            conductors_number="Nc",
            polarity="Positive",
            name="CS1",
        )
        m2d.assign_coil(
            assignment=["Coil_1"],
            conductors_number="Nc",
            polarity="Positive",
            name="CS2",
        )
        m2d.assign_coil(
            assignment=["Coil_2"],
            conductors_number="Nc",
            polarity="Positive",
            name="CS3",
        )
        m2d.assign_winding(
            assignment=None,
            winding_type="Current",
            is_solid=False,
            current=I_A,
            parallel_branches="ParallelPaths",
            name="PhaseA",
        )
        m2d.add_winding_coils(assignment="PhaseA", coils=["CS1", "CS2", "CS3"])
        m2d.assign_coil(
            assignment=["Coil_6"],
            conductors_number="Nc",
            polarity="Positive",
            name="CS7",
        )
        m2d.assign_coil(
            assignment=["Coil_7"],
            conductors_number="Nc",
            polarity="Positive",
            name="CS8",
        )
        m2d.assign_coil(
            assignment=["Coil_8"],
            conductors_number="Nc",
            polarity="Positive",
            name="CS9",
        )
        m2d.assign_winding(
            assignment=None,
            winding_type="Current",
            is_solid=False,
            current=I_B,
            parallel_branches="ParallelPaths",
            name="PhaseB",
        )
        m2d.add_winding_coils(assignment="PhaseB", coils=["CS7", "CS8", "CS9"])
        m2d.assign_coil(
            assignment=["Coil_3"],
            conductors_number="Nc",
            polarity="Negative",
            name="CS4",
        )
        m2d.assign_coil(
            assignment=["Coil_4"],
            conductors_number="Nc",
            polarity="Negative",
            name="CS5",
        )
        m2d.assign_coil(
            assignment=["Coil_5"],
            conductors_number="Nc",
            polarity="Negative",
            name="CS6",
        )
        m2d.assign_winding(
            assignment=None,
            winding_type="Current",
            is_solid=False,
            current=I_C,
            parallel_branches="ParallelPaths",
            name="PhaseC",
        )
        m2d.add_winding_coils(assignment="PhaseC", coils=["CS4", "CS5", "CS6"])

    def inductance_computation(self, m2d: Maxwell2d) -> None:
        m2d.change_inductance_computation(compute_transient_inductance=True, incremental_matrix=False)

    def set_variables(self, m2d: Maxwell2d, *args):
        pass

    def extract_results(self, solutions):
        return solutions.data_magnitude()
