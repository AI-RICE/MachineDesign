from abc import ABC, abstractmethod

from ansys.aedt.core import Maxwell2d

from .geometry import GeometryBase

# Ansys intrinsic/reserved variable names
reserved_variable_names = frozenset(
    name.lower()
    for name in (
        "_Empty",
        "_I1",
        "_I2",
        "_I3",
        "_I4",
        "_I5",
        "_I6",
        "_I7",
        "_I8",
        "_I9",
        "_t",
        "_u",
        "_v",
        "_V1",
        "_V2",
        "_V3",
        "_V4",
        "_V5",
        "_V6",
        "_V7",
        "_V8",
        "_V9",
        "Ang",
        "Budget_Index",
        "Distance",
        "ElectricalDegree",
        "F",
        "F1",
        "F2",
        "F3",
        "FNoi",
        "Freq",
        "Hmax",
        "Hmin",
        "Ia",
        "Ib",
        "Index",
        "IWavePhi",
        "IWaveTheta",
        "Normalized Deformation",
        "Normalized Distance",
        "OP",
        "Pass",
        "Phase",
        "Phi",
        "Position",
        "R",
        "Rho",
        "RSpeed",
        "Spectrum",
        "Speed",
        "Temp",
        "Tend",
        "Theta",
        "Time",
        "Time0",
        "Vac",
        "Vbe",
        "Vce",
        "Vds",
        "Vgs",
        "X",
        "Y",
        "Z",
        "ZAng",
        "ZRho",
    )
)


class ComputationBase(ABC):
    def __init__(self, geometry: GeometryBase) -> None:
        self.geometry = geometry
        self.setup_name = "Setup1"
        self.set_oper_params()
        self.set_solution_expressions()
        self.set_output_vars()
        self.set_post_params()
        self._check_no_reserved_variable_names()

    def _check_no_reserved_variable_names(self) -> None:
        for name in list(self.oper_params) + list(self.output_vars):
            if name.lower() in reserved_variable_names:
                raise ValueError(f"'{name}' is an Ansys intrinsic variable name and cannot be used as a design variable")

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
