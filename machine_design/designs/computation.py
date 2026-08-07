from abc import ABC, abstractmethod

from ansys.aedt.core import Maxwell2d

from .geometry import GeometryBase


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