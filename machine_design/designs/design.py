from ansys.aedt.core import Desktop, Maxwell2d

from .computation import ComputationBase
from .geometry import GeometryBase


class Design:
    def __init__(self, m2d: Maxwell2d, geometry: GeometryBase, computation: ComputationBase) -> None:
        assert computation.geometry is geometry
        self.m2d = m2d
        self.geometry = geometry
        self.computation = computation

    @classmethod
    def create(cls, project_name: str, design_name: str, file_name: str, geometry: GeometryBase, computation: ComputationBase, **kwargs) -> "Design":
        m2d = Maxwell2d(project=project_name, design=design_name, solution_type="TransientXY", **kwargs)
        obj = cls(m2d, geometry, computation)
        obj.add_stator()
        obj.save_project(file_name)
        return obj

    @classmethod
    def load(cls, file_name: str, geometry: GeometryBase, computation: ComputationBase, **kwargs) -> "Design":
        desktop = Desktop(**kwargs)
        desktop.load_project(file_name)

        m2d = Maxwell2d()
        m2d.set_active_design("Design01")

        return cls(m2d, geometry, computation)

    def add_stator(self) -> None:
        self.computation.push_variables(self.m2d)
        self.geometry.build_stator(self.m2d)
        self.computation.create_setup(self.m2d)

    def add_rotor(self, barriers=None, magnets=None, segment_type=None) -> None:
        self.geometry.build_rotor(self.m2d)
        if barriers is not None:
            for barrier in barriers:
                self.geometry.add_rotor_barrier(self.m2d, barrier, segment_type)
        if magnets is not None:
            for magnet in magnets:
                material = self.geometry.create_pm_material(self.m2d, "Magnet")
                self.geometry.add_rotor_magnet(self.m2d, magnet, material)
        self.geometry.delete_motion_setup(self.m2d)
        self.geometry.assign_motion_setup(self.m2d)

    def delete_rotor(self) -> None:
        self.geometry.delete_rotor(self.m2d)

    def compute(self, *args, NUM_CORES: int = 1):
        return self.computation.compute(self.m2d, self.geometry.rotor_id, *args, NUM_CORES=NUM_CORES)

    def save_design(self, file_name: str, **kwargs) -> None:
        show = kwargs.pop("show", False)
        view = kwargs.pop("view", "xy")
        self.m2d.plot(show=show, output_file=file_name, view=view)

    def save_project(self, file_name: str | None = None) -> None:
        if file_name is None:
            self.m2d.save_project()
        else:
            self.m2d.save_project(file_name)

    def close_project(self) -> None:
        self.m2d.close_desktop()
