"""Anchor `synrm_3f_36s`: 3-phase synchronous reluctance machine, 36 stator slots."""

from ansys.aedt.core import Maxwell2d

from motors.synrm_3f_36s import Computation as Computation3f36s
from motors.synrm_3f_36s import Geometry as Geometry3f36s

__all__ = ["Geometry", "Computation"]


class Geometry(Geometry3f36s):
    def build_stator(self, m2d: Maxwell2d) -> None:
        super().build_stator(m2d)
        self._create_material(m2d, self.magnet, **self.magnet_props)

    def set_magnets(self):
        self.magnet = "NdFeb"
        self.magnet_props = dict(permeability=1.05, conductivity=0, density=7500, coercivity=900000, coercivity_dir=(1.0, 0.0, 0.0))


class Computation(Computation3f36s):
    pass
