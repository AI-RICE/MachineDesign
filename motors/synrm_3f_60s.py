"""Anchor `synrm_3f_60s`: 3-phase synchronous reluctance machine, 60 stator slots."""

from .synrm_3f_36s import Computation
from .synrm_3f_36s import Geometry as BaseGeometry

__all__ = ["Computation", "Geometry"]


class Geometry(BaseGeometry):
    def set_geom_params(self):
        super().set_geom_params()
        self.geom_params["SlotNumber"] = "60"

    def set_slot_params(self):
        super().set_slot_params()
        self.slot_params["Bs0"] = "1.5mm"
        self.slot_params["Bs1"] = "2.0mm"
        self.slot_params["Bs2"] = "2.85mm"
        self.slot_params["Rs"] = "1.0mm"
        self.slot_params["SetAngle"] = "6deg"

    def set_winds_params(self):
        super().set_winds_params()
        self.wind_params["CoilPitch"] = "15"  # full pitch over one pole (60/4)
