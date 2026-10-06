"""Anchor `synrm_3f_36s`: 3-phase synchronous reluctance machine, 36 stator slots."""

import numpy as np
from ansys.aedt.core import Maxwell2d

from motors.synrm_3f_36s import Computation as Computation3f36s
from motors.synrm_3f_36s import Geometry as Geometry3f36s

__all__ = ["Geometry", "Computation"]


class Geometry(Geometry3f36s):
    def build_stator(self, m2d: Maxwell2d) -> None:
        super().build_stator(m2d)
        self._create_material(m2d, self.magnet, **self.magnet_props)


class Computation(Computation3f36s):
    def set_oper_params(self) -> None:
        f = 50  # [Hz]
        RotSpeed = 60 * f / self.geometry.PolePairs  # [rpm]
        w = 2 * np.pi * f
        self.oper_params = {
            "Im": "1.5*sqrt(2)A",
            "epsI": "pi/4",  # current angle
            "InitPos": "-30deg",
            "w": f"{w}Hz",
            "RotSpeed": f"{RotSpeed}rpm",
            "Nper": "1/6",  # number of included periods
            "PointPer": "101",  # number of time points per period
        }

    def set_solution_expressions(self) -> None:
        self.solution_expressions = "Moving1.Torque"

    def set_output_vars(self) -> None:
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

    def set_post_params(self) -> None:
        super().set_post_params()
        self.post_params.update(
            {
                ("I_d", "I_q"): "Current_dq",
                ("Flux_d", "Flux_q"): "FluxLinkage_dq",
                ("Ui_d", "Ui_q"): "InducedVoltage_dq",
                ("L_d", "L_q"): "Inductance_dq",
            }
        )

    def set_variables(self, m2d: Maxwell2d, *args) -> None:
        pass

    def extract_results(self, solutions):
        return solutions.data_magnitude()
