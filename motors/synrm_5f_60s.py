"""Anchor `synrm_5f_60s`: 5-phase synchronous reluctance machine, 60 stator slots.

Derives from `synrm_5f_40s`, overriding only the slotting and winding it needs for five
phases. Excitation is dq1 + dq3.
"""

import numpy as np
from ansys.aedt.core import Maxwell2d

from machine_design.designs.computation import ComputationBase
from machine_design.winding import phase_groups

from .synrm_5f_40s import Geometry as BaseGeometry


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
        self.wind_params["CoilPitch"] = "15"


class Computation(ComputationBase):
    def set_oper_params(self):
        f = 50  # [Hz]
        RotSpeed = 60 * f / self.geometry.PolePairs  # [rpm]
        w = 2 * np.pi * f
        self.oper_params = {
            "Id1": "0.0A",
            "Iq1": "0.0A",
            "Id3": "0.0A",
            "Iq3": "0.0A",
            "epsI1": "atan2(Iq1,Id1)",  # current angle, 1st harmonic
            "epsI3": "atan2(Iq3,Id3)",  # current angle, 3rd harmonic
            "Im1": "sqrt(Id1^2+Iq1^2)",
            "Im3": "sqrt(Id3^2+Iq3^2)",
            "InitPos": "-45deg",
            "w": f"{w}Hz",
            "RotSpeed": f"{RotSpeed}rpm",
            "Nper": "1/10",  # number of included periods
            "PointPer": "101",  # number of time points per period
        }

    def set_solution_expressions(self):
        self.solution_expressions = ["Moving1.Torque"]

    def set_output_vars(self):
        self.output_vars = {}

    def assign_stator_coils(self, m2d: Maxwell2d) -> None:
        # Excitations
        I_A = "Im1*cos(w*Time+epsI1-pi) + Im3*cos(3*(w*Time)+epsI3-pi)"
        I_B = "Im1*cos(w*Time-72deg+epsI1-pi) + Im3*cos(3*(w*Time-72deg)+epsI3-pi)"
        I_C = "Im1*cos(w*Time-144deg+epsI1-pi) + Im3*cos(3*(w*Time-144deg)+epsI3-pi)"
        I_D = "Im1*cos(w*Time-216deg+epsI1-pi) + Im3*cos(3*(w*Time-216deg)+epsI3-pi)"
        I_E = "Im1*cos(w*Time-288deg+epsI1-pi) + Im3*cos(3*(w*Time-288deg)+epsI3-pi)"

        Q = int(self.geometry.geom_params["SlotNumber"])
        p = self.geometry.PolePairs
        # belt_offset=1 reproduces the 5-phase(40-slot) base winding.
        groups = phase_groups(Q, p, 5, belt_offset=1)

        for group in groups:
            for coil_name, polarity in group:
                m2d.assign_coil(assignment=[coil_name], conductors_number="Nc", polarity=polarity, name=f"CS_{coil_name}")

        for phase_name, current, group in zip("ABCDE", [I_A, I_B, I_C, I_D, I_E], groups):
            m2d.assign_winding(
                assignment=None,
                winding_type="Current",
                is_solid=False,
                current=current,
                parallel_branches="ParallelPaths",
                name=f"Phase{phase_name}",
            )
            m2d.add_winding_coils(assignment=f"Phase{phase_name}", coils=[f"CS_{coil_name}" for coil_name, _ in group])

    def inductance_computation(self, m2d: Maxwell2d) -> None:
        m2d.change_inductance_computation(compute_transient_inductance=True, incremental_matrix=True)

    def set_variables(self, m2d: Maxwell2d, Id1, Iq1, Id3, Iq3):
        self.Id1, self.Iq1, self.Id3, self.Iq3 = Id1, Iq1, Id3, Iq3
        m2d.variable_manager["Id1"] = f"{Id1}A"
        m2d.variable_manager["Iq1"] = f"{Iq1}A"
        m2d.variable_manager["Id3"] = f"{Id3}A"
        m2d.variable_manager["Iq3"] = f"{Iq3}A"

    def extract_results(self, solutions):
        return {"Moving1.Torque": np.array(solutions.data_real("Moving1.Torque"))}
