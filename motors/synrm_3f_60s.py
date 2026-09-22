"""Anchor `synrm_3f_60s`: 3-phase synchronous reluctance machine, 60 stator slots."""

import numpy as np
from ansys.aedt.core import Maxwell2d

from machine_design.designs.computation import ComputationBase
from machine_design.winding import phase_groups

from .synrm_3f_36s import Geometry as BaseGeometry


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


class Computation(ComputationBase):
    def set_oper_params(self):
        f = 50  # [Hz]
        RotSpeed = 60 * f / self.geometry.PolePairs  # [rpm]
        w = 2 * np.pi * f
        self.oper_params = {
            "Id": "0.0A",
            "Iq": "0.0A",
            "epsI": "atan2(Iq,Id)",  # current angle
            "Im": "sqrt(Id^2+Iq^2)",
            "InitPos": "-30deg",
            "w": f"{w}Hz",
            "RotSpeed": f"{RotSpeed}rpm",
            "Nper": "1/6",  # number of included periods
            "PointPer": "101",  # number of time points per period
        }

    def set_solution_expressions(self):
        self.solution_expressions = ["Moving1.Torque"]

    def set_output_vars(self):
        self.output_vars = {}

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
        }

    def assign_stator_coils(self, m2d: Maxwell2d) -> None:
        # Excitations
        I_A = "Im * cos(w*Time+epsI)"
        I_B = "Im * cos(w*Time-120deg+epsI)"
        I_C = "Im * cos(w*Time-240deg+epsI)"

        Q = int(self.geometry.geom_params["SlotNumber"])
        p = self.geometry.PolePairs
        # belt_offset=0 reproduces the legacy 3-phase(36-slot) base winding.
        groups = phase_groups(Q, p, 3, belt_offset=0)

        for group in groups:
            for coil_name, polarity in group:
                m2d.assign_coil(assignment=[coil_name], conductors_number="Nc", polarity=polarity, name=f"CS_{coil_name}")

        for phase_name, current, group in zip("ABC", [I_A, I_B, I_C], groups):
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
        m2d.change_inductance_computation(compute_transient_inductance=True, incremental_matrix=False)

    def set_variables(self, m2d: Maxwell2d, Id, Iq):
        self.Id, self.Iq = Id, Iq
        m2d.variable_manager["Id"] = f"{Id}A"
        m2d.variable_manager["Iq"] = f"{Iq}A"

    def extract_results(self, solutions):
        return {"Moving1.Torque": np.array(solutions.data_real("Moving1.Torque"))}
