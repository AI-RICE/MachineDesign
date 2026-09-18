"""Anchor `synrm_5f_40s`: 5-phase synchronous reluctance machine, 40 stator slots.

Derives from `synrm_3f_36s`, overriding only the slotting and winding it needs for five
phases. Excitation is dq1 + dq3.
"""

import numpy as np
from ansys.aedt.core import Maxwell2d

from machine_design.designs.computation import ComputationBase

from .synrm_3f_36s import Geometry as BaseGeometry


class Geometry(BaseGeometry):
    def set_geom_params(self):
        super().set_geom_params()
        self.geom_params["SlotNumber"] = "40"

    def set_slot_params(self):
        super().set_slot_params()
        self.slot_params["Bs1"] = "3.0mm"
        self.slot_params["Bs2"] = "4.3mm"
        self.slot_params["SetAngle"] = "9deg"

    def set_winds_params(self):
        super().set_winds_params()
        self.wind_params["Nc"] = "113"


class Computation(ComputationBase):
    def set_oper_params(self):
        f = 50  # [Hz]
        RotSpeed = 60 * f / self.geometry.PolePairs  # [rpm]
        w = 2 * np.pi * f
        self.InitPos = -45.0 # deg
        self.RotSign = 1
        self.w = w
        self.oper_params = {
            "Id1": "0.0A",
            "Iq1": "0.0A",
            "Id3": "0.0A",
            "Iq3": "0.0A",
            "epsI1": "atan2(Iq1,Id1)",  # current angle, 1st harmonic
            "epsI3": "atan2(Iq3,Id3)",  # current angle, 1st harmonic
            "Im1": "sqrt(Id1^2+Iq1^2)",
            "Im3": "sqrt(Id3^2+Iq3^2)",
            "InitPos": f"{self.InitPos}deg",
            "w": f"{w}Hz",
            "RotSpeed": f"{RotSpeed}rpm",
            "Nper": "1/10",  # number of included periods
            "PointPer": "101",  # number of time points per period
        }

    def set_solution_expressions(self):
        self.solution_expressions = [
            "Moving1.Position",
            "Moving1.Torque",
            "FluxLinkage(PhaseA)",
            "FluxLinkage(PhaseB)",
            "FluxLinkage(PhaseC)",
            "FluxLinkage(PhaseD)",
            "FluxLinkage(PhaseE)",
            "InducedVoltage(PhaseA)",
            "InducedVoltage(PhaseB)",
            "InducedVoltage(PhaseC)",
            "InducedVoltage(PhaseD)",
            "InducedVoltage(PhaseE)",
            "InputCurrent(PhaseA)",
            "InputCurrent(PhaseB)",
            "InputCurrent(PhaseC)",
            "InputCurrent(PhaseD)",
            "InputCurrent(PhaseE)",
            "L(PhaseA,PhaseA)",
            "L(PhaseA,PhaseB)",
            "L(PhaseA,PhaseC)",
            "L(PhaseA,PhaseD)",
            "L(PhaseA,PhaseE)",
            "L(PhaseB,PhaseA)",
            "L(PhaseB,PhaseB)",
            "L(PhaseB,PhaseC)",
            "L(PhaseB,PhaseD)",
            "L(PhaseB,PhaseE)",
            "L(PhaseC,PhaseA)",
            "L(PhaseC,PhaseB)",
            "L(PhaseC,PhaseC)",
            "L(PhaseC,PhaseD)",
            "L(PhaseC,PhaseE)",
            "L(PhaseD,PhaseA)",
            "L(PhaseD,PhaseB)",
            "L(PhaseD,PhaseC)",
            "L(PhaseD,PhaseD)",
            "L(PhaseD,PhaseE)",
            "L(PhaseE,PhaseA)",
            "L(PhaseE,PhaseB)",
            "L(PhaseE,PhaseC)",
            "L(PhaseE,PhaseD)",
            "L(PhaseE,PhaseE)",
        ]

    def set_output_vars(self):
        self.output_vars = {}

    def set_post_params(self):
        self.post_params = {  # reports
            (
                "InducedVoltage(PhaseA)",
                "InducedVoltage(PhaseB)",
                "InducedVoltage(PhaseC)",
                "InducedVoltage(PhaseD)",
                "InducedVoltage(PhaseE)",
            ): "InducedVoltage",
            ("Moving1.Torque"): "Torque",
            (
                "InputCurrent(PhaseA)",
                "InputCurrent(PhaseB)",
                "InputCurrent(PhaseC)",
                "InputCurrent(PhaseD)",
                "InputCurrent(PhaseE)",
            ): "Current",
            (
                "FluxLinkage(PhaseA)",
                "FluxLinkage(PhaseB)",
                "FluxLinkage(PhaseC)",
                "FluxLinkage(PhaseD)",
                "FluxLinkage(PhaseE)",
            ): "FluxLinkage",
            ("I_d1", "I_q1", "I_d3", "I_q3"): "Current_dq",
            ("Flux_d1", "Flux_q1", "Flux_d3", "Flux_q3"): "FluxLinkage_dq",
            ("Flux_e_d1", "Flux_e_q1", "Flux_e_d3", "Flux_e_q3"): "FluxLinkage excitation_dq",
            ("Vind_d1", "Vind_q1", "Vind_d3", "Vind_q3"): "InducedVoltage_dq",
            ("V_d1", "V_q1", "V_d3", "V_q3"): "TerminalVoltage_dq",
            ("Ld1", "Lq1", "Ld3", "Lq3"): "Inductance_dq main",
            ("Ld1q1", "Ld1d3", "Ld1q3", "Lq1d3", "Lq1q3", "Ld3q3"): "Inductance_dq cross-coupling",
        }

    def assign_stator_coils(self, m2d: Maxwell2d) -> None:
        # Excitations
        I_A = "Im1*cos(w*Time+epsI1-pi) + Im3*cos(3*(w*Time)+epsI3-pi)"
        I_B = "Im1*cos(w*Time-72deg+epsI1-pi) + Im3*cos(3*(w*Time-72deg)+epsI3-pi)"
        I_C = "Im1*cos(w*Time-144deg+epsI1-pi) + Im3*cos(3*(w*Time-144deg)+epsI3-pi)"
        I_D = "Im1*cos(w*Time-216deg+epsI1-pi) + Im3*cos(3*(w*Time-216deg)+epsI3-pi)"
        I_E = "Im1*cos(w*Time-288deg+epsI1-pi) + Im3*cos(3*(w*Time-288deg)+epsI3-pi)"
        m2d.assign_coil
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
            polarity="Negative",
            name="CS2",
        )
        m2d.assign_coil(
            assignment=["Coil_2"],
            conductors_number="Nc",
            polarity="Negative",
            name="CS3",
        )
        m2d.assign_coil(
            assignment=["Coil_3"],
            conductors_number="Nc",
            polarity="Positive",
            name="CS4",
        )
        m2d.assign_coil(
            assignment=["Coil_4"],
            conductors_number="Nc",
            polarity="Positive",
            name="CS5",
        )
        m2d.assign_coil(
            assignment=["Coil_5"],
            conductors_number="Nc",
            polarity="Negative",
            name="CS6",
        )
        m2d.assign_coil(
            assignment=["Coil_6"],
            conductors_number="Nc",
            polarity="Negative",
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
        m2d.assign_coil(
            assignment=["Coil_9"],
            conductors_number="Nc",
            polarity="Negative",
            name="CS10",
        )

        m2d.assign_winding(
            assignment=None,
            winding_type="Current",
            is_solid=False,
            current=I_A,
            parallel_branches="ParallelPaths",
            name="PhaseA",
        )
        m2d.assign_winding(
            assignment=None,
            winding_type="Current",
            is_solid=False,
            current=I_B,
            parallel_branches="ParallelPaths",
            name="PhaseB",
        )
        m2d.assign_winding(
            assignment=None,
            winding_type="Current",
            is_solid=False,
            current=I_C,
            parallel_branches="ParallelPaths",
            name="PhaseC",
        )
        m2d.assign_winding(
            assignment=None,
            winding_type="Current",
            is_solid=False,
            current=I_D,
            parallel_branches="ParallelPaths",
            name="PhaseD",
        )
        m2d.assign_winding(
            assignment=None,
            winding_type="Current",
            is_solid=False,
            current=I_E,
            parallel_branches="ParallelPaths",
            name="PhaseE",
        )

        m2d.add_winding_coils(assignment="PhaseA", coils=["CS1", "CS10"])
        m2d.add_winding_coils(assignment="PhaseB", coils=["CS4", "CS5"])
        m2d.add_winding_coils(assignment="PhaseC", coils=["CS8", "CS9"])
        m2d.add_winding_coils(assignment="PhaseD", coils=["CS2", "CS3"])
        m2d.add_winding_coils(assignment="PhaseE", coils=["CS6", "CS7"])

    def inductance_computation(self, m2d: Maxwell2d) -> None:
        m2d.change_inductance_computation(compute_transient_inductance=True, incremental_matrix=True)

    def set_variables(self, m2d: Maxwell2d, Id1, Iq1, Id3, Iq3):
        m2d.variable_manager["Id1"] = f"{Id1}A"
        m2d.variable_manager["Iq1"] = f"{Iq1}A"
        m2d.variable_manager["Id3"] = f"{Id3}A"
        m2d.variable_manager["Iq3"] = f"{Iq3}A"

    def extract_results(self, solutions):
        out = {}
        for expr in self.solution_expressions:
            val = solutions.data_real(expr)

            if expr.startswith("Ld") or expr.startswith("Lq"):
                val = np.array(val) / 1e9
            elif expr.startswith("I_"):
                val = np.array(val) / 1e3

            # TODO: possibly assign val[:-1]. check whether values are identical
            out[expr] = val
        return out
