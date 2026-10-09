"""Anchor `synrm_5f_40s`: 5-phase synchronous reluctance machine, 40 stator slots.

Derives from `synrm_3f_36s`, overriding only the slotting and winding it needs for five
phases. Excitation is dq1 + dq3.
"""

import numpy as np
from ansys.aedt.core import Maxwell2d

from machine_design.designs.computation import ComputationBase
from machine_design.generic.transforms import electrical_angle, to_dq

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

    def set_mod_params(self):
        super().set_mod_params()
        self.n_phases = 5
        self.belt_offset = 1


class Computation(ComputationBase):
    def set_oper_params(self):
        f = 50  # [Hz]
        RotSpeed = 60 * f / self.geometry.PolePairs  # [rpm]
        w = 2 * np.pi * f
        self.RotSign = 1
        self.Rstat = 19.0
        self.Lew = 0.0
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
            "w": f"{w}Hz",
            "RotSpeed": f"{RotSpeed}rpm",
            "Nper": "1/10",  # number of included periods
            "PointPer": "101",  # number of time points per period
        }
        self.set_initpos()

    def assign_stator_coils(self, m2d: Maxwell2d) -> None:
        # Excitations
        m = self.geometry.n_phases
        phase_currents = [f"Im1 * cos(w*Time-{360 * k / m}deg+epsI1-pi)+Im3*cos(3*(w*Time-{360 * k / m}deg)+epsI3-pi)" for k in range(m)]
        self.assign_phase_windings(m2d, phase_currents)

    def inductance_computation(self, m2d: Maxwell2d) -> None:
        m2d.change_inductance_computation(compute_transient_inductance=True, incremental_matrix=True)

    def set_variables(self, m2d: Maxwell2d, Id1, Iq1, Id3, Iq3):
        self.Id1, self.Iq1, self.Id3, self.Iq3 = Id1, Iq1, Id3, Iq3
        m2d.variable_manager["Id1"] = f"{Id1}A"
        m2d.variable_manager["Iq1"] = f"{Iq1}A"
        m2d.variable_manager["Id3"] = f"{Id3}A"
        m2d.variable_manager["Iq3"] = f"{Iq3}A"

    def extract_results(self, solutions):
        # SI units
        position = self.extract_expression(solutions, "Moving1.Position")
        torque = self.extract_expression(solutions, "Moving1.Torque")

        theta_el = electrical_angle(position, np.deg2rad(self.InitPos), self.geometry.PolePairs, self.RotSign, degrees=False)

        flux_phases, vind_phases, current_phases, L_raw = self.extract_phase_results(solutions)

        Flux_d1, Flux_q1 = to_dq(flux_phases, theta_el, harmonic=1)
        Flux_d3, Flux_q3 = to_dq(flux_phases, theta_el, harmonic=3)
        Vind_d1, Vind_q1 = to_dq(vind_phases, theta_el, harmonic=1)
        Vind_d3, Vind_q3 = to_dq(vind_phases, theta_el, harmonic=3)
        I_d1, I_q1 = to_dq(current_phases, theta_el, harmonic=1)
        I_d3, I_q3 = to_dq(current_phases, theta_el, harmonic=3)

        Im1 = np.sqrt(self.Id1**2 + self.Iq1**2)
        Im3 = np.sqrt(self.Id3**2 + self.Iq3**2)
        epsI1 = np.atan2(self.Iq1, self.Id1)
        epsI3 = np.atan2(self.Iq3, self.Id3)

        time = np.array(solutions.primary_sweep_values)
        dI_dt_phases = np.zeros((len(time), self.geometry.n_phases))
        for k in range(self.geometry.n_phases):
            phase_offset = -2 * np.pi * k / self.geometry.n_phases
            dI_dt_phases[:, k] = -Im1 * self.w * np.sin(self.w * time + phase_offset + epsI1 - np.pi) - Im3 * 3 * self.w * np.sin(3 * (self.w * time + phase_offset) + epsI3 - np.pi)

        V_phases = vind_phases + self.Rstat * current_phases + self.Lew * dI_dt_phases

        V_d1, V_q1 = to_dq(V_phases, theta_el, harmonic=1)
        V_d3, V_q3 = to_dq(V_phases, theta_el, harmonic=3)

        L_d1_row = np.zeros((len(time), self.geometry.n_phases))
        L_q1_row = np.zeros((len(time), self.geometry.n_phases))
        L_d3_row = np.zeros((len(time), self.geometry.n_phases))
        L_q3_row = np.zeros((len(time), self.geometry.n_phases))
        for i, L_row in enumerate(L_raw):
            L_d1_row[:, i], L_q1_row[:, i] = (v * self.geometry.n_phases / 2 for v in to_dq(L_row, theta_el, harmonic=1))
            L_d3_row[:, i], L_q3_row[:, i] = (v * self.geometry.n_phases / 2 for v in to_dq(L_row, theta_el, harmonic=3))

        Ld1, Ld1q1 = to_dq(L_d1_row, theta_el, harmonic=1)
        Lq1d1, Lq1 = to_dq(L_q1_row, theta_el, harmonic=1)
        Ld1d3, Ld1q3 = to_dq(L_d1_row, theta_el, harmonic=3)
        Lq1d3, Lq1q3 = to_dq(L_q1_row, theta_el, harmonic=3)
        Ld3d1, Ld3q1 = to_dq(L_d3_row, theta_el, harmonic=1)
        Lq3d1, Lq3q1 = to_dq(L_q3_row, theta_el, harmonic=1)
        Ld3, Ld3q3 = to_dq(L_d3_row, theta_el, harmonic=3)
        Lq3d3, Lq3 = to_dq(L_q3_row, theta_el, harmonic=3)

        Flux_e_d1 = Flux_d1 - (Ld1 * I_d1 + Ld1q1 * I_q1 + Ld1d3 * I_d3 + Ld1q3 * I_q3)
        Flux_e_q1 = Flux_q1 - (Lq1d1 * I_d1 + Lq1 * I_q1 + Lq1d3 * I_d3 + Lq1q3 * I_q3)
        Flux_e_d3 = Flux_d3 - (Ld3d1 * I_d1 + Ld3q1 * I_q1 + Ld3 * I_d3 + Ld3q3 * I_q3)
        Flux_e_q3 = Flux_q3 - (Lq3d1 * I_d1 + Lq3q1 * I_q1 + Lq3d3 * I_d3 + Lq3 * I_q3)

        Torque_dq = self.geometry.n_phases / 2 * self.geometry.PolePairs * ((Flux_d1 * I_q1 - Flux_q1 * I_d1) + 3 * (Flux_d3 * I_q3 - Flux_q3 * I_d3))

        out = {
            "V_d1": V_d1,
            "V_q1": V_q1,
            "V_d3": V_d3,
            "V_q3": V_q3,
            "Flux_d1": Flux_d1,
            "Flux_q1": Flux_q1,
            "Flux_d3": Flux_d3,
            "Flux_q3": Flux_q3,
            "Vind_d1": Vind_d1,
            "Vind_q1": Vind_q1,
            "Vind_d3": Vind_d3,
            "Vind_q3": Vind_q3,
            "I_d1": I_d1,
            "I_q1": I_q1,
            "I_d3": I_d3,
            "I_q3": I_q3,
            "Ld1": Ld1,
            "Ld1q1": Ld1q1,
            "Lq1": Lq1,
            "Ld1d3": Ld1d3,
            "Ld1q3": Ld1q3,
            "Lq1d3": Lq1d3,
            "Lq1q3": Lq1q3,
            "Ld3": Ld3,
            "Ld3q3": Ld3q3,
            "Lq3": Lq3,
            "Flux_e_d1": Flux_e_d1,
            "Flux_e_q1": Flux_e_q1,
            "Flux_e_d3": Flux_e_d3,
            "Flux_e_q3": Flux_e_q3,
            "Moving1.Torque": torque,
            "Torque_dq": Torque_dq,
        }

        return out
