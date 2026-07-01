import numpy as np

from .design import Design


class Design2(Design):
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

    def set_oper_params(self):
        f = 50  # [Hz]
        RotSpeed = 60 * f / self.PolePairs  # [rpm]
        self.oper_params = {
            "Id1": "0.0A",
            "Iq1": "0.0A",
            "Id3": "0.0A",
            "Iq3": "0.0A",
            "epsI1": "atan2(Iq1,Id1)",  # current angle, 1st harmonic
            "epsI3": "atan2(Iq3,Id3)",  # current angle, 1st harmonic
            "Im1": "sqrt(Id1^2+Iq1^2)",
            "Im3": "sqrt(Id3^2+Iq3^2)",
            "InitPos": "-45deg",
            "f": f"{f}Hz",
            "RotSpeed": f"{RotSpeed}rpm",
            # Full electrical period at PointPer=101 (~100 steps) = a COARSE
            # FEA fidelity, kept for BO throughput (1-core, K-way parallel).
            # NOTE: the "dq3-dependent short-window bias" we first attributed to
            # broken m-phase symmetry turned out to be a RESOLUTION artifact, not
            # physics. At a converged mesh (rotor + airgap 0.5 mm, ported from
            # ReluctanceDrive/newparam via compute(mesh_length=, airgap_mesh=))
            # and PointPer >= 201, the torque is genuinely Te/(2m)-periodic and
            # the 1/10 window is unbiased (<0.05%). At this 3 mm / 101 setting the
            # MEAN torque carries a ~0.5-1.5% resolution bias (mildly operating-
            # point-dependent), so final results need converged-fidelity
            # re-confirmation (see paper caveats and docs/setup-bayes.md).
            "Nper": "1",  # number of included electrical periods
            "PointPer": "101",  # number of time points per period
        }

    def set_derived_params(self):
        # Derive rotor_r_min/max from the (inherited) shaft and stator-gap
        # diameters; the BarrierGenerators require these. Previously stubbed
        # `pass`, which left them unset and broke barrier generation for the
        # 5-phase design.
        super().set_derived_params()

    def set_solution_expressions(self):
        self.solution_expressions = [
            "V_d1",
            "V_q1",
            "V_d3",
            "V_q3",
            "Flux_d1",
            "Flux_q1",
            "Flux_d3",
            "Flux_q3",
            "Flux_e_d1",
            "Flux_e_q1",
            "Flux_e_d3",
            "Flux_e_q3",
            "I_d1",
            "I_q1",
            "I_d3",
            "I_q3",
            "Ld1",
            "Ld1q1",
            "Ld1d3",
            "Ld1q3",
            "Lq1",
            "Lq1d3",
            "Lq1q3",
            "Ld3",
            "Ld3q3",
            "Lq3",
            "Moving1.Torque",
        ]

    def set_output_vars(self):
        self.output_vars = {
            "PolePairs": "2",
            "RotSign": "1",
            "Rstat": "19",   # phase resistance consistent with Nc=113 (the 0.19 was a mistake: it decoupled R from the winding/back-EMF)
            "Lew": "0",      # end-winding leakage inductance [H] — set by set_derived_params() analytical estimate
            "theta_el": "RotSign*(Moving1.Position - InitPos) * PolePairs - pi",
            "cos0_1": "cos(1*(theta_el - 2*PI*0/5))",
            "sin0_1": "sin(-1*(theta_el - 2*PI*0/5))",
            "cos1_1": "cos(1*(theta_el - 2*PI*1/5))",
            "sin1_1": "sin(-1*(theta_el - 2*PI*1/5))",
            "cos2_1": "cos(1*(theta_el - 2*PI*2/5))",
            "sin2_1": "sin(-1*(theta_el - 2*PI*2/5))",
            "cos3_1": "cos(1*(theta_el - 2*PI*3/5))",
            "sin3_1": "sin(-1*(theta_el - 2*PI*3/5))",
            "cos4_1": "cos(1*(theta_el - 2*PI*4/5))",
            "sin4_1": "sin(-1*(theta_el - 2*PI*4/5))",
            "cos0_3": "cos(3*(theta_el - 2*PI*0/5))",
            "sin0_3": "sin(-3*(theta_el - 2*PI*0/5))",
            "cos1_3": "cos(3*(theta_el - 2*PI*1/5))",
            "sin1_3": "sin(-3*(theta_el - 2*PI*1/5))",
            "cos2_3": "cos(3*(theta_el - 2*PI*2/5))",
            "sin2_3": "sin(-3*(theta_el - 2*PI*2/5))",
            "cos3_3": "cos(3*(theta_el - 2*PI*3/5))",
            "sin3_3": "sin(-3*(theta_el - 2*PI*3/5))",
            "cos4_3": "cos(3*(theta_el - 2*PI*4/5))",
            "sin4_3": "sin(-3*(theta_el - 2*PI*4/5))",
            "Flux_d1": "(FluxLinkage(PhaseA)*cos0_1 + FluxLinkage(PhaseB)*cos1_1 + FluxLinkage(PhaseC)*cos2_1 + FluxLinkage(PhaseD)*cos3_1 + FluxLinkage(PhaseE)*cos4_1) * 2/5",
            "Flux_q1": "(FluxLinkage(PhaseA)*sin0_1 + FluxLinkage(PhaseB)*sin1_1 + FluxLinkage(PhaseC)*sin2_1 + FluxLinkage(PhaseD)*sin3_1 + FluxLinkage(PhaseE)*sin4_1) * 2/5",
            "Flux_d3": "(FluxLinkage(PhaseA)*cos0_3 + FluxLinkage(PhaseB)*cos1_3 + FluxLinkage(PhaseC)*cos2_3 + FluxLinkage(PhaseD)*cos3_3 + FluxLinkage(PhaseE)*cos4_3) * 2/5",
            "Flux_q3": "(FluxLinkage(PhaseA)*sin0_3 + FluxLinkage(PhaseB)*sin1_3 + FluxLinkage(PhaseC)*sin2_3 + FluxLinkage(PhaseD)*sin3_3 + FluxLinkage(PhaseE)*sin4_3) * 2/5",
            "Vind_d1": "(InducedVoltage(PhaseA)*cos0_1 + InducedVoltage(PhaseB)*cos1_1 + InducedVoltage(PhaseC)*cos2_1 + InducedVoltage(PhaseD)*cos3_1 + InducedVoltage(PhaseE)*cos4_1) * 2/5",
            "Vind_q1": "(InducedVoltage(PhaseA)*sin0_1 + InducedVoltage(PhaseB)*sin1_1 + InducedVoltage(PhaseC)*sin2_1 + InducedVoltage(PhaseD)*sin3_1 + InducedVoltage(PhaseE)*sin4_1) * 2/5",
            "Vind_d3": "(InducedVoltage(PhaseA)*cos0_3 + InducedVoltage(PhaseB)*cos1_3 + InducedVoltage(PhaseC)*cos2_3 + InducedVoltage(PhaseD)*cos3_3 + InducedVoltage(PhaseE)*cos4_3) * 2/5",
            "Vind_q3": "(InducedVoltage(PhaseA)*sin0_3 + InducedVoltage(PhaseB)*sin1_3 + InducedVoltage(PhaseC)*sin2_3 + InducedVoltage(PhaseD)*sin3_3 + InducedVoltage(PhaseE)*sin4_3) * 2/5",
            "V_A": "InducedVoltage(PhaseA) + Rstat*InputCurrent(PhaseA) + Lew*deriv(InputCurrent(PhaseA))",
            "V_B": "InducedVoltage(PhaseB) + Rstat*InputCurrent(PhaseB) + Lew*deriv(InputCurrent(PhaseB))",
            "V_C": "InducedVoltage(PhaseC) + Rstat*InputCurrent(PhaseC) + Lew*deriv(InputCurrent(PhaseC))",
            "V_D": "InducedVoltage(PhaseD) + Rstat*InputCurrent(PhaseD) + Lew*deriv(InputCurrent(PhaseD))",
            "V_E": "InducedVoltage(PhaseE) + Rstat*InputCurrent(PhaseE) + Lew*deriv(InputCurrent(PhaseE))",
            "V_AC": "V_A - V_C",
            "V_BD": "V_B - V_D",
            "V_CE": "V_C - V_E",
            "V_DA": "V_D - V_A",
            "V_EB": "V_E - V_B",
            "Vterm_A": "1/5*(2*V_AC + -1*V_BD + 1*V_CE + -2*V_DA)",
            "Vterm_B": "1/5*(2*V_AC + 4*V_BD + 1*V_CE + 3*V_DA)",
            "Vterm_C": "1/5*(-3*V_AC + -1*V_BD + 1*V_CE + -2*V_DA)",
            "Vterm_D": "1/5*(2*V_AC + -1*V_BD + 1*V_CE + 3*V_DA)",
            "Vterm_E": "1/5*(-3*V_AC + -1*V_BD + -4*V_CE + -2*V_DA)",
            "I_d1": "(InputCurrent(PhaseA)*cos0_1 + InputCurrent(PhaseB)*cos1_1 + InputCurrent(PhaseC)*cos2_1 + InputCurrent(PhaseD)*cos3_1 + InputCurrent(PhaseE)*cos4_1) * 2/5",
            "I_q1": "(InputCurrent(PhaseA)*sin0_1 + InputCurrent(PhaseB)*sin1_1 + InputCurrent(PhaseC)*sin2_1 + InputCurrent(PhaseD)*sin3_1 + InputCurrent(PhaseE)*sin4_1) * 2/5",
            "I_d3": "(InputCurrent(PhaseA)*cos0_3 + InputCurrent(PhaseB)*cos1_3 + InputCurrent(PhaseC)*cos2_3 + InputCurrent(PhaseD)*cos3_3 + InputCurrent(PhaseE)*cos4_3) * 2/5",
            "I_q3": "(InputCurrent(PhaseA)*sin0_3 + InputCurrent(PhaseB)*sin1_3 + InputCurrent(PhaseC)*sin2_3 + InputCurrent(PhaseD)*sin3_3 + InputCurrent(PhaseE)*sin4_3) * 2/5",
            "V_d1": "(V_A*cos0_1 + V_B*cos1_1 + V_C*cos2_1 + V_D*cos3_1 + V_E*cos4_1) * 2/5",
            "V_q1": "(V_A*sin0_1 + V_B*sin1_1 + V_C*sin2_1 + V_D*sin3_1 + V_E*sin4_1) * 2/5",
            "V_d3": "(V_A*cos0_3 + V_B*cos1_3 + V_C*cos2_3 + V_D*cos3_3 + V_E*cos4_3) * 2/5",
            "V_q3": "(V_A*sin0_3 + V_B*sin1_3 + V_C*sin2_3 + V_D*sin3_3 + V_E*sin4_3) * 2/5",
            "L0d_1": "L(PhaseA,PhaseA)*cos0_1 + L(PhaseA,PhaseB)*cos1_1 + L(PhaseA,PhaseC)*cos2_1 + L(PhaseA,PhaseD)*cos3_1 + L(PhaseA,PhaseE)*cos4_1",
            "L0q_1": "L(PhaseA,PhaseA)*sin0_1 + L(PhaseA,PhaseB)*sin1_1 + L(PhaseA,PhaseC)*sin2_1 + L(PhaseA,PhaseD)*sin3_1 + L(PhaseA,PhaseE)*sin4_1",
            "L1d_1": "L(PhaseB,PhaseA)*cos0_1 + L(PhaseB,PhaseB)*cos1_1 + L(PhaseB,PhaseC)*cos2_1 + L(PhaseB,PhaseD)*cos3_1 + L(PhaseB,PhaseE)*cos4_1",
            "L1q_1": "L(PhaseB,PhaseA)*sin0_1 + L(PhaseB,PhaseB)*sin1_1 + L(PhaseB,PhaseC)*sin2_1 + L(PhaseB,PhaseD)*sin3_1 + L(PhaseB,PhaseE)*sin4_1",
            "L2d_1": "L(PhaseC,PhaseA)*cos0_1 + L(PhaseC,PhaseB)*cos1_1 + L(PhaseC,PhaseC)*cos2_1 + L(PhaseC,PhaseD)*cos3_1 + L(PhaseC,PhaseE)*cos4_1",
            "L2q_1": "L(PhaseC,PhaseA)*sin0_1 + L(PhaseC,PhaseB)*sin1_1 + L(PhaseC,PhaseC)*sin2_1 + L(PhaseC,PhaseD)*sin3_1 + L(PhaseC,PhaseE)*sin4_1",
            "L3d_1": "L(PhaseD,PhaseA)*cos0_1 + L(PhaseD,PhaseB)*cos1_1 + L(PhaseD,PhaseC)*cos2_1 + L(PhaseD,PhaseD)*cos3_1 + L(PhaseD,PhaseE)*cos4_1",
            "L3q_1": "L(PhaseD,PhaseA)*sin0_1 + L(PhaseD,PhaseB)*sin1_1 + L(PhaseD,PhaseC)*sin2_1 + L(PhaseD,PhaseD)*sin3_1 + L(PhaseD,PhaseE)*sin4_1",
            "L4d_1": "L(PhaseE,PhaseA)*cos0_1 + L(PhaseE,PhaseB)*cos1_1 + L(PhaseE,PhaseC)*cos2_1 + L(PhaseE,PhaseD)*cos3_1 + L(PhaseE,PhaseE)*cos4_1",
            "L4q_1": "L(PhaseE,PhaseA)*sin0_1 + L(PhaseE,PhaseB)*sin1_1 + L(PhaseE,PhaseC)*sin2_1 + L(PhaseE,PhaseD)*sin3_1 + L(PhaseE,PhaseE)*sin4_1",
            "L0d_3": "L(PhaseA,PhaseA)*cos0_3 + L(PhaseA,PhaseB)*cos1_3 + L(PhaseA,PhaseC)*cos2_3 + L(PhaseA,PhaseD)*cos3_3 + L(PhaseA,PhaseE)*cos4_3",
            "L0q_3": "L(PhaseA,PhaseA)*sin0_3 + L(PhaseA,PhaseB)*sin1_3 + L(PhaseA,PhaseC)*sin2_3 + L(PhaseA,PhaseD)*sin3_3 + L(PhaseA,PhaseE)*sin4_3",
            "L1d_3": "L(PhaseB,PhaseA)*cos0_3 + L(PhaseB,PhaseB)*cos1_3 + L(PhaseB,PhaseC)*cos2_3 + L(PhaseB,PhaseD)*cos3_3 + L(PhaseB,PhaseE)*cos4_3",
            "L1q_3": "L(PhaseB,PhaseA)*sin0_3 + L(PhaseB,PhaseB)*sin1_3 + L(PhaseB,PhaseC)*sin2_3 + L(PhaseB,PhaseD)*sin3_3 + L(PhaseB,PhaseE)*sin4_3",
            "L2d_3": "L(PhaseC,PhaseA)*cos0_3 + L(PhaseC,PhaseB)*cos1_3 + L(PhaseC,PhaseC)*cos2_3 + L(PhaseC,PhaseD)*cos3_3 + L(PhaseC,PhaseE)*cos4_3",
            "L2q_3": "L(PhaseC,PhaseA)*sin0_3 + L(PhaseC,PhaseB)*sin1_3 + L(PhaseC,PhaseC)*sin2_3 + L(PhaseC,PhaseD)*sin3_3 + L(PhaseC,PhaseE)*sin4_3",
            "L3d_3": "L(PhaseD,PhaseA)*cos0_3 + L(PhaseD,PhaseB)*cos1_3 + L(PhaseD,PhaseC)*cos2_3 + L(PhaseD,PhaseD)*cos3_3 + L(PhaseD,PhaseE)*cos4_3",
            "L3q_3": "L(PhaseD,PhaseA)*sin0_3 + L(PhaseD,PhaseB)*sin1_3 + L(PhaseD,PhaseC)*sin2_3 + L(PhaseD,PhaseD)*sin3_3 + L(PhaseD,PhaseE)*sin4_3",
            "L4d_3": "L(PhaseE,PhaseA)*cos0_3 + L(PhaseE,PhaseB)*cos1_3 + L(PhaseE,PhaseC)*cos2_3 + L(PhaseE,PhaseD)*cos3_3 + L(PhaseE,PhaseE)*cos4_3",
            "L4q_3": "L(PhaseE,PhaseA)*sin0_3 + L(PhaseE,PhaseB)*sin1_3 + L(PhaseE,PhaseC)*sin2_3 + L(PhaseE,PhaseD)*sin3_3 + L(PhaseE,PhaseE)*sin4_3",
            "Ld1": "(L0d_1*cos0_1 + L1d_1*cos1_1 + L2d_1*cos2_1 + L3d_1*cos3_1 + L4d_1*cos4_1) * 2/5",
            "Ld1q1": "(L0d_1*sin0_1 + L1d_1*sin1_1 + L2d_1*sin2_1 + L3d_1*sin3_1 + L4d_1*sin4_1) * 2/5",
            "Lq1d1": "(L0q_1*cos0_1 + L1q_1*cos1_1 + L2q_1*cos2_1 + L3q_1*cos3_1 + L4q_1*cos4_1) * 2/5",
            "Lq1": "(L0q_1*sin0_1 + L1q_1*sin1_1 + L2q_1*sin2_1 + L3q_1*sin3_1 + L4q_1*sin4_1) * 2/5",
            "Ld1d3": "(L0d_1*cos0_3 + L1d_1*cos1_3 + L2d_1*cos2_3 + L3d_1*cos3_3 + L4d_1*cos4_3) * 2/5",
            "Ld1q3": "(L0d_1*sin0_3 + L1d_1*sin1_3 + L2d_1*sin2_3 + L3d_1*sin3_3 + L4d_1*sin4_3) * 2/5",
            "Lq1d3": "(L0q_1*cos0_3 + L1q_1*cos1_3 + L2q_1*cos2_3 + L3q_1*cos3_3 + L4q_1*cos4_3) * 2/5",
            "Lq1q3": "(L0q_1*sin0_3 + L1q_1*sin1_3 + L2q_1*sin2_3 + L3q_1*sin3_3 + L4q_1*sin4_3) * 2/5",
            "Ld3d1": "(L0d_3*cos0_1 + L1d_3*cos1_1 + L2d_3*cos2_1 + L3d_3*cos3_1 + L4d_3*cos4_1) * 2/5",
            "Ld3q1": "(L0d_3*sin0_1 + L1d_3*sin1_1 + L2d_3*sin2_1 + L3d_3*sin3_1 + L4d_3*sin4_1) * 2/5",
            "Lq3d1": "(L0q_3*cos0_1 + L1q_3*cos1_1 + L2q_3*cos2_1 + L3q_3*cos3_1 + L4q_3*cos4_1) * 2/5",
            "Lq3q1": "(L0q_3*sin0_1 + L1q_3*sin1_1 + L2q_3*sin2_1 + L3q_3*sin3_1 + L4q_3*sin4_1) * 2/5",
            "Ld3": "(L0d_3*cos0_3 + L1d_3*cos1_3 + L2d_3*cos2_3 + L3d_3*cos3_3 + L4d_3*cos4_3) * 2/5",
            "Ld3q3": "(L0d_3*sin0_3 + L1d_3*sin1_3 + L2d_3*sin2_3 + L3d_3*sin3_3 + L4d_3*sin4_3) * 2/5",
            "Lq3d3": "(L0q_3*cos0_3 + L1q_3*cos1_3 + L2q_3*cos2_3 + L3q_3*cos3_3 + L4q_3*cos4_3) * 2/5",
            "Lq3": "(L0q_3*sin0_3 + L1q_3*sin1_3 + L2q_3*sin2_3 + L3q_3*sin3_3 + L4q_3*sin4_3) * 2/5",
            "Flux_e_d1": "Flux_d1 - (Ld1*I_d1 + Ld1q1*I_q1 + Ld1d3*I_d3 + Ld1q3*I_q3)",
            "Flux_e_q1": "Flux_q1 - (Lq1d1*I_d1 + Lq1*I_q1 + Lq1d3*I_d3 + Lq1q3*I_q3)",
            "Flux_e_d3": "Flux_d3 - (Ld3d1*I_d1 + Ld3q1*I_q1 + Ld3*I_d3 + Ld3q3*I_q3)",
            "Flux_e_q3": "Flux_q3 - (Lq3d1*I_d1 + Lq3q1*I_q1 + Lq3d3*I_d3 + Lq3*I_q3)",
            "Torque_dq": "5/2*PolePairs*(1*(Flux_d1*I_q1 - Flux_q1*I_d1) + 3*(Flux_d3*I_q3 - Flux_q3*I_d3))",
        }

    def set_post_params(self):
        self.post_params = {  # reports
            ("InducedVoltage(PhaseA)", "InducedVoltage(PhaseB)", "InducedVoltage(PhaseC)", "InducedVoltage(PhaseD)", "InducedVoltage(PhaseE)"): "InducedVoltage",
            ("Moving1.Torque"): "Torque",
            ("InputCurrent(PhaseA)", "InputCurrent(PhaseB)", "InputCurrent(PhaseC)", "InputCurrent(PhaseD)", "InputCurrent(PhaseE)"): "Current",
            ("FluxLinkage(PhaseA)", "FluxLinkage(PhaseB)", "FluxLinkage(PhaseC)", "FluxLinkage(PhaseD)", "FluxLinkage(PhaseE)"): "FluxLinkage",
            ("I_d1", "I_q1", "I_d3", "I_q3"): "Current_dq",
            ("Flux_d1", "Flux_q1", "Flux_d3", "Flux_q3"): "FluxLinkage_dq",
            ("Flux_e_d1", "Flux_e_q1", "Flux_e_d3", "Flux_e_q3"): "FluxLinkage excitation_dq",
            ("Vind_d1", "Vind_q1", "Vind_d3", "Vind_q3"): "InducedVoltage_dq",
            ("V_d1", "V_q1", "V_d3", "V_q3"): "TerminalVoltage_dq",
            ("Ld1", "Lq1", "Ld3", "Lq3"): "Inductance_dq main",
            ("Ld1q1", "Ld1d3", "Ld1q3", "Lq1d3", "Lq1q3", "Ld3q3"): "Inductance_dq cross-coupling",
        }

    def assign_stator_coils(self):
        # Excitations: fundamental + 3rd harmonic; phase k shifted by -360/m*k deg.
        I_A = "Im1*cos(2*pi*f*time+epsI1-pi) + Im3*cos(3*(2*pi*f*time)+epsI3-pi)"
        I_B = "Im1*cos(2*pi*f*time-72deg+epsI1-pi) + Im3*cos(3*(2*pi*f*time-72deg)+epsI3-pi)"
        I_C = "Im1*cos(2*pi*f*time-144deg+epsI1-pi) + Im3*cos(3*(2*pi*f*time-144deg)+epsI3-pi)"
        I_D = "Im1*cos(2*pi*f*time-216deg+epsI1-pi) + Im3*cos(3*(2*pi*f*time-216deg)+epsI3-pi)"
        I_E = "Im1*cos(2*pi*f*time-288deg+epsI1-pi) + Im3*cos(3*(2*pi*f*time-288deg)+epsI3-pi)"
        # winding layout (coil->phase->polarity) generated by winding.py;
        # belt_offset=1 reproduces the original 40-slot/5-phase map exactly.
        self.assign_stator_coils_generic([I_A, I_B, I_C, I_D, I_E], belt_offset=1)

    def inductance_computation(self):
        self.m2d.change_inductance_computation(compute_transient_inductance=True, incremental_matrix=True)

    def set_variables(self, Id1, Iq1, Id3, Iq3):
        self.m2d.variable_manager["Id1"] = f"{Id1}A"
        self.m2d.variable_manager["Iq1"] = f"{Iq1}A"
        self.m2d.variable_manager["Id3"] = f"{Id3}A"
        self.m2d.variable_manager["Iq3"] = f"{Iq3}A"

    def extract_results(self, solutions):
        # Return both the torque time-series (for mean/ripple) and the
        # period-mean of every dq quantity (for voltage / flux / current,
        # used by the voltage-limit machinery). data[:-1] drops the duplicated
        # endpoint of the one-period window.
        means = {}
        tor = None
        for expr in self.solution_expressions:
            data = solutions.data_real(expr)
            means[expr] = float(np.mean(data[:-1]))
            if expr == "Moving1.Torque":
                tor = np.asarray(data, dtype=float)
        return {"Tor": tor, "means": means}


class Design2_60(Design2):
    """5-phase machine on a 60-slot common stator (slot widths scaled ~x0.66 from
    the 40-slot design to fit the bore pitch). Validated to build/mesh/solve."""

    def set_geom_params(self):
        super().set_geom_params()
        self.geom_params["SlotNumber"] = "60"

    def set_slot_params(self):
        super().set_slot_params()
        self.slot_params.update(Bs0="1.5mm", Bs1="2.0mm", Bs2="2.85mm",
                                Rs="1.0mm", SetAngle="6deg")

    def set_winds_params(self):
        super().set_winds_params()
        self.wind_params["CoilPitch"] = "15"
