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
            "epsI1": "atan2(Iq1,Id1)", #current angle, 1st harmonic
            "epsI3": "atan2(Iq3,Id3)", #current angle, 1st harmonic
            "Im1": "sqrt(Id1^2+Iq1^2)",
            "Im3": "sqrt(Id3^2+Iq3^2)",
            "InitPos": "-45deg",
            "f": f"{f}Hz",
            "RotSpeed": f"{RotSpeed}rpm",
            "Nper": "1/10",  # number of included periods
            "PointPer": "101",  # number of time points per period
        }

    def set_derived_params(self):
        pass

    def set_solution_expressions(self):
        self.expressions = [
            "V_d1", "V_q1", "V_d3", "V_q3",
            "Flux_e_d1", "Flux_e_q1", "Flux_e_d3", "Flux_e_q3",
            "I_d1", "I_q1", "I_d3", "I_q3",
            "Ld1", "Ld1q1", "Ld1d3", "Ld1q3",
            "Lq1", "Lq1d3", "Lq1q3",
            "Ld3", "Ld3q3",
            "Lq3",
            "Moving1.Torque"
        ]

    def set_udp_par_list_stator(self):
        self.udp_par_list_stator = [
            ["DiaGap", "DiaStatorGap"],
            ["DiaYoke", "DiaStatorYoke"],
            ["Length", "0mm"],
            ["Skew", "0deg"],
            ["Slots", "SlotNumber"],
            ["SlotType", "SlotType"],
            ["Hs0", "Hs0"],
            ["Hs01", "0mm"],
            ["Hs1", "Hs1"],
            ["Hs2", "Hs2"],
            ["Bs0", "Bs0"],
            ["Bs1", "Bs1"],
            ["Bs2", "Bs2"],
            ["Rs", "Rs"],
            ["FilletType", "0"],
            ["HalfSlot", "0"],
            ["SegAngle", "0deg"],
            ["LenRegion", "0mm"],
            ["InfoCore", "0"],
        ]
    
    def set_output_vars(self):
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
            ("I_d", "I_q"): "Current_dq",
            ("Flux_d", "Flux_q"): "FluxLinkage_dq",
            ("Ui_d", "Ui_q"): "InducedVoltage_dq",
            ("L_d", "L_q"): "Inductance_dq",
        }


