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

    def set_rot_points(self):
        self.rot_points = [
            ["DiaShaft/2*cos(360deg/SymmetryFactor)", "DiaShaft/2*sin(360deg/SymmetryFactor)", "0mm"],
            ["DiaShaft/2*cos(360deg/(2*SymmetryFactor))", "DiaShaft/2*sin(360deg/(2*SymmetryFactor))", "0mm"],
            ["DiaShaft/2", "0mm", "0mm"],
            ["DiaStatorGap/2-Airgap", "0mm", "0mm"],
            ["(DiaStatorGap/2-Airgap)*cos(360deg/(2*SymmetryFactor))", "(DiaStatorGap/2-Airgap)*sin(360deg/(2*SymmetryFactor))", "0mm"],
            ["(DiaStatorGap/2-Airgap)*cos(360deg/SymmetryFactor)", "(DiaStatorGap/2-Airgap)*sin(360deg/SymmetryFactor)", "0mm"],
        ]

    def set_derived_params(self):
        pass