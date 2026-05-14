from .design import Design


class Design2(Design):
    def set_iron(self):
        self.Fe = "Cogent Power - M350-50A, B-H at 50Hz"

    def set_geom_params(self):
        self.geom_params = {
            "DiaStatorGap": "79mm",
            "DiaStatorYoke": "125mm",
            "Airgap": "0.225mm",
            "SlotNumber": "36",
            "SlotType": "3",
            "DiaShaft": "25mm",
            "StackLength": "85mm",
        }

    def set_slot_params(self):
        self.slot_params = {
            "Hs0": "0.95mm",
            "Hs1": "0.31mm",
            "Hs2": "8.24mm",
            "Bs0": "2.2mm",
            "Bs1": "3.37mm",
            "Bs2": "4.8mm",
            "Rs": "1.5mm",
            "SetAngle": "10deg",
        }

    def set_winds_params(self):
        self.wind_params = {
            "Layers": "1",
            "ParallelPaths": "1",
            "CoilPitch": "9",  # coil pitch in slots
            "SlotLiner": "0.3mm",
            "SpaceLayers": "0.2mm",
            "Nc": "68",  # turns per coil
        }

    def set_mod_params(self):
        self.PolePairs = 2
        self.mod_params = {
            "Poles": f"2*{self.PolePairs}",
            "ModelLength": "85mm",
            "SymmetryFactor": "Poles",
            "StatorSkewAngle": "0deg",
        }

    def set_oper_params(self):
        f = 50  # [Hz]
        RotSpeed = 60 * f / self.PolePairs  # [rpm]
        self.oper_params = {
            "Im": "1.5*sqrt(2)A",
            "epsI": "pi/4",  # current angle
            "InitPos": "-30deg",
            "f": f"{f}Hz",
            "RotSpeed": f"{RotSpeed}rpm",
            "Nper": "1/6",  # number of included periods
            "PointPer": "101",  # number of time points per period
        }

    def set_rot_points(self):
        self.rot_points = [
            ["DiaShaft/2*cos(360deg/SymmetryFactor)", "DiaShaft/2*sin(360deg/SymmetryFactor)", "0mm"],
            ["DiaShaft/2*cos(360deg/(2*SymmetryFactor))", "DiaShaft/2*sin(360deg/(2*SymmetryFactor))", "0mm"],
            ["DiaShaft/2", "0mm", "0mm"],
            ["DiaStatorGap/2-Airgap", "0mm", "0mm"],
            [
                "(DiaStatorGap/2-Airgap)*cos(360deg/(2*SymmetryFactor))",
                "(DiaStatorGap/2-Airgap)*sin(360deg/(2*SymmetryFactor))",
                "0mm",
            ],
            [
                "(DiaStatorGap/2-Airgap)*cos(360deg/SymmetryFactor)",
                "(DiaStatorGap/2-Airgap)*sin(360deg/SymmetryFactor)",
                "0mm",
            ],
        ]

    def set_derived_params(self):
        self.rotor_r_min = self.mm_to_str("geom_params", "DiaShaft") / 2
        self.rotor_r_max = self.mm_to_str("geom_params", "DiaStatorGap") / 2 - self.mm_to_str("geom_params", "Airgap")