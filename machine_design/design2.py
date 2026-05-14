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

    def add_rotor(self) -> None:
        modeler = self.m2d.modeler
        assert isinstance(modeler, Modeler2D)

        rotor_id = modeler.create_polyline(
            points=self.rot_points, segment_type=["Arc", "Line", "Arc"], cover_surface=True, name="Rotor"
        )
        self.rotor_id = rotor_id
        rotor_id.material_name = self.Fe
        rotor_id.color = (192, 192, 192)  # rgb
        rotor_id.transparency = 0.0

    def add_rotor_barrier(self, barrier_points, segment_type=None) -> None:
        modeler = self.m2d.modeler
        assert isinstance(modeler, Modeler2D)

        # Round it for Ansys
        barrier_points = np.round(barrier_points, 6)

        # Potentially add the z axis
        if barrier_points.shape[1] == 2:
            barrier_points = np.hstack((barrier_points, np.zeros((len(barrier_points), 1))))

        # Convert them into a string format and interpolate
        points_str = [[str(y) for y in x] for x in barrier_points]
        barrier_id = modeler.create_polyline(
            points=points_str, segment_type=segment_type, cover_surface=True, name="Barrier"
        )

        # Remove the barrier
        self.rotor_id.subtract(barrier_id)
        modeler.delete(barrier_id)
