import numpy as np
from ansys.aedt.core import Desktop, Maxwell2d
from ansys.aedt.core.modeler.modeler_2d import Modeler2D


class Design:
    def __init__(self, m2d: Maxwell2d) -> None:
        self.set_parameters()
        self.m2d = m2d

    @classmethod
    def create(cls, project_name: str, design_name: str, file_name: str, **kwargs) -> "Design":
        m2d = Maxwell2d(project=project_name, design=design_name, solution_type="TransientXY", **kwargs)
        obj = cls(m2d)
        obj.create_stator()
        obj.save_project(file_name)
        return obj

    @classmethod
    def load(cls, file_name: str, **kwargs) -> "Design":
        desktop = Desktop(**kwargs)
        desktop.load_project(file_name)

        m2d = Maxwell2d()
        m2d.set_active_design("Design01")

        return cls(m2d)

    def set_parameters(self) -> None:
        # materials
        self.Fe = "Cogent Power - M350-50A, B-H at 50Hz"

        # main definitions
        self.geom_params = {
            "DiaStatorGap": "79mm",
            "DiaStatorYoke": "125mm",
            "Airgap": "0.225mm",
            "SlotNumber": "36",
            "SlotType": "3",
            "DiaShaft": "25mm",
            "StackLength": "85mm",
        }
        # stator slot
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
        # winding
        self.wind_params = {
            "Layers": "1",
            "ParallelPaths": "1",
            "CoilPitch": "9",  # coil pitch in slots
            "SlotLiner": "0.3mm",
            "SpaceLayers": "0.2mm",
            "Nc": "68",  # turns per coil
        }
        # model parameters
        PolePairs = 2
        f = 50  # [Hz]
        RotSpeed = 60 * f / PolePairs  # [rpm]
        self.mod_params = {
            "Poles": f"2*{PolePairs}",
            "ModelLength": "85mm",
            "SymmetryFactor": "Poles",
            "StatorSkewAngle": "0deg",
        }
        # operation parameters
        self.oper_params = {
            "Im": "1.5*sqrt(2)A",
            "epsI": "pi/4",  # current angle
            "InitPos": "-30deg",
            "f": f"{f}Hz",
            "RotSpeed": f"{RotSpeed}rpm",
            "Nper": "1/6",  # number of included periods
            "PointPer": "101",  # number of time points per period
        }
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
        self.setup_name = "Setup1"
        self.rotor_r_min = self.mm_to_str("geom_params", "DiaShaft") / 2
        self.rotor_r_max = self.mm_to_str("geom_params", "DiaStatorGap") / 2 - self.mm_to_str("geom_params", "Airgap")

    def create_stator(self) -> None:
        m2d = self.m2d
        modeler = m2d.modeler
        assert isinstance(modeler, Modeler2D)

        # Define design variables from the created dictionaries.
        modeler.model_units = "mm"
        for k, v in self.geom_params.items():
            m2d[k] = v
        for k, v in self.wind_params.items():
            m2d[k] = v
        for k, v in self.slot_params.items():
            m2d[k] = v
        for k, v in self.mod_params.items():
            m2d[k] = v
        for k, v in self.oper_params.items():
            m2d[k] = v

        # VACUUM OBJECTS
        # Outer region
        region_id = modeler.create_circle(
            origin=[0, 0, 0],
            radius="DiaStatorYoke/2",
            # num_sides="SegAngle",
            is_covered=True,
            name="Region",
        )
        # Band
        band_id = modeler.create_circle(
            origin=[0, 0, 0],
            radius="(DiaStatorGap - (1.0 * Airgap))/2",
            # num_sides="mapping_angle",
            is_covered=True,
            name="Band",
        )
        # Shaft
        shaft_id = modeler.create_circle(
            origin=[0, 0, 0],
            radius="DiaShaft/2",
            is_covered=True,
            name="Shaft",
        )

        # motion setup
        m2d.assign_rotate_motion(
            assignment="Band",
            coordinate_system="Global",
            axis="Z",
            positive_movement=True,
            start_position="InitPos",
            angular_velocity="RotSpeed",
            has_rotation_limits=False,
        )

        # Together
        vacuum_obj_id = [
            shaft_id,
            region_id,
            band_id,
        ]  # put shaft first
        for item in vacuum_obj_id:
            item.color = (0, 255, 255)
            item.transparency = 0.95

        # Fit all view
        modeler.fit_all()

        # Stator geometry
        udp_par_list_stator = [
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
        stator_id = modeler.create_udp(
            dll="RMxprt/SlotCore.dll",
            parameters=udp_par_list_stator,
            library="syslib",
            name="Stator",
            # SolveInside="True",
        )
        # Stator properties
        stator_id.material_name = self.Fe
        stator_id.color = (192, 192, 192)  # rgb
        stator_id.transparency = 0.0

        # Winding
        coil_id = modeler.create_rectangle(
            origin=["DiaStatorGap/2+Hs0+Hs1+SlotLiner", "-(Bs1/2-SlotLiner)", 0],
            sizes=["(Hs2+Rs/2-2*SlotLiner)/Layers-(Layers-1)*SpaceLayers/2", "Bs1-2*SlotLiner", 0],
            name="Coil",
            material="copper",
        )
        coil_id.color = (255, 128, 0)
        coil_id.transparency = 0.0
        modeler.rotate(assignment=coil_id, axis="Z", angle="360deg/SlotNumber/2")
        coil_id.duplicate_around_axis(axis="Z", angle="360deg/SlotNumber", clones="CoilPitch", create_new_objects=True)
        id_coils = modeler.get_objects_w_string(string_name="Coil", case_sensitive=True)

        # Create section of machine
        object_list = [stator_id] + vacuum_obj_id
        modeler.create_coordinate_system(
            origin=[0, 0, 0],
            reference_cs="Global",
            name="Section",
            mode="axis",
            x_pointing=["cos(360deg/SymmetryFactor)", "sin(360deg/SymmetryFactor)", 0],
            y_pointing=["-sin(360deg/SymmetryFactor)", "cos(360deg/SymmetryFactor)", 0],
        )
        modeler.set_working_coordinate_system("Section")
        modeler.split(assignment=object_list, plane="ZX", sides="NegativeOnly")
        modeler.set_working_coordinate_system("Global")
        modeler.split(assignment=object_list, plane="ZX", sides="PositiveOnly")

        # Symmetrical boundary conditions
        pos_1 = "DiaStatorGap/4"
        id_bc_1 = modeler.get_edgeid_from_position(position=[pos_1, 0, 0], assignment="Region")
        id_bc_2 = modeler.get_edgeid_from_position(
            position=[
                pos_1 + "*cos((360deg/SymmetryFactor))",
                pos_1 + "*sin((360deg/SymmetryFactor))",
                0,
            ],
            assignment="Region",
        )
        m2d.assign_master_slave(
            independent=id_bc_1,
            dependent=id_bc_2,
            reverse_master=True,
            reverse_slave=False,
            same_as_master=False,
            boundary="Symmetry",
        )
        # Zero vector potential
        pos_2 = "(DiaStatorYoke/2)"
        id_bc_az = modeler.get_edgeid_from_position(
            position=[
                pos_2 + "*cos((360deg/SymmetryFactor/2))",
                pos_2 + "*sin((360deg/SymmetryFactor)/2)",
                0,
            ],
            assignment="Region",
        )
        m2d.assign_vector_potential(assignment=id_bc_az, vector_value=0, boundary="A0")

        # Excitations
        I_A = "Im * cos(2*pi*f*time+epsI)"
        I_B = "Im * cos(2*pi*f*time-120deg+epsI)"
        I_C = "Im * cos(2*pi*f*time-240deg+epsI)"

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
            polarity="Positive",
            name="CS2",
        )
        m2d.assign_coil(
            assignment=["Coil_2"],
            conductors_number="Nc",
            polarity="Positive",
            name="CS3",
        )
        m2d.assign_winding(
            assignment=None,
            winding_type="Current",
            is_solid=False,
            current=I_A,
            parallel_branches="ParallelPaths",
            name="PhaseA",
        )
        m2d.add_winding_coils(assignment="PhaseA", coils=["CS1", "CS2", "CS3"])
        m2d.assign_coil(
            assignment=["Coil_6"],
            conductors_number="Nc",
            polarity="Positive",
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
        m2d.assign_winding(
            assignment=None,
            winding_type="Current",
            is_solid=False,
            current=I_B,
            parallel_branches="ParallelPaths",
            name="PhaseB",
        )
        m2d.add_winding_coils(assignment="PhaseB", coils=["CS7", "CS8", "CS9"])
        m2d.assign_coil(
            assignment=["Coil_3"],
            conductors_number="Nc",
            polarity="Negative",
            name="CS4",
        )
        m2d.assign_coil(
            assignment=["Coil_4"],
            conductors_number="Nc",
            polarity="Negative",
            name="CS5",
        )
        m2d.assign_coil(
            assignment=["Coil_5"],
            conductors_number="Nc",
            polarity="Negative",
            name="CS6",
        )
        m2d.assign_winding(
            assignment=None,
            winding_type="Current",
            is_solid=False,
            current=I_C,
            parallel_branches="ParallelPaths",
            name="PhaseC",
        )
        m2d.add_winding_coils(assignment="PhaseC", coils=["CS4", "CS5", "CS6"])

        # Mesh operation
        m2d.mesh.assign_length_mesh(
            assignment=id_coils,
            inside_selection=True,
            maximum_length=3,
            maximum_elements=None,
            name="coils",
        )
        m2d.mesh.assign_length_mesh(
            assignment=stator_id,
            inside_selection=True,
            maximum_length=3,
            maximum_elements=None,
            name="stator",
        )

        # core loss
        m2d.set_core_losses("Stator", core_loss_on_field=False)

        # inductance calculation
        m2d.change_inductance_computation(compute_transient_inductance=True, incremental_matrix=False)
        # model depth
        m2d.model_depth = "StackLength"
        # symmetry
        m2d.change_symmetry_multiplier("SymmetryFactor")
        # Calculation setup
        setup = m2d.create_setup(name=self.setup_name)
        setup.props["StopTime"] = "Nper/f"
        setup.props["TimeStep"] = "1/(f*(PointPer-1))"
        setup.props["SaveFieldsType"] = "None"
        setup.props["OutputPerObjectCoreLoss"] = False
        setup.props["OutputPerObjectSolidLoss"] = True
        setup.props["OutputError"] = True
        setup.update()
        m2d.validate_simple()

        # ooutput variables
        output_vars = {
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
        for k, v in output_vars.items():
            m2d.create_output_variable(k, v)

        # Definitions for plots
        post_params = {  # reports
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
        # Create Report
        for k, v in post_params.items():
            expressions = list(k) if isinstance(k, tuple) else [k]  # if multiple report, use list(k). Else, use k
            m2d.post.create_report(
                expressions=expressions,
                setup_sweep_name="",
                domain="Sweep",
                variations=None,
                primary_sweep_variable="Time",
                secondary_sweep_variable=None,
                report_category=None,
                plot_type="Rectangular Plot",
                context=None,
                subdesign_id=None,
                polyline_points=1001,
                plot_name=v,
            )

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

    def compute(self, NUM_CORES: int = 1):
        m2d = self.m2d
        assert m2d.mesh is not None
        assert m2d.post is not None

        m2d.mesh.assign_length_mesh(
            assignment=self.rotor_id,
            inside_selection=True,
            maximum_length=3,
            maximum_elements=None,
            name="rotor",
        )
        # core loss rotor
        m2d.set_core_losses("Rotor", core_loss_on_field=False)

        # Analyze
        m2d.analyze_setup(self.setup_name, use_auto_settings=False, cores=NUM_CORES)

        solutions = m2d.post.get_solution_data(expressions="Moving1.Torque", primary_sweep_variable="Time")
        try:
            return solutions.data_magnitude()
        except AttributeError:
            return None

    def delete_rotor(self) -> None:
        assert isinstance(self.m2d.modeler, Modeler2D)
        self.m2d.modeler.delete(self.rotor_id)

    def save_design(self, file_name: str, **kwargs) -> None:
        show = kwargs.pop("show", False)
        view = kwargs.pop("view", "xy")
        self.m2d.plot(show=show, output_file=file_name, view=view)

    def save_project(self, file_name: str | None = None) -> None:
        if file_name is None:
            self.m2d.save_project()
        else:
            self.m2d.save_project(file_name)

    def close_project(self) -> None:
        self.m2d.close_desktop()

    def mm_to_str(self, var, field) -> float:
        val = getattr(self, var)[field]
        if not val.endswith("mm"):
            raise Exception("val must end with mm")
        return float(val[:-2])
