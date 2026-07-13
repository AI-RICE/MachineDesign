from abc import ABC, abstractmethod

import numpy as np
from ansys.aedt.core import Maxwell2d
from ansys.aedt.core.modeler.modeler_2d import Modeler2D


class GeometryBase(ABC):
    def __init__(self) -> None:
        self.set_iron()
        self.set_geom_params()
        self.set_slot_params()
        self.set_winds_params()
        self.set_mod_params()
        self.set_rot_points()
        self.set_derived_params()
        self.set_udp_par_list_stator()

    @abstractmethod
    def set_iron(self): ...

    @abstractmethod
    def set_geom_params(self): ...

    @abstractmethod
    def set_slot_params(self): ...

    @abstractmethod
    def set_winds_params(self): ...

    @abstractmethod
    def set_mod_params(self): ...

    @abstractmethod
    def set_rot_points(self): ...

    @abstractmethod
    def set_derived_params(self): ...

    @abstractmethod
    def set_udp_par_list_stator(self): ...

    @abstractmethod
    def build_stator(self, m2d: Maxwell2d) -> None: ...

    @abstractmethod
    def build_rotor(self, m2d: Maxwell2d) -> None: ...

    def mm_to_str(self, var, field) -> float:
        val = getattr(self, var)[field]
        if not val.endswith("mm"):
            raise Exception("val must end with mm")
        return float(val[:-2])

    def add_rotor_barrier(self, m2d: Maxwell2d, barrier_points, segment_type=None) -> None:
        modeler = m2d.modeler
        assert isinstance(modeler, Modeler2D)

        # Round it for Ansys
        barrier_points = np.round(barrier_points, 6)

        # Potentially add the z axis
        if barrier_points.shape[1] == 2:
            barrier_points = np.hstack((barrier_points, np.zeros((len(barrier_points), 1))))

        # Convert them into a string format and interpolate
        points_str = [[str(y) for y in x] for x in barrier_points]
        barrier_id = modeler.create_polyline(points=points_str, segment_type=segment_type, cover_surface=True, name="Barrier")

        # Remove the barrier
        self.rotor_id.subtract(barrier_id)
        modeler.delete(barrier_id)

    def delete_rotor(self, m2d: Maxwell2d) -> None:
        assert isinstance(m2d.modeler, Modeler2D)
        m2d.modeler.delete(self.rotor_id)


class Geometry(GeometryBase):
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
        self.rotor_r_min = self.mm_to_str("geom_params", "DiaShaft") / 2
        self.rotor_r_max = self.mm_to_str("geom_params", "DiaStatorGap") / 2 - self.mm_to_str("geom_params", "Airgap")

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

    def build_stator(self, m2d: Maxwell2d) -> None:
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
        stator_id = modeler.create_udp(
            dll="RMxprt/SlotCore.dll",
            parameters=self.udp_par_list_stator,
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
        coil_id.duplicate_around_axis(axis="Z", angle="360deg/SlotNumber", clones="SlotNumber/Poles", create_new_objects=True)
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

        self.stator_id = stator_id
        self.band_id = band_id
        self.id_coils = id_coils

    def build_rotor(self, m2d: Maxwell2d) -> None:
        modeler = m2d.modeler
        assert isinstance(modeler, Modeler2D)

        rotor_id = modeler.create_polyline(points=self.rot_points, segment_type=["Arc", "Line", "Arc"], cover_surface=True, name="Rotor")
        self.rotor_id = rotor_id
        rotor_id.material_name = self.Fe
        rotor_id.color = (192, 192, 192)  # rgb
        rotor_id.transparency = 0.0


class Geometry2(Geometry):
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

    def set_derived_params(self):
        pass
