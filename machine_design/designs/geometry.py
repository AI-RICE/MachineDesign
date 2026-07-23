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

    def mm_to_str(self, var, field) -> float:
        val = getattr(self, var)[field]
        if not val.endswith("mm"):
            raise Exception("val must end with mm")
        return float(val[:-2])

    def delete_motion_setup(self, m2d: Maxwell2d) -> None:
        module = m2d.odesign.GetModule("ModelSetup")
        existing = list(module.GetMotionSetupNames())
        if existing:
            module.DeleteMotionSetup(existing)

    def assign_motion_setup(self, m2d: Maxwell2d) -> None:
        m2d.assign_rotate_motion(
            assignment="Band",
            coordinate_system="Global",
            axis="Z",
            positive_movement=True,
            start_position="InitPos",
            angular_velocity="RotSpeed",
            has_rotation_limits=False,
        )

    def _set_appearance(self, obj, color, transparency: float) -> None:
        obj.color = color
        obj.transparency = transparency

    def set_appearances(self, m2d: Maxwell2d) -> None:
        for item in [self.shaft_id, self.region_id, self.band_id]:
            self._set_appearance(item, (0, 255, 255), 0.95)
        self._set_appearance(self.stator_id, (192, 192, 192), 0.0)
        for name in self.id_coils:
            self._set_appearance(m2d.modeler[name], (255, 128, 0), 0.0)
        self._set_appearance(self.rotor_id, (192, 192, 192), 0.0)

    def build_rotor(self, m2d: Maxwell2d) -> None:
        modeler = m2d.modeler
        assert isinstance(modeler, Modeler2D)

        rotor_id = modeler.create_polyline(points=self.rot_points, segment_type=["Arc", "Line", "Arc"], cover_surface=True, name="Rotor")
        self.rotor_id = rotor_id
        rotor_id.material_name = self.Fe

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

    def _push_stator_variables(self, m2d: Maxwell2d) -> None:
        modeler = m2d.modeler
        assert isinstance(modeler, Modeler2D)

        modeler.model_units = "mm"
        for k, v in self.geom_params.items():
            m2d[k] = v
        for k, v in self.wind_params.items():
            m2d[k] = v
        for k, v in self.slot_params.items():
            m2d[k] = v
        for k, v in self.mod_params.items():
            m2d[k] = v

    def _build_vacuum_regions(self, m2d: Maxwell2d):
        modeler = m2d.modeler
        assert isinstance(modeler, Modeler2D)

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

        # Fit all view
        modeler.fit_all()

        return shaft_id, region_id, band_id

    def _split_for_symmetry(self, m2d: Maxwell2d, object_list) -> None:
        modeler = m2d.modeler
        assert isinstance(modeler, Modeler2D)

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

    def _assign_boundary_conditions(self, m2d: Maxwell2d) -> None:
        modeler = m2d.modeler
        assert isinstance(modeler, Modeler2D)

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
