import numpy as np
from ansys.aedt.core import Maxwell2d
from ansys.aedt.core.modeler.modeler_2d import Modeler2D

from motors.synrm_3f_36s import Computation as Computation3f36s
from motors.synrm_3f_36s import Geometry as Geometry3f36s

__all__ = ["Geometry", "Computation"]

# for now only one template is used - PMCore


class Geometry(Geometry3f36s):
    def __init__(self) -> None:
        super().__init__()
        self.set_magnets()
        self.set_rotor_params()
        self.magnet_ids = []

    def set_magnets(self):
        self.magnet = "NdFe35"

    def set_rotor_params(self):
        self.rotor_params = {
            "PoleType": "5",
            "Embrace": "0.8",
            "ThickMag": "4mm",
            "WidthMag": "28mm",
            "Offset": "0mm",
            "Bridge": "1mm",
            "Rib": "0mm",
        }

    def _udp_par_list_rotor(self, info_core: str) -> list[list[str]]:
        return [
            ["DiaGap", "DiaStatorGap-2*Airgap"],
            ["DiaYoke", "DiaShaft"],
            ["Length", "0mm"],
            ["Poles", "Poles"],
            *[[name, f"Rotor{name}"] for name in self.rotor_params],
            ["LenRegion", "0mm"],
            ["InfoCore", info_core],
        ]

    def build_rotor(self, m2d: Maxwell2d) -> None:
        modeler = m2d.modeler
        assert isinstance(modeler, Modeler2D)

        for k, v in self.rotor_params.items():
            m2d[f"Rotor{k}"] = v

        rotor_id = modeler.create_udp(dll="RMxprt/PMCore.dll", parameters=self._udp_par_list_rotor("0"), library="syslib", name="Rotor")
        modeler.purge_history([rotor_id.name])

        magnets_id = modeler.create_circle(origin=[0, 0, 0], radius="DiaStatorGap/2-Airgap", is_covered=True, name="Magnet")
        shaft_tmp = modeler.create_circle(origin=[0, 0, 0], radius="DiaShaft/2", is_covered=True, name="ShaftTmp")
        modeler.subtract(blank_list=[magnets_id], tool_list=[shaft_tmp], keep_originals=False)
        modeler.subtract(blank_list=[magnets_id], tool_list=[rotor_id], keep_originals=True)
        magnet_ids = modeler.separate_bodies(assignment=magnets_id.name) or [magnets_id]

        slivers = [m.name for m in magnet_ids if m.faces[0].area < 1e-3]
        if slivers:
            modeler.delete(slivers)
        magnet_ids = [m for m in magnet_ids if m.name not in slivers]

        x, y, _ = magnet_ids[0].faces[0].center
        pole_pitch = 360 / (2 * self.PolePairs)
        delta = (pole_pitch / 2 - np.degrees(np.arctan2(y, x))) % pole_pitch
        names = [rotor_id.name] + [m.name for m in magnet_ids]

        if not np.isclose(delta, 0.0) and not np.isclose(delta, pole_pitch):
            modeler.rotate(assignment=names, axis="Z", angle=f"{delta}deg")

        modeler.set_working_coordinate_system("Section")
        modeler.split(assignment=names, plane="ZX", sides="NegativeOnly")
        modeler.set_working_coordinate_system("Global")
        modeler.split(assignment=names, plane="ZX", sides="PositiveOnly")

        rotor_id.material_name = self.Fe
        self.rotor_id = rotor_id
        self.magnet_ids = [modeler[m.name] for m in magnet_ids if m.name in modeler.object_names]

        for mag in self.magnet_ids:
            mx, my, _ = mag.faces[0].center
            self.assign_magnet_cs(m2d, mag.name, float(np.degrees(np.arctan2(my, mx))) + 180)
            mag.material_name = self.magnet
            mag.solve_inside = True
            self._set_appearance(mag, (255, 0, 0), 0.0)

        modeler.set_working_coordinate_system("Global")
        self.reassign_motion_setup(m2d)

    def reassign_motion_setup(self, m2d: Maxwell2d) -> None:
        motions = [b.name for b in m2d.boundaries if b.type == "Band"]
        if motions:
            m2d.omodelsetup.DeleteMotionSetup(motions)
        self._assign_rotor_motion(m2d)

    def delete_rotor(self, m2d: Maxwell2d) -> None:
        names = [m.name for m in self.magnet_ids]
        if names:
            m2d.modeler.delete(names)
        super().delete_rotor(m2d)
        for cs in [cs for cs in m2d.modeler.coordinate_systems if cs.name in {f"CS_{n}" for n in names}]:
            cs.delete()
        self.magnet_ids = []


class Computation(Computation3f36s):
    pass
