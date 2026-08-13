from ansys.aedt.core import Maxwell2d
from ansys.aedt.core.modeler.modeler_2d import Modeler2D

from machine_design.designs.computation import ComputationBase
from machine_design.designs.geometry import GeometryBase


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
        self._push_stator_variables(m2d)
        shaft_id, region_id, band_id = self._build_vacuum_regions(m2d)
        Geometry.assign_motion_setup(m2d)
        stator_id = self._build_stator_core(m2d)
        id_coils = self._build_stator_coils(m2d)
        self._split_for_symmetry(m2d, [stator_id, shaft_id, region_id, band_id])
        self._assign_boundary_conditions(m2d)
        self._assign_stator_mesh(m2d, stator_id, id_coils)

        # core loss
        m2d.set_core_losses("Stator", core_loss_on_field=False)

        self.stator_id = stator_id
        self.shaft_id = shaft_id
        self.region_id = region_id
        self.band_id = band_id
        self.id_coils = id_coils

    def _build_stator_core(self, m2d: Maxwell2d):
        modeler = m2d.modeler
        assert isinstance(modeler, Modeler2D)

        stator_id = modeler.create_udp(
            dll="RMxprt/SlotCore.dll",
            parameters=self.udp_par_list_stator,
            library="syslib",
            name="Stator",
            # SolveInside="True",
        )
        stator_id.material_name = self.Fe

        return stator_id

    def _build_stator_coils(self, m2d: Maxwell2d):
        modeler = m2d.modeler
        assert isinstance(modeler, Modeler2D)

        coil_id = modeler.create_rectangle(
            origin=["DiaStatorGap/2+Hs0+Hs1+SlotLiner", "-(Bs1/2-SlotLiner)", 0],
            sizes=["(Hs2+Rs/2-2*SlotLiner)/Layers-(Layers-1)*SpaceLayers/2", "Bs1-2*SlotLiner", 0],
            name="Coil",
            material="copper",
        )
        modeler.rotate(assignment=coil_id, axis="Z", angle="360deg/SlotNumber/2")
        coil_id.duplicate_around_axis(axis="Z", angle="360deg/SlotNumber", clones="SlotNumber/Poles", create_new_objects=True)
        id_coils = modeler.get_objects_w_string(string_name="Coil", case_sensitive=True)

        return id_coils

    def _assign_stator_mesh(self, m2d: Maxwell2d, stator_id, id_coils) -> None:
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


class Computation(ComputationBase):
    def set_oper_params(self):
        f = 50  # [Hz]
        RotSpeed = 60 * f / self.geometry.PolePairs  # [rpm]
        self.oper_params = {
            "Im": "1.5*sqrt(2)A",
            "epsI": "pi/4",  # current angle
            "InitPos": "-30deg",
            "f": f"{f}Hz",
            "RotSpeed": f"{RotSpeed}rpm",
            "Nper": "1/6",  # number of included periods
            "PointPer": "101",  # number of time points per period
        }

    def set_solution_expressions(self):
        self.solution_expressions = "Moving1.Torque"

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

    def assign_stator_coils(self, m2d: Maxwell2d) -> None:
        # Excitations
        I_A = "Im * cos(2*pi*f*time+epsI)"
        I_B = "Im * cos(2*pi*f*time-120deg+epsI)"
        I_C = "Im * cos(2*pi*f*time-240deg+epsI)"

        phases_polarity = ["Positive", "Negative", "Positive"]
        phases_name = ["A", "C", "B"]
        phases_current = [I_A, I_C, I_B]
        i_coil = 0
        for phase_polarity, phase_name, phase_current in zip(phases_polarity, phases_name, phases_current):
            names = []
            for _ in range(3):
                m2d.assign_coil(
                    assignment=[self.geometry.id_coils[i_coil]],
                    conductors_number="Nc",
                    polarity=phase_polarity,
                    name=f"CS{i_coil + 1}",
                )
                names.append(f"CS{i_coil + 1}")
                i_coil += 1
            m2d.assign_winding(
                assignment=None,
                winding_type="Current",
                is_solid=False,
                current=phase_current,
                parallel_branches="ParallelPaths",
                name=f"Phase{phase_name}",
            )
            m2d.add_winding_coils(assignment=f"Phase{phase_name}", coils=names)

    def inductance_computation(self, m2d: Maxwell2d) -> None:
        m2d.change_inductance_computation(compute_transient_inductance=True, incremental_matrix=False)

    def set_variables(self, m2d: Maxwell2d, *args):
        pass

    def extract_results(self, solutions):
        return solutions.data_magnitude()