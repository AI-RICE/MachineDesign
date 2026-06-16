from __future__ import annotations

import os

import numpy as np

from machine_design import Design2, load_design
from machine_design.design2_adapter import (
    output_to_raw_dict,
    build_design2_si_result,
)
from machine_design.current_setpoints_bridge import (
    build_current_setpoint_bridge,
    solve_min_current_for_torque,
)


aedt_version = "2025.2"
project_name = "Design2_smoke_adapter"
design_name = "Design01"
num_cores = 4

# [Id1, Iq1, Id3, Iq3], unit = A
CURRENT_SETPOINT = (0.0, 5.0, 1e-6, 0.0)

NC_VALUE = "113"
NPER = "1"
POINT_PER = "201"
# Speed (rpm)
SPEED_RPM_FOR_BRIDGE_TEST = 0.0
# whether to send I_new to Ansys 
VERIFY_I_NEW_IN_ANSYS = True


SAFE_SOLUTION_EXPRESSIONS = [
    "V_d1", "V_q1", "V_d3", "V_q3",
    "Vind_d1", "Vind_q1", "Vind_d3", "Vind_q3",

    "Flux_d1", "Flux_q1", "Flux_d3", "Flux_q3",
    "Flux_e_d1", "Flux_e_q1", "Flux_e_d3", "Flux_e_q3",

    "I_d1", "I_q1", "I_d3", "I_q3",

    "Ld1", "Ld1q1", "Ld1d3", "Ld1q3",
    "Lq1", "Lq1d3", "Lq1q3",
    "Ld3", "Ld3q3",
    "Lq3",

    "Torque_dq",
    "Moving1.Torque", # Ansys Moving1.Torque
]

#avoid atan2(0,0)
def regularize_currents(curr, eps: float = 1e-6):
    Id1, Iq1, Id3, Iq3 = [float(x) for x in curr]

    if abs(Id1) < eps and abs(Iq1) < eps:
        Id1 = eps

    if abs(Id3) < eps and abs(Iq3) < eps:
        Id3 = eps

    return Id1, Iq1, Id3, Iq3

def main() -> None:
    path_data = os.path.join(os.getcwd(), "data")
    os.makedirs(path_data, exist_ok=True)
    file_name_aedt = os.path.join(path_data, f"{project_name}.aedt")

    design = None

    try:
        design = load_design(
            file_name_aedt,
            project_name,
            design_name,
            aedt_version,
            design_cls=Design2,
            non_graphical=True,
            new_desktop=True,
            close_on_exit=False,
        )

        design.add_rotor()

        design.solution_expressions = SAFE_SOLUTION_EXPRESSIONS

        design.m2d["Nc"] = NC_VALUE
        design.m2d["Nper"] = NPER
        design.m2d["PointPer"] = POINT_PER

        Id1, Iq1, Id3, Iq3 = CURRENT_SETPOINT

        out = design.compute( Id1,  Iq1, Id3, Iq3, NUM_CORES=num_cores)

        if out is None:
            print("FAILED: no result returned from Design2.compute().")
            return

        res_raw = output_to_raw_dict(design.solution_expressions, out)
        result = build_design2_si_result(res_raw)
        # real SI results obtained from Ansys

        print(f"Maxwell torque = {result.torque_nm:.8g} Nm")

        # Use zero fixed flux for this interface smoke test.
        # Flux_e is not verified as a fixed PM/excitation flux.
        bridge = build_current_setpoint_bridge(
            result,
            R_stat_ohm=0.19,
            n_ppairs=2,
            curr_max_A=30.0,
            volt_max_V=13.0,
            omega_max_rpm=1800.0,
            add_volt_0=False,
            flux_override=np.zeros(4),
            use_design2_flux_e=False,
        )
        #create bridge with results from SI results....,used bellow

        I_new, success = solve_min_current_for_torque(
            bridge,
            target_torque_nm=result.torque_nm,
            speed_rpm=SPEED_RPM_FOR_BRIDGE_TEST,
            initial_guess=result.current_dq,
        )

        print(f"optimizer success = {success}")
        print(f"I_new [A] = {I_new}")

        if success and VERIFY_I_NEW_IN_ANSYS:

            Id1_new, Iq1_new, Id3_new, Iq3_new = regularize_currents(I_new)

            out_new = design.compute( Id1_new, Iq1_new, Id3_new, Iq3_new, NUM_CORES=num_cores, )
            # cpoute with I_new in Ansys, verify torque and currents.

            if out_new is None:
                print("Ansys verification failed: no result returned for I_new.")
            else:
                res_raw_new = output_to_raw_dict(design.solution_expressions, out_new)
                result_new = build_design2_si_result(res_raw_new)
                print(f"Ansys I_new torque = {result_new.torque_nm:.8g} Nm")

    finally:
        if design is not None:
                try:
                    design.delete_rotor()
                except Exception as exc:
                    print(f"Warning: failed to delete rotor: {exc}")

                try:
                    design.close_project()
                except Exception as exc:
                    print(f"Warning: failed to close AEDT project: {exc}")


if __name__ == "__main__":
    main()