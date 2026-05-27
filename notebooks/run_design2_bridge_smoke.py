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
    evaluate_current_point,
    solve_min_current_for_torque,
)


aedt_version = "2025.1"
project_name = "Design2_smoke_adapter"
design_name = "Design01"
num_cores = 4

# [Id1, Iq1, Id3, Iq3], unit = A
CURRENT_SETPOINT = (0.0, 5.0, 1e-6, 0.0)

NC_VALUE = "113"
NPER = "1"
POINT_PER = "201"

SPEED_RPM_FOR_BRIDGE_TEST = 0.0

OPEN_AEDT_GUI = False
KEEP_PROJECT_OPEN = False
DELETE_ROTOR_AT_END = True


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
    "Moving1.Torque",
]


def fmt_array(x) -> str:
    return np.array2string(
        np.asarray(x, dtype=float),
        precision=6,
        suppress_small=False,
    )


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
            non_graphical=not OPEN_AEDT_GUI,
            new_desktop=True,
            close_on_exit=False,
        )

        design.add_rotor()

        design.solution_expressions = SAFE_SOLUTION_EXPRESSIONS

        design.m2d.variable_manager["Nc"] = NC_VALUE
        design.m2d.variable_manager["Nper"] = NPER
        design.m2d.variable_manager["PointPer"] = POINT_PER

        Id1, Iq1, Id3, Iq3 = CURRENT_SETPOINT

        print("\nDesign2 bridge smoke test")
        print("-------------------------")
        print(f"AEDT file: {file_name_aedt}")
        print(f"Current setpoint [A]: ({Id1}, {Iq1}, {Id3}, {Iq3})")
        print(f"Settings: Nc={NC_VALUE}, Nper={NPER}, PointPer={POINT_PER}")

        out = design.compute(
            Id1,
            Iq1,
            Id3,
            Iq3,
            NUM_CORES=num_cores,
        )

        if out is None:
            print("\nFAILED: no result returned from Design2.compute().")
            return

        res_raw = output_to_raw_dict(design.solution_expressions, out)
        result = build_design2_si_result(res_raw)

        print("\nDesign2 result")
        print("--------------")
        print(f"current_dq [A] = {fmt_array(result.current_dq)}")
        print(f"voltage_dq [V] = {fmt_array(result.voltage_dq)}")
        print(f"Maxwell torque = {result.torque_nm:.8g} Nm")

        # Use zero fixed flux for this interface smoke test.
        # Flux_e is not yet verified as a fixed PM/excitation flux.
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

        eval_initial = evaluate_current_point(
            bridge,
            result.current_dq,
            speed_rpm=SPEED_RPM_FOR_BRIDGE_TEST,
        )

        I_new, success = solve_min_current_for_torque(
            bridge,
            target_torque_nm=result.torque_nm,
            speed_rpm=SPEED_RPM_FOR_BRIDGE_TEST,
            initial_guess=result.current_dq,
        )

        print("\nTereza current_setpoints")
        print("------------------------")
        print(f"initial model torque = {float(eval_initial['torque_nm']):.8g} Nm")
        print(f"optimizer success    = {success}")
        print(f"I_new [A]            = {fmt_array(I_new)}")

        if success:
            eval_new = evaluate_current_point(
                bridge,
                I_new,
                speed_rpm=SPEED_RPM_FOR_BRIDGE_TEST,
            )
            print(f"new model torque     = {float(eval_new['torque_nm']):.8g} Nm")
            print(f"new current peak     = {float(eval_new['curr_peak_A']):.8g} A")
            print(f"new voltage peak     = {float(eval_new['volt_peak_V']):.8g} V")

        print("\nBridge smoke test completed.")
        print("Note: I_new verifies the interface only; it is not yet a Maxwell-validated optimum.")

        if OPEN_AEDT_GUI and KEEP_PROJECT_OPEN:
            input("Press Enter when ready to cleanup...")

    finally:
        if design is not None:
            if DELETE_ROTOR_AT_END:
                try:
                    design.delete_rotor()
                except Exception as exc:
                    print(f"Warning: failed to delete rotor: {exc}")

            if not KEEP_PROJECT_OPEN:
                try:
                    design.close_project()
                except Exception as exc:
                    print(f"Warning: failed to close AEDT project: {exc}")


if __name__ == "__main__":
    main()