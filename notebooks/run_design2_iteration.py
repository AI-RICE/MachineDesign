from __future__ import annotations

import csv
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

aedt_version = "2025.1"
project_name = "Design2_smoke_adapter"
design_name = "Design01"
num_cores = 4
Max_Iter = 10 # maximum number of iterations
Conv_Tol = 1e-4 # convergence tolerance

# [Id1, Iq1, Id3, Iq3], unit = A
# norminal current is 1.3A, maximum is 2.6A
CURRENT_SETPOINT = (0.0, 1.3, 1e-6, 0.0)

NC_VALUE = "113"
NPER = "1"
POINT_PER = "201"
# Speed (rpm)
SPEED_RPM_FOR_BRIDGE_TEST = 1500.0


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

CSV_COLUMNS = [
    "iteration",
    "Id1", "Iq1", "Id3", "Iq3",
    "torque_nm",
    "flux_e_d1", "flux_e_q1", "flux_e_d3", "flux_e_q3",
    "Ld1", "Lq1", "Ld3", "Lq3",
    "delta_I",
    "optimizer_success",
]

#avoid atan2(0,0)
def regularize_currents(curr, eps: float = 1e-6):
    Id1, Iq1, Id3, Iq3 = [float(x) for x in curr]

    if abs(Id1) < eps and abs(Iq1) < eps:
        Id1 = eps

    if abs(Id3) < eps and abs(Iq3) < eps:
        Id3 = eps

    return Id1, Iq1, Id3, Iq3


def make_row(iteration, result, delta_I=None, optimizer_success=None):
    I = result.current_dq
    fe = result.flux_e
    Ls = result.Ls  # 4x4, diagonal = [Ld1, Lq1, Ld3, Lq3]
    return {
        "iteration": iteration,
        "Id1": I[0], "Iq1": I[1], "Id3": I[2], "Iq3": I[3],
        "torque_nm": result.torque_nm,
        "flux_e_d1": fe[0], "flux_e_q1": fe[1], "flux_e_d3": fe[2], "flux_e_q3": fe[3],
        "Ld1": Ls[0, 0], "Lq1": Ls[1, 1], "Ld3": Ls[2, 2], "Lq3": Ls[3, 3],
        "delta_I": delta_I,
        "optimizer_success": optimizer_success,
    }


def main() -> None:
    path_data = os.path.join(os.getcwd(), "data")
    os.makedirs(path_data, exist_ok=True)
    file_name_aedt = os.path.join(path_data, f"{project_name}.aedt")
    file_name_csv = os.path.join(path_data, "iteration_results.csv")

    design = None
    rows = []

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

        out = design.compute(Id1, Iq1, Id3, Iq3, NUM_CORES=num_cores)

        if out is None:
            return

        res_raw = output_to_raw_dict(design.solution_expressions, out)
        result = build_design2_si_result(res_raw)

        # print(f"Maxwell torque = {result.torque_nm:.8g} Nm")
        # print(f"flux_e from Ansys = {result.flux_e}")

        rows.append(make_row(0, result))

        torque_target = result.torque_nm
        I_current = result.current_dq

        for i in range(Max_Iter):
            bridge = build_current_setpoint_bridge(
                result,
                R_stat_ohm=19,
                n_ppairs=2,
                curr_max_A=2.6,
                volt_max_V=230 * np.sqrt(2),
                omega_max_rpm=1500.0,
                add_volt_0=False,
                flux_override=np.zeros(4),
            )

            I_new, success = solve_min_current_for_torque(
                bridge,
                target_torque_nm=torque_target,
                speed_rpm=SPEED_RPM_FOR_BRIDGE_TEST,
                initial_guess=I_current,
            )

            if not success or np.any(np.isnan(I_new)):
                break

            delta_I = np.linalg.norm(I_new - I_current)
            if delta_I < Conv_Tol:
                break

            Id1_new, Iq1_new, Id3_new, Iq3_new = regularize_currents(I_new)
            out = design.compute(Id1_new, Iq1_new, Id3_new, Iq3_new, NUM_CORES=num_cores)

            if out is None:
                break

            res_raw = output_to_raw_dict(design.solution_expressions, out)
            result = build_design2_si_result(res_raw)
            I_current = result.current_dq

            rows.append(make_row(i + 1, result, delta_I=delta_I, optimizer_success=success))

        # print(f"Final Ansys torque = {result.torque_nm:.8g} Nm")
        # print(f"Final I_current [A] = {I_current}")

    finally:
        if rows:
            with open(file_name_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)

        if design is not None:
            try:
                design.delete_rotor()
            except Exception:
                pass

            try:
                design.close_project()
            except Exception:
                pass

if __name__ == "__main__":
    main()