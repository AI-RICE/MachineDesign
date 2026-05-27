from __future__ import annotations

from dataclasses import dataclass

import numpy as np


CURRENT_KEYS = ["I_d1", "I_q1", "I_d3", "I_q3"]
VOLTAGE_KEYS = ["V_d1", "V_q1", "V_d3", "V_q3"]
FLUX_E_KEYS = ["Flux_e_d1", "Flux_e_q1", "Flux_e_d3", "Flux_e_q3"]

INDUCTANCE_KEYS = [
    "Ld1", "Ld1q1", "Ld1d3", "Ld1q3",
    "Lq1", "Lq1d3", "Lq1q3",
    "Ld3", "Ld3q3",
    "Lq3",
]


@dataclass(frozen=True)
class Design2SIResult:
    raw: dict[str, float]
    si: dict[str, float]

    current_dq: np.ndarray   # A, [Id1, Iq1, Id3, Iq3]
    voltage_dq: np.ndarray   # V, [Vd1, Vq1, Vd3, Vq3]
    flux_e: np.ndarray       # Wb, [Flux_e_d1, Flux_e_q1, Flux_e_d3, Flux_e_q3]
    Ls: np.ndarray           # H, 4x4, rows/cols [d1, q1, d3, q3]
    torque_nm: float         # Nm


def output_to_raw_dict(solution_expressions: list[str], out) -> dict[str, float]:
    values = np.asarray(out, dtype=float)

    if len(solution_expressions) != values.size:
        raise ValueError(
            f"Length mismatch: {len(solution_expressions)} expressions, "
            f"but output has {values.size} values."
        )

    return {
        key: float(value)
        for key, value in zip(solution_expressions, values)
    }


def _require_keys(res: dict[str, float], keys: list[str]) -> None:
    missing = [key for key in keys if key not in res]
    if missing:
        raise KeyError(f"Missing required Design2 result keys: {missing}")


def convert_raw_to_si(res_raw: dict[str, float]) -> dict[str, float]:
    """
    Convert AEDT raw output to SI units.

    Current interpretation:
        I_dq raw: mA -> A
        V_dq raw: V -> V
        L_dq raw: nH -> H
        Flux_e raw: Wb -> Wb
        Moving1.Torque: Nm -> Nm
    """
    _require_keys(
        res_raw,
        CURRENT_KEYS + VOLTAGE_KEYS + FLUX_E_KEYS + INDUCTANCE_KEYS,
    )

    res_si = dict(res_raw)

    for key in CURRENT_KEYS:
        res_si[key] = res_raw[key] * 1e-3

    for key in VOLTAGE_KEYS:
        res_si[key] = res_raw[key]

    for key in FLUX_E_KEYS:
        res_si[key] = res_raw[key]

    for key in INDUCTANCE_KEYS:
        res_si[key] = res_raw[key] * 1e-9

    if "Moving1.Torque" in res_raw:
        res_si["Moving1.Torque"] = res_raw["Moving1.Torque"]

    if "Torque_dq" in res_raw:
        res_si["Torque_dq"] = res_raw["Torque_dq"]

    return res_si


def build_Ls(res_si: dict[str, float]) -> np.ndarray:
    return np.array(
        [
            [res_si["Ld1"],    res_si["Ld1q1"],  res_si["Ld1d3"],  res_si["Ld1q3"]],
            [res_si["Ld1q1"],  res_si["Lq1"],    res_si["Lq1d3"],  res_si["Lq1q3"]],
            [res_si["Ld1d3"],  res_si["Lq1d3"],  res_si["Ld3"],    res_si["Ld3q3"]],
            [res_si["Ld1q3"],  res_si["Lq1q3"],  res_si["Ld3q3"],  res_si["Lq3"]],
        ],
        dtype=float,
    )


def build_design2_si_result(res_raw: dict[str, float]) -> Design2SIResult:
    res_si = convert_raw_to_si(res_raw)

    return Design2SIResult(
        raw=res_raw,
        si=res_si,
        current_dq=np.array([res_si[k] for k in CURRENT_KEYS], dtype=float),
        voltage_dq=np.array([res_si[k] for k in VOLTAGE_KEYS], dtype=float),
        flux_e=np.array([res_si[k] for k in FLUX_E_KEYS], dtype=float),
        Ls=build_Ls(res_si),
        torque_nm=float(res_si.get("Moving1.Torque", np.nan)),
    )


# =============================================================================
# Minimal print helpers
# Keep these names so existing notebooks do not break.
# =============================================================================

def print_raw_result(result: Design2SIResult) -> None:
    # Intentionally quiet.
    return


def print_si_result(result: Design2SIResult) -> None:
    # Intentionally quiet.
    return


def print_current_optimization_inputs(result: Design2SIResult) -> None:
    print("\nDesign2 SI summary")
    print("------------------")
    print(f"current_dq [A] = {result.current_dq}")
    print(f"voltage_dq [V] = {result.voltage_dq}")
    print(f"torque_nm      = {result.torque_nm:.8g}")
    print(f"diag(Ls) [H]   = {np.diag(result.Ls)}")


def print_consistency_checks(result: Design2SIResult, pole_pairs: int = 2) -> None:
    # Intentionally quiet. Keep this function only for notebook compatibility.
    return


def print_warnings(result: Design2SIResult) -> None:
    problems = []

    if not np.all(np.isfinite(result.current_dq)):
        problems.append("current_dq contains non-finite values.")

    if not np.all(np.isfinite(result.voltage_dq)):
        problems.append("voltage_dq contains non-finite values.")

    if not np.all(np.isfinite(result.flux_e)):
        problems.append("flux_e contains non-finite values.")

    if not np.all(np.isfinite(result.Ls)):
        problems.append("Ls contains non-finite values.")

    if not np.isfinite(result.torque_nm):
        problems.append("torque_nm is not finite.")

    if problems:
        print("\nWarnings")
        print("--------")
        for item in problems:
            print(f"- {item}")