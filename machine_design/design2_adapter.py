from __future__ import annotations

from dataclasses import dataclass

import numpy as np


CURRENT_KEYS = ["I_d1", "I_q1", "I_d3", "I_q3"] # I_s on page3
VOLTAGE_KEYS = ["V_d1", "V_q1", "V_d3", "V_q3"]
FLUX_E_KEYS = ["Flux_e_d1", "Flux_e_q1", "Flux_e_d3", "Flux_e_q3"]

INDUCTANCE_KEYS = [
    "Ld1", "Ld1q1", "Ld1d3", "Ld1q3",
    "Lq1", "Lq1d3", "Lq1q3",
    "Ld3", "Ld3q3",
    "Lq3",
]


@dataclass(frozen=True) #read only
class Design2SIResult:
    raw: dict[str, float] # AEDT raw output
    si: dict[str, float]  # SI-converted values

    current_dq: np.ndarray   # A, [Id1, Iq1, Id3, Iq3]
    voltage_dq: np.ndarray   # V, [Vd1, Vq1, Vd3, Vq3]
    flux_e: np.ndarray       # Wb, [Flux_e_d1, Flux_e_q1, Flux_e_d3, Flux_e_q3]
    Ls: np.ndarray           # H, 4x4, rows/cols [d1, q1, d3, q3]
    torque_nm: float         # Nm


def output_to_raw_dict(solution_expressions: list[str], out) -> dict[str, float]:
    # Convert output to a numpy array of floats
    # and convert to a dictionary with keys from solution_expressions.
    values = np.asarray(out, dtype=float)
    # convert output to a numpy array of floats

    if len(solution_expressions) != values.size:
       # Check for length mismatch and raise an error if it occurs
        raise ValueError(
            f"Length mismatch: {len(solution_expressions)} expressions, "
            f"but output has {values.size} values."
        )

    return {
        key: float(value)
        for key, value in zip(solution_expressions, values)
    }
    # Convert to a dictionary with keys from solution_expressions and values from the output

def _require_keys(res: dict[str, float], keys: list[str]) -> None:
    # Check that all required keys are present in the result dictionary.
    missing = [key for key in keys if key not in res]
    # find missing keys and raise an error if any are missing
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
    # Check that all required keys are present in the result dictionary.

    res_si = dict(res_raw)
    # create a new dictionary to store the SI values

    for key in CURRENT_KEYS:
        res_si[key] = res_raw[key] * 1e-3 
        # convert mA to A

    for key in VOLTAGE_KEYS:
        if key in res_raw:
            res_si[key] = res_raw[key]
        # convert V to V

    for key in FLUX_E_KEYS:
        res_si[key] = res_raw[key]
        # convert Wb to Wb

    for key in INDUCTANCE_KEYS:
        res_si[key] = res_raw[key] * 1e-9
        # convert nH to H

    if "Moving1.Torque" in res_raw:
        res_si["Moving1.Torque"] = res_raw["Moving1.Torque"]
        # convert Nm to Nm

    if "Torque_dq" in res_raw:
        res_si["Torque_dq"] = res_raw["Torque_dq"]
        # convert Nm to Nm

    return res_si


def build_Ls(res_si: dict[str, float]) -> np.ndarray:
    # Build the Ls 4*4 matrix from the SI values.
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
    # Convert AEDT raw output to SI units and build the Ls matrix.

    return Design2SIResult(
        # Create a Design2SIResult object with the raw and SI values.
        raw=res_raw,
        si=res_si,
        current_dq=np.array([res_si[k] for k in CURRENT_KEYS], dtype=float),
        voltage_dq=np.array([res_si.get(k, 0.0) for k in VOLTAGE_KEYS], dtype=float),
        flux_e=np.array([res_si[k] for k in FLUX_E_KEYS], dtype=float),
        Ls=build_Ls(res_si),
        torque_nm=float(res_si.get("Moving1.Torque", np.nan)),
    )
