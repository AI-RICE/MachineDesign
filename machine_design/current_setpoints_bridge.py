from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from current_setpoints.parameters import BaseMachine, ConstantFlux
from current_setpoints.simulation import Transform
from current_setpoints.optimization import ModelAnalytical, MotorOptimizer

from .design2_adapter import Design2SIResult


class Design2CurrentSetpointMachine(BaseMachine):
    def __init__(
        self,
        L_stat: np.ndarray,
        *,
        R_stat_ohm: float = 0.19,
        n_ppairs: int = 2,
    ) -> None:
        self.n_phases = 5
        self.n_ppairs = int(n_ppairs)

        L_stat = np.asarray(L_stat, dtype=float)
        if L_stat.shape != (4, 4):
            raise ValueError(f"L_stat must be 4x4. Got shape {L_stat.shape}.")

        self.L_stat = L_stat
        self.R_stat = float(R_stat_ohm) * np.eye(4)

        super().__init__()


@dataclass
class CurrentSetpointBridge:
    machine: Design2CurrentSetpointMachine
    flux: ConstantFlux
    transform: Transform
    model: ModelAnalytical
    optimizer: MotorOptimizer


def mechanical_rpm_to_electrical_rad_per_sec(
    speed_rpm: float,
    n_ppairs: int,
) -> float:
    return float(speed_rpm) * 2.0 * np.pi / 60.0 * int(n_ppairs)


def build_current_setpoint_bridge(
    result: Design2SIResult,
    *,
    R_stat_ohm: float = 0.19,
    n_ppairs: int = 2,
    curr_max_A: float = 30.0,
    volt_max_V: float = 13.0,
    omega_max_rpm: float = 1800.0,
    add_volt_0: bool = False,
    n_theta: int = 700,
    solver_opts: dict[str, Any] | None = None,
    flux_override: np.ndarray | None = None,
    use_design2_flux_e: bool = False,
) -> CurrentSetpointBridge:
    machine = Design2CurrentSetpointMachine(
        result.Ls,
        R_stat_ohm=R_stat_ohm,
        n_ppairs=n_ppairs,
    )

    machine.set_max_pars(
        curr_max=curr_max_A,
        volt_max=volt_max_V,
        omega_max=omega_max_rpm,
    )

    if flux_override is not None:
        flux_vec = np.asarray(flux_override, dtype=float)
    elif use_design2_flux_e:
        flux_vec = np.asarray(result.flux_e, dtype=float)
    else:
        flux_vec = np.zeros(4, dtype=float)

    if flux_vec.shape != (4,):
        raise ValueError(f"flux vector must have shape (4,), got {flux_vec.shape}.")

    flux = ConstantFlux(
        ieee_flux_volt=flux_vec,
        ieee_flux_torq=flux_vec,
    )

    transform = Transform(
        machine=machine,
        flux=flux,
        add_volt_0=add_volt_0,
        n_theta=n_theta,
    )

    model = ModelAnalytical(
        machine=machine,
        flux=flux,
    )

    if solver_opts is None:
        solver_opts = {
            "disp": False,
            "ftol": 1e-8,
            "maxiter": 500,
            "eps": 1e-8,
        }

    optimizer = MotorOptimizer(
        model=model,
        opts=solver_opts,
    )

    return CurrentSetpointBridge(
        machine=machine,
        flux=flux,
        transform=transform,
        model=model,
        optimizer=optimizer,
    )


def evaluate_current_point(
    bridge: CurrentSetpointBridge,
    curr_dq: np.ndarray,
    *,
    speed_rpm: float,
) -> dict[str, float | np.ndarray]:
    curr_dq = np.asarray(curr_dq, dtype=float)
    omega_e = mechanical_rpm_to_electrical_rad_per_sec(
        speed_rpm,
        bridge.machine.n_ppairs,
    )

    torque_nm = bridge.model.calculate_torque(
        omega=omega_e,
        curr_dq=curr_dq,
    )

    curr_peak, curr_ang_diff, volt_peak, volt_ang_diff = bridge.transform.get_max_vals(
        omega=omega_e,
        curr_dq=curr_dq,
    )

    volt_dq = bridge.transform.get_volt_dq(
        omega=omega_e,
        curr_dq=curr_dq,
    )

    return {
        "omega_e_rad_s": omega_e,
        "torque_nm": torque_nm,
        "curr_peak_A": curr_peak,
        "volt_peak_V": volt_peak,
        "curr_ang_diff": curr_ang_diff,
        "volt_ang_diff": volt_ang_diff,
        "volt_dq_V": volt_dq,
    }


def solve_min_current_for_torque(
    bridge: CurrentSetpointBridge,
    *,
    target_torque_nm: float,
    speed_rpm: float,
    initial_guess: np.ndarray | None = None,
) -> tuple[np.ndarray, bool]:
    omega_e = mechanical_rpm_to_electrical_rad_per_sec(
        speed_rpm,
        bridge.machine.n_ppairs,
    )

    I_new, success = bridge.optimizer.minimize_current(
        torq_target=float(target_torque_nm),
        omega=omega_e,
        transform=bridge.transform,
        vec_curr_guess=None if initial_guess is None else np.asarray(initial_guess, dtype=float),
    )

    return np.asarray(I_new, dtype=float), bool(success)


def solve_max_torque(
    bridge: CurrentSetpointBridge,
    *,
    speed_rpm: float,
    initial_guess: np.ndarray | None = None,
) -> tuple[np.ndarray, float, bool]:
    omega_e = mechanical_rpm_to_electrical_rad_per_sec(
        speed_rpm,
        bridge.machine.n_ppairs,
    )

    I_new, torque_nm, success = bridge.optimizer.maximize_torque(
        omega=omega_e,
        transform=bridge.transform,
        vec_curr_guess=None if initial_guess is None else np.asarray(initial_guess, dtype=float),
    )

    return np.asarray(I_new, dtype=float), float(torque_nm), bool(success)