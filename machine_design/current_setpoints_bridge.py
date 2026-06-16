from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from current_setpoints.parameters import BaseMachine, ConstantFlux
from current_setpoints.simulation import Transform
from current_setpoints.optimization import ModelAnalytical, MotorOptimizer

from .design2_adapter import Design2SIResult


class Design2CurrentSetpointMachine(BaseMachine):
    def __init__( self, L_stat: np.ndarray, *, R_stat_ohm: float = 0.19, n_ppairs: int = 2, ) -> None:
        # Initialize the machine with the given parameters obtained from the Design2 results (Ansys).
        # * means the following parameters must be used adding their names, e.g. R_stat_ohm=0.19
        self.n_phases = 5
        self.n_ppairs = int(n_ppairs)

        L_stat = np.asarray(L_stat, dtype=float)
        if L_stat.shape != (4, 4):
            raise ValueError(f"L_stat must be 4x4. Got shape {L_stat.shape}.")

        self.L_stat = L_stat # Ls in Eq.(4)
        self.R_stat = float(R_stat_ohm) * np.eye(4) # R_stat in Eq.(4)

        super().__init__()


@dataclass # put @dataclass to automatically generate init. (machine, flux, transform, model, optimizer).
class CurrentSetpointBridge:
    machine: Design2CurrentSetpointMachine
    flux: ConstantFlux
    transform: Transform
    model: ModelAnalytical
    optimizer: MotorOptimizer


def mechanical_rpm_to_electrical_rad_per_sec( speed_rpm: float, n_ppairs: int, ) -> float:
    # Convert mechanical speed in RPM to electrical angular velocity in radians per second.
    # omega_e = speed_rpm * 2.0 * np.pi / 60.0 * n_ppairs, in Eq. (4)
    return float(speed_rpm) * 2.0 * np.pi / 60.0 * int(n_ppairs)


def build_current_setpoint_bridge(
    result: Design2SIResult, #results from SI conversion of Design2 (Ansys), containing Ls, flux_e, etc.
    *, #* means the following parameters must be used adding their names, e.g. R_stat_ohm=0.19.
    R_stat_ohm: float = 0.19,
    n_ppairs: int = 2,
    curr_max_A: float = 30.0, #current limit in A, Eq. (11c)
    volt_max_V: float = 13.0, #voltage limit in V, Eq. (11d)
    omega_max_rpm: float = 1800.0, #max electrical speed in RPM
    add_volt_0: bool = False, #with/without ZSC, Eq. (9) vs0, true with ZSC, false without ZSC, vs0=0.
    n_theta: int = 700, #number of theta points, discretization
    solver_opts: dict[str, Any] | None = None, #optimizer settings, use defaults if None.
    flux_override: np.ndarray | None = None, #override the flux vector, Eq. (5), use Design2 flux if None.
    use_design2_flux_e: bool = False, #use Design2 flux_e if True, otherwise use zeros, only used if flux_override is None.
) -> CurrentSetpointBridge:
    machine = Design2CurrentSetpointMachine(
        result.Ls,
        R_stat_ohm=R_stat_ohm,
        n_ppairs=n_ppairs,
    )
    #wrap design2 machine parameters into Tereza's BaseMachine format.

    machine.set_max_pars(
        curr_max=curr_max_A,
        volt_max=volt_max_V,
        omega_max=omega_max_rpm,
    )
    #match constraints in Eq. (11c), (11d), and max speed in Tereza's machines.py

    if flux_override is not None:
        flux_vec = np.asarray(flux_override, dtype=float)
    elif use_design2_flux_e:
        flux_vec = np.asarray(result.flux_e, dtype=float)
    else:
        flux_vec = np.zeros(4, dtype=float)
    #use Design2 flux_e if True, or zeros if False, or flux_override is provided. Eq.(5)

    if flux_vec.shape != (4,):
        raise ValueError(f"flux vector must have shape (4,), got {flux_vec.shape}.")
    #check if flux_vec has shape (4,), otherwise raise ValueError.

    flux = ConstantFlux(
        ieee_flux_volt=flux_vec,
        ieee_flux_torq=flux_vec,
    )
    #use the same ΨPM vector for both voltage and torque, as in Eq.(4) and Eq.(7).

    transform = Transform(
        machine=machine,
        flux=flux,
        add_volt_0=add_volt_0,
        n_theta=n_theta,
    )
    #defined by Tereza's Transform class in transform.py, 
    #Convert dq setpoints to phase waveforms and check current/voltage peaks.
    #constraints in Eq. (11c), (11d) are for phase currents and voltages.

    model = ModelAnalytical(
        machine=machine,
        flux=flux,
    )
    #calculate torque from curr_dq using machine and flux. Eq.(11b), defined in Tereza's models.py.

    if solver_opts is None:
        solver_opts = {
            "disp": False, #display optimization progress
            "ftol": 1e-8, #tolerance for stopping optimization
            "maxiter": 500, #maximum number of iterations
        }
    #defined in Tereza's optimizer.py, 
    #used for optimization settings, can be overridden by user.

    optimizer = MotorOptimizer(
        model=model,
        opts=solver_opts,
    )
    #defined in Tereza's optimizer.py, 
    #Create optimizer; actual current minimization is done by optimizer.minimize_current().

    return CurrentSetpointBridge(
        machine=machine,
        flux=flux,
        transform=transform,
        model=model,
        optimizer=optimizer,
    )
    #Return all current_setpoints objects as one bridge container.

def solve_min_current_for_torque(
    bridge: CurrentSetpointBridge,
    *,
    target_torque_nm: float,
    speed_rpm: float,
    initial_guess: np.ndarray | None = None, #initial guess for current setpoints, [I_d1, I_q1, I_d3, I_q3]
) -> tuple[np.ndarray, bool]: #obtain I_new and success flag from optimization
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
    #compute with the minize_current function in Tereza's MotorOptimizer, defined in optimizer.py

    return np.asarray(I_new, dtype=float), bool(success)
#Target torque + speed -> optimized dq current I_new using Tereza model.

def solve_max_torque(
    bridge: CurrentSetpointBridge,
    *,
    speed_rpm: float,
    initial_guess: np.ndarray | None = None, #initial guess for current setpoints, [I_d1, I_q1, I_d3, I_q3]
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
    #compute with the max_torque function in Tereza's MotorOptimizer, defined in optimizer.py

    return np.asarray(I_new, dtype=float), float(torque_nm), bool(success)
#Maximize torque for given speed -> optimized dq current I_new and max torque using Tereza model.