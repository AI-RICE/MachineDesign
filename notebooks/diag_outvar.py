"""Isolate which token in the Design2 terminal-voltage output vars Maxwell rejects.

Loads the already-built SynRM5f_torque project (fast, no geometry rebuild) and
tries creating candidate output variables one by one, reporting OK/FAIL.
"""
import os

from machine_design.design2 import Design2

fn = os.path.join(os.getcwd(), "data", "SynRM5f_torque.aedt")
d = Design2.load(fn, version="2024.2", non_graphical=True, new_desktop=False, close_on_exit=True)
m2d = d.m2d

tests = {
    "diag_induced_only": "InducedVoltage(PhaseA)",
    "diag_ind_plus_ri": "InducedVoltage(PhaseA) + Rstat*InputCurrent(PhaseA)",
    "diag_ddt_only": "ddt(InputCurrent(PhaseA))",
    "diag_lew_ddt": "Lew*ddt(InputCurrent(PhaseA))",
}
for name, expr in tests.items():
    try:
        ok = m2d.create_output_variable(name, expr)
        print(f"RESULT {name}: ok={ok}  <- {expr}")
    except Exception as e:
        print(f"RESULT {name}: EXC {type(e).__name__}: {e}  <- {expr}")

d.close_project()
print("[diag] done")
