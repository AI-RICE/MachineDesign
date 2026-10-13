"""
investigate torque/voltage dq identities
Run with: 
    python scripts/dq_identity_sweep.py
"""
import os

import subprocess
import sys
import time

import numpy as np

from machine_design.config import load_config
from machine_design.designs.design import Design
from machine_design.optimization.generators import HacklGenerator_OneLambda
from motors.synrm_5f_40s import Computation
from motors.synrm_5f_40s import Geometry as Geometry40s
from motors.synrm_5f_60s import Geometry as Geometry60s

AEDT_VERSION = load_config()["aedt_version"]
R_STATOR_END = 0.7
OFFSET = 0.35
SEED = 42
NUM_CORES = 1
current_setpoint = (0.8, 0.9, 0.2, -0.15)
geometries = {"40s": Geometry40s, "60s": Geometry60s}

configs = [
    ("40s", 201),
    ("40s", 401),
    ("40s", 601),
    ("40s", 801),
    ("40s", 1201),
    ("40s", 2001),
    ("40s", 3001),
    ("60s", 201),
    ("60s", 401),
    ("60s", 601),
    ("60s", 801),
    ("60s", 1201),
    ("60s", 2001),
    ("60s", 3001),
]

timeout_s = 2 * 3600
results_path = "results/dq_identity_sweep.csv"
os.makedirs("results", exist_ok=True)
header = "geometry,pointper,time,torque,h1_q,h1_d,h3_q,h3_d"


def _feasible_barriers(geometry):
    dummy = Design(m2d=None, geometry=geometry, computation=Computation(geometry))
    generator = HacklGenerator_OneLambda(dummy, R_STATOR_END, offset=OFFSET)

    np.random.seed(SEED)
    while True:
        generator.set_parameters(generator.random_parameters())
        barriers = generator.split_barriers(generator.generate_barriers())
        if generator.feasible_barriers(barriers):
            return barriers


def rel_err(actual, expected):
    return abs(actual - expected) / abs(expected)


def append_row(*fields):
    with open(results_path, "a") as f:
        f.write(",".join(str(v) for v in fields) + "\n")


def measure_one(geometry_name, pointper):
    class ComputationMultiPeriod(Computation):
        def set_oper_params(self):
            super().set_oper_params()
            self.oper_params["Nper"] = "1"
            self.oper_params["PointPer"] = str(pointper)

    geometry = geometries[geometry_name]()
    computation = ComputationMultiPeriod(geometry)
    barriers = _feasible_barriers(geometry)

    start = time.time()
    design = None
    try:
        design = Design.create(
            f"DqIdentitySweep_{geometry_name}_{pointper}",
            "Design01",
            f"dq_identity_{geometry_name}_{pointper}.aedt",
            geometry,
            computation,
            version=AEDT_VERSION,
            non_graphical=True,
            new_desktop=False,
            close_on_exit=False,
        )
        design.add_rotor()
        for barrier in barriers:
            design.add_rotor_barrier(barrier)
        out = design.compute(*current_setpoint, NUM_CORES=NUM_CORES)
        assert out is not None, "solve returned no results"

        errs = {"torque": rel_err(np.mean(out["Torque_dq"]), np.mean(out["Moving1.Torque"]))}

        for h in (1, 3):
            w = h * computation.w
            d, q = f"d{h}", f"q{h}"
            vl_q = out[f"V_{q}"] - computation.Rstat * out[f"I_{q}"]
            vl_d = out[f"V_{d}"] - computation.Rstat * out[f"I_{d}"]
            errs[f"h{h}_q"] = rel_err(np.mean(vl_q), np.mean(w * out[f"Flux_{d}"]))
            errs[f"h{h}_d"] = rel_err(np.mean(vl_d), np.mean(-w * out[f"Flux_{q}"]))
    finally:
        if design is not None:
            design.close_project()

    elapsed = time.time() - start
    append_row(geometry_name, pointper, f"{elapsed:.0f}", errs["torque"], errs["h1_q"], errs["h1_d"], errs["h3_q"], errs["h3_d"])


def sweep():
    append_row(header)
    for geometry_name, pointper in configs:
        try:
            proc = subprocess.run([sys.executable, __file__, geometry_name, str(pointper)], timeout=timeout_s)
            if proc.returncode != 0:
                append_row(geometry_name, pointper, "-", "crash", "-", "-", "-", "-")
        except subprocess.TimeoutExpired:
            append_row(geometry_name, pointper, "-", "timeout", "-", "-", "-", "-")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        measure_one(sys.argv[1], int(sys.argv[2]))
    else:
        sweep()
