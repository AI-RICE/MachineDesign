"""HPC-license probe: solve ONE design at --num-cores in an ISOLATED AEDT desktop
+ fresh project, so several can run concurrently. Used to settle whether a
1-core Maxwell-2D solve draws HPC_PARALLEL (cap=4 total) or not (cap=#seats).

Reads the HPC checkout from licdebug afterwards (see probe orchestration).

  ~/Public/PFN/venv_newparam/bin/python notebooks/probe_hpc.py \
      --aedt-project data/probe_A.aedt --num-cores 1 --tag A
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from machine_design import load_design  # noqa: E402
from machine_design.generators import HacklGenerator_OneLambda  # noqa: E402
from machine_design.geometry import analyze_results  # noqa: E402
from machine_design.lumped.geometry import REFERENCE_MACHINE  # noqa: E402

ROTOR, AIRGAP = 0.5, 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aedt-project", required=True)
    ap.add_argument("--aedt-version", default="2024.2")
    ap.add_argument("--num-cores", type=int, default=1)
    ap.add_argument("--tag", default="A")
    ap.add_argument("--designs-npz", default="notebooks/step3_designs.npz")
    args = ap.parse_args()

    print(f"[{args.tag}] pid={os.getpid()} cores={args.num_cores} project={args.aedt_project}",
          flush=True)
    dd = np.load(args.designs_npz, allow_pickle=True)
    # first OneLambda design in the file
    i = list(dd["shorts"]).index("OneLambda") if "OneLambda" in list(dd["shorts"]) else 0
    X = np.asarray(dd["Xs"][i], float)

    hk = HacklGenerator_OneLambda(REFERENCE_MACHINE, r_stator_end=0.7, offset=0.35)
    hk.set_parameters(hk.X_to_params(X))
    bars = hk.split_barriers(hk.generate_barriers())

    design = load_design(args.aedt_project, f"probe_{args.tag}", "Design01",
                         args.aedt_version, new_desktop=True)
    design.add_rotor()
    for b in bars:
        design.add_rotor_barrier(b)
    t0 = time.time()
    print(f"[{args.tag}] SOLVE START {time.strftime('%H:%M:%S')}", flush=True)
    Tor = design.compute(args.num_cores, mesh_length=ROTOR, airgap_mesh=AIRGAP)
    dt = time.time() - t0
    Tm, _, Tr = analyze_results(np.asarray(Tor, float))
    print(f"[{args.tag}] SOLVE DONE  {time.strftime('%H:%M:%S')}  T_mean={Tm:.3f} "
          f"ripple={Tr:.2f}% in {dt:.0f}s", flush=True)
    design.delete_rotor()
    design.close_project()


if __name__ == "__main__":
    main()
