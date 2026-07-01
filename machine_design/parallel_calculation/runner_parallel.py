import os
import shutil
import numpy as np
import h5py
import time
from multiprocessing import Pool

from ansys.aedt.core import Desktop, Maxwell2d

import calculate_combination

# =========================================================
# CONFIGURATION
# =========================================================
AEDT_VERSION = os.getenv('AEDT_VERSION', '2026.1')

PATH = "d:\\DATA\\Test"
PROJECT_DIR = os.getenv('ANSYS_PROJECT_DIR', PATH)
BASE_PROJECT = "SynRM_orig"
DESIGN_NAME = "Design02_def_Idq"
SETUP_NAME = "Setup1"

N_WORKERS = 22 # less then number of logical processors

# =========================================================
# PROJECT COPY
# =========================================================
def copy_project(worker_id):
    src = os.path.join(PROJECT_DIR, BASE_PROJECT + ".aedt")
    dst_name = f"SynRM_worker_{worker_id}.aedt"
    dst = os.path.join(PROJECT_DIR, dst_name)

    shutil.copyfile(src, dst)

    return dst_name

# =========================================================
# WORKER
# =========================================================
def worker(args):

    worker_id, task_chunk = args

    project_name = copy_project(worker_id)

    desktop = Desktop(
        specified_version=AEDT_VERSION,
        non_graphical=True,
        new_desktop=True,
        close_on_exit=False
    )

    app = Maxwell2d(
        project=project_name,
        design=DESIGN_NAME
    )

    results = calculate_combination.main(app, task_chunk, SETUP_NAME)

    desktop.release_desktop(close_projects=True, close_on_exit=False)

    return results


# =========================================================
# MAIN
# =========================================================
def run():

    Id_vec = np.arange(0.0, 3.2, 0.2) #16 values
    Iq_vec = np.arange(0.0, 3.2, 0.2) #16 values

    tasks = []
    for i, Id in enumerate(Id_vec):
        for j, Iq in enumerate(Iq_vec):
            tasks.append((Id, Iq, i, j))

    chunk_size = int(np.ceil(len(tasks) / N_WORKERS))
    chunks = [tasks[i:i + chunk_size] for i in range(0, len(tasks), chunk_size)]

    args = [(i, chunk) for i, chunk in enumerate(chunks)]

    results = np.zeros((len(Id_vec), len(Iq_vec), 11))

    with Pool(N_WORKERS) as pool:
        outputs = pool.map(worker, args)

    rows = []
    for chunk in outputs:
        for i, j, res in chunk:
            results[i, j, :] = res
            rows.append(list(results[i, j, :]))
    data_out = np.array(rows)

    header = "Id,Iq,Flux_d,Flux_q,Flux_e_d,Flux_e_q,L_d,L_q,L_dq,L_qd,Torque"
    np.savetxt(
        "results_grid.csv",
        data_out,
        delimiter=",",
        header=header,
        comments=""
    )

    with h5py.File("results_grid.h5", "w") as f:
        f.create_dataset("Id", data=Id_vec)
        f.create_dataset("Iq", data=Iq_vec)
        f.create_dataset("results", data=results)

    print("[INFO] DONE")

    return results


# =========================================================
# ENTRY POINT
# =========================================================
if __name__ == "__main__":
    start = time.perf_counter()
    
    run()
    
    end = time.perf_counter()

    print(f"Time of calculation: {end - start:.6f} s")