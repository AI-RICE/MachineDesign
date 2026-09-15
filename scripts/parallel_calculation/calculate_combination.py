import numpy as np


def main(m2d, task_chunk, setup_name):

    def run_simulation(Id, Iq):
        m2d.variable_manager["Id"] = f"{Id}A"
        m2d.variable_manager["Iq"] = f"{Iq}A"

        m2d.save_project()
        m2d.analyze_setup(setup_name, use_auto_settings=False, cores=1, tasks=1)

        expressions = ["I_d1", "I_q1", "Flux_d1", "Flux_q1", "Flux_e_d1", "Flux_e_q1", "Ld1", "Lq1", "Ld1q1", "Lq1d1", "Moving1.Torque"]

        sols = m2d.post.get_solution_data(expressions=expressions, primary_sweep_variable="Time")

        out = np.zeros(len(expressions))

        for i, expr in enumerate(expressions):
            data = sols.data_real(expr)[:-1]
            val = float(np.mean(data))

            if expr.startswith("L"):
                val /= 1e9

            out[i] = val

        return out

    # --- chunk processing ---
    results_local = []
    total = len(task_chunk)
    for k, (Id, Iq, i, j) in enumerate(task_chunk, start=1):
        print(f"[INFO] {k}/{total} | Id={Id:.3f} A | Iq={Iq:.3f} A")

        res = run_simulation(Id, Iq)

        results_local.append((i, j, res))

    return results_local
