# Running a parameter grid across parallel Ansys sessions

This is the parallel Ansys computation pattern originally in
`machine_design/parallel_calculation/` (added by Jan Laksar), moved here in
issue #21 along with the two scripts (`runner_parallel.py`,
`calculate_combination.py`). The large `SynRM_orig.aedt` example project was
dropped since it does not belong in git. Running these scripts requires your
own base `.aedt` project.

## Problem

A single Ansys Electronics Desktop session can only run one design at a time,
so sweeping a grid of operating points (for example Id/Iq current setpoints)
sequentially in one session is slow. Running several independent Desktop
sessions in parallel, each solving its own share of the grid, cuts wall time
roughly by the number of workers.

## Method

1. Split the full grid of tasks into `N_WORKERS` contiguous chunks.
2. For each worker, copy the base `.aedt` project file to its own file before
   opening it. Ansys locks a project file while it is open, so every worker
   needs its own copy rather than sharing one file.
3. Start `multiprocessing.Pool(N_WORKERS)` and run one worker function per
   chunk. Each worker opens its own `Desktop(non_graphical=True,
   new_desktop=True, close_on_exit=False)` and `Maxwell2d` instance against
   its private project copy, so the workers do not interfere with each
   other's Ansys session.
4. Inside a worker, solve each task with `analyze_setup(setup_name,
   use_auto_settings=False, cores=1, tasks=1)`, pinning each solve to a single
   core. Several workers are already running concurrently, so letting each one
   grab multiple cores would oversubscribe the machine.
5. Extract the quantities of interest with
   `post.get_solution_data(expressions=..., primary_sweep_variable="Time")`,
   then average over `data_real(expr)[:-1]` (dropping the last point, which
   repeats the first point of the periodic waveform).
6. The main process collects every worker's results and writes them out
   once, as both a plain CSV and an HDF5 file.

## Example

`runner_parallel.py` swept a 16x16 grid of `Id`/`Iq` values (0.0 to 3.2 A
in 0.2 A steps) and recorded flux linkages, inductances, and torque for
each point, with `calculate_combination.py` doing the per-point solve and
extraction described above.
