#!/bin/bash
#SBATCH --job-name=rotor-sweep
#SBATCH --partition=main
#SBATCH --array=0-19%8                        # 20 parameter values; at most 8 run at once
#SBATCH --cpus-per-task=1                      # one FEA solve = one core
#SBATCH --mem=6G                               # per-task cap (raise if your solve needs more)
#SBATCH --licenses=electronics_desktop:1       # one AEDT session per task
#SBATCH --time=2:00:00
#SBATCH --output=results/sweep/task-%a.log
# ---------------------------------------------------------------------------------------
# GENERIC rotor-sensitivity sweep TEMPLATE (job array). Each SLURM_ARRAY_TASK_ID solves ONE
# parameter value as an independent 1-core / --mem / 1-license job; SLURM runs them
# concurrently up to the shared budget. This is the standard pattern for any parameter
# sweep on bayes -- copy it and adapt the two marked lines to your model.
#   submit:  sbatch bayes-queue/sweep.sbatch
#   watch:   squeue -u $USER        cancel: scancel <jobid>   (or <jobid>_<k> for one task)
#   collect: results/sweep/task-*.json
# ---------------------------------------------------------------------------------------
set -euo pipefail
cd ~/Public/MachineDesign-5f
mkdir -p results/sweep
export ANSYSEM_ROOT242=/data/Ansys/v242/Linux64/
export PYTHONPATH=notebooks
export OMP_NUM_THREADS=1 MKL_THREADING_LAYER=SEQUENTIAL KMP_DUPLICATE_LIB_OK=TRUE

# --- (1) define the swept parameter (e.g. a rotor flux-barrier angle over its range) ----
PMIN=0.10; PMAX=0.35; N=20
VAL=$(python3 -c "print($PMIN + ($PMAX-$PMIN)*$SLURM_ARRAY_TASK_ID/($N-1))")

# --- (2) the single-point evaluation: one geometry variant -> torque/ripple/flux --------
#     REPLACE with your model's one-solve entry point (must accept the swept value and
#     write its own result file). It should open its own AEDT session and do ONE solve.
srun ./venv_5f/bin/python notebooks/solve_one.py \
     --param "$VAL" --out "results/sweep/task-${SLURM_ARRAY_TASK_ID}.json"
