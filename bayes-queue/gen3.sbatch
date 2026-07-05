#!/bin/bash
#SBATCH --job-name=gen3
#SBATCH --partition=main
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --licenses=electronics_desktop:10
#SBATCH --time=12:00:00
#SBATCH --output=results/gen3_500w/gen3-%j.log
#SBATCH --open-mode=append
# gen-3 driver as a single SLURM job: the whole process tree (main + 8 FEA workers + the
# torch-isolated select subprocess + the geometry-gen AEDT session) runs inside ONE cgroup
# capped at --mem=32G. A runaway is killed in-cgroup, never the box (proven). The epilog +
# cgroup proctrack reap any ANSYS strays on exit. Resumes from results/gen3_500w/gen3.npz.
# Submit:  sbatch bayes-queue/gen3.sbatch      Watch: tail -f results/gen3_500w/gen3-<jobid>.log
cd ~/Public/MachineDesign-5f || exit 1
export ANSYSEM_ROOT242=/data/Ansys/v242/Linux64/
export OMP_NUM_THREADS=1 MKL_THREADING_LAYER=SEQUENTIAL KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONPATH=notebooks
./venv_5f/bin/python notebooks/gen3.py \
  --out results/gen3_500w --demands 4,8,6 --fhz 50 \
  --n-seed 64 --n-rounds 8 --q 8 --n-paths 256 --n-gcand 256 --n-icand 512 \
  --n-workers 8 --ncores 1 --slots 60 --phases 5 --seed 0
