#!/bin/bash
#SBATCH --job-name=gen3
#SBATCH --partition=main
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --licenses=electronics_desktop:10
#SBATCH --time=16:00:00
#SBATCH --output=results/gen3_500w_v/gen3-%j.log
#SBATCH --open-mode=append
# gen-3 driver (voltage-constrained) as one SLURM job: whole process tree (main + FEA workers
# + torch-isolated select subprocess + geometry-gen AEDT) in one cgroup capped at --mem. A
# runaway is killed in-cgroup, never the box. Resumes from results/gen3_500w_v/gen3.npz.
# Voltage limit at ALL 3 operating points: P1/P2/P3 = 25/16/63 Hz elec, V_MAX=400 V, Lew=2.4mH.
cd ~/Public/MachineDesign-5f || exit 1
mkdir -p results/gen3_500w_v
export ANSYSEM_ROOT242=/data/Ansys/v242/Linux64/
export OMP_NUM_THREADS=1 MKL_THREADING_LAYER=SEQUENTIAL KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONPATH=notebooks
./venv_5f/bin/python notebooks/gen3.py \
  --out results/gen3_500w_v --demands 4,8,2.5 --speeds 25,16,63 --v-max 400 --fhz 50 \
  --n-seed 64 --n-rounds 8 --q 8 --n-paths 256 --n-gcand 256 --n-icand 512 \
  --n-workers 8 --ncores 1 --slots 60 --phases 5 --seed 0
