#!/bin/bash
#SBATCH --job-name=gen3v500
#SBATCH --partition=main
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --licenses=electronics_desktop:10
#SBATCH --time=16:00:00
#SBATCH --output=results/gen3_500w_v500/gen3-%j.log
#SBATCH --open-mode=append
# VOLTAGE-SENSITIVITY run: identical to the 400 V run EXCEPT V_MAX=500 V. Fully separate
# directory (results/gen3_500w_v500) and own FEA pool -- NOT shared with the 400 V run
# (results/gen3_500w_v). Same seed (0) so the only difference is the voltage limit -> a
# controlled A/B for the sensitivity figure. Demands/speeds/machine all unchanged.
cd ~/Public/MachineDesign-5f || exit 1
mkdir -p results/gen3_500w_v500
export ANSYSEM_ROOT242=/data/Ansys/v242/Linux64/
export OMP_NUM_THREADS=1 MKL_THREADING_LAYER=SEQUENTIAL KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONPATH=notebooks
./venv_5f/bin/python notebooks/gen3.py \
  --out results/gen3_500w_v500 --demands 4,8,2.5 --speeds 25,16,63 --v-max 500 --fhz 50 \
  --n-seed 64 --n-rounds 8 --q 8 --n-paths 256 --n-gcand 256 --n-icand 512 \
  --n-workers 8 --ncores 1 --slots 60 --phases 5 --seed 0
