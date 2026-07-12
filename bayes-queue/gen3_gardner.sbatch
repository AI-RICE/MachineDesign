#!/bin/bash
#SBATCH --job-name=gen3gard
#SBATCH --partition=main
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --licenses=electronics_desktop:10
#SBATCH --time=6:00:00
#SBATCH --output=results/gen3_500w_gardner/gen3-%j.log
#SBATCH --open-mode=append
# GARDNER constrained-EHVI turns-free enrichment run. Seeded from the 400V+500V union
# (581 pts / 208 geoms, pre-written). The outer acquisition is now proper constrained EHVI:
#   alpha(g) = mean_paths[ 1{feasible} * HVI(obj | confirmed feasible front) ]  = EHVI * P(feasible)
# (Gardner 2014), MC-estimated on Matheron/Thompson sample paths -- the feasibility mask is used,
# not discarded. Lew corrected to 2.4 mH (machine spec) via --lew. ~3 rounds x q=4 ~= 30 new FEA
# to ENRICH the confirmed turns-free front. Separate dir so nothing mixes with prior runs.
cd ~/Public/MachineDesign-5f || exit 1
mkdir -p results/gen3_500w_gardner
export ANSYSEM_ROOT242=/data/Ansys/v242/Linux64/
export OMP_NUM_THREADS=1 MKL_THREADING_LAYER=SEQUENTIAL KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONPATH=notebooks
./venv_5f/bin/python notebooks/gen3.py \
  --out results/gen3_500w_gardner --demands 4,8,2.5 --speeds 25,16,63 \
  --v-max 400 --lew 0.0024 --turns-free --nc-base 113 --fhz 50 \
  --n-seed 64 --n-rounds 3 --q 4 --n-paths 256 --n-gcand 256 --n-icand 512 \
  --n-workers 8 --ncores 1 --slots 60 --phases 5 --seed 1
