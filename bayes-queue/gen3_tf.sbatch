#!/bin/bash
#SBATCH --job-name=gen3tf
#SBATCH --partition=main
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --licenses=electronics_desktop:10
#SBATCH --time=8:00:00
#SBATCH --output=results/gen3_500w_tf/gen3-%j.log
#SBATCH --open-mode=append
# TURNS-FREE run: seeded from the 400V+500V union (581 pts, pre-written by prep_tf.py), V_MAX=400,
# speeds 25/16/63. Winding turns are a free per-design analytic transformer (feasible iff one Nc
# satisfies all points' I and V limits; V~Nc, I~1/Nc; torque/ripple/loss turns-invariant). ~3
# rounds of new FEA on top of the seed. Separate dir results/gen3_500w_tf.
cd ~/Public/MachineDesign-5f || exit 1
mkdir -p results/gen3_500w_tf
export ANSYSEM_ROOT242=/data/Ansys/v242/Linux64/
export OMP_NUM_THREADS=1 MKL_THREADING_LAYER=SEQUENTIAL KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONPATH=notebooks
./venv_5f/bin/python notebooks/gen3.py \
  --out results/gen3_500w_tf --demands 4,8,2.5 --speeds 25,16,63 --v-max 400 --fhz 50 \
  --turns-free --nc-base 113 \
  --n-seed 64 --n-rounds 3 --q 8 --n-paths 256 --n-gcand 256 --n-icand 512 \
  --n-workers 8 --ncores 1 --slots 60 --phases 5 --seed 0
