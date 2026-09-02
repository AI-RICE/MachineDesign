#!/bin/bash
#SBATCH --job-name=smoke_ruff
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=results/smoke_postruff/smoke-%j.log
#SBATCH --open-mode=append
# Post-ruff validity check for commit e32404f (import reordering + 3 dead symbols).
# Deliberately needs NO ANSYS and NO license seats: the pools are unchanged, so the
# only thing to prove is (1) every changed module's imports still resolve in the new
# order, and (2) the FEA-free analyses reproduce the PUBLISHED gen-3 numbers from
# those unchanged pools. Reference values (paper/gen3_report.tex, Tables 1-2):
#   400 V / Nc=113 front : min-loss 0.811 A^2 @ 0.675 Nm ; min-ripple 0.850 @ 0.409 Nm
#   pool = gen3_500w_v + gen3_500w_v500 + gen3_500w_gardner (220 geoms; NOT _tf)
#   feasible geometries  : 12 of 220 pooled
#   regimes              : P2 ipk = 98% Imax (current-bound) ; P3 Vpk = 91% Vmax
cd ~/Public/MachineDesign-5f || exit 1
mkdir -p results/smoke_postruff
export OMP_NUM_THREADS=1 MKL_THREADING_LAYER=SEQUENTIAL KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONPATH=notebooks
PY=./venv_5f/bin/python
rc=0

echo "===== (1) import smoke test ====="
$PY notebooks/smoke_imports.py e32404f; s=$?; echo "-> smoke_imports exit=$s"; [ $s -ne 0 ] && rc=1

echo; echo "===== (2) turns_eval, Nc=113, published 220-geom pool (FEA-free) ====="
$PY notebooks/turns_eval.py --pools results/gen3_500w_v,results/gen3_500w_v500,results/gen3_500w_gardner --nc 113
s=$?; echo "-> turns_eval exit=$s"; [ $s -ne 0 ] && rc=1

echo; echo "===== (3) analyze_front on results/gen3_500w_v (FEA-free) ====="
$PY notebooks/analyze_front.py results/gen3_500w_v
s=$?; echo "-> analyze_front exit=$s"; [ $s -ne 0 ] && rc=1

echo; echo "===== VERDICT: $([ $rc -eq 0 ] && echo PASS || echo FAIL) ====="
exit $rc
