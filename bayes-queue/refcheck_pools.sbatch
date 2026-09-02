#!/bin/bash
#SBATCH --job-name=refcheck
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=results/smoke_postruff/refcheck-%j.log
#SBATCH --open-mode=append
# Reference regression on the FULL pooled geometry set, post-ruff. No ANSYS, no licenses.
# The first pass pooled only 400V+500V (208 geoms) and reproduced the 0.55 Nm floor that
# paper/gen3_report.tex attributes to the pre-Gardner feasibility-blind acquisition. The
# published front needs the corrected-acquisition (gardner) pool too.
# Reference (gen3_report.tex Table 1, 400 V / Nc=113, 220 pooled geoms):
#   min loss 0.811 A^2 @ 0.675 Nm (8.2%) ; min ripple 0.850 A^2 @ 0.409 Nm (5.0%) ; 12 feasible
cd ~/Public/MachineDesign-5f || exit 1
mkdir -p results/smoke_postruff
export OMP_NUM_THREADS=1 MKL_THREADING_LAYER=SEQUENTIAL KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONPATH=notebooks
PY=./venv_5f/bin/python

echo "===== per-pool geometry counts ====="
$PY - <<'PYEOF'
import numpy as np
tot = set()
for d in ("gen3_500w", "gen3_500w_v", "gen3_500w_v500", "gen3_500w_tf", "gen3_500w_gardner"):
    try:
        z = np.load(f"results/{d}/gen3.npz", allow_pickle=True)
        g = {tuple(np.round(x, 6)) for x in np.array(z["Xg"])}
        tot |= g
        print(f"  {d:22s} {len(np.array(z['Xg'])):4d} solves  {len(g):4d} distinct geoms")
    except Exception as e:
        print(f"  {d:22s} -- {type(e).__name__}: {e}")
print(f"  {'UNION':22s} {'':4s}          {len(tot):4d} distinct geoms")
PYEOF

for POOLS in \
  "results/gen3_500w_v,results/gen3_500w_v500" \
  "results/gen3_500w_v,results/gen3_500w_v500,results/gen3_500w_gardner" \
  "results/gen3_500w_v,results/gen3_500w_v500,results/gen3_500w_tf,results/gen3_500w_gardner"
do
  echo; echo "===== turns_eval Nc=113 | pools: $POOLS ====="
  $PY notebooks/turns_eval.py --pools "$POOLS" --nc 113
done

echo; echo "===== analyze_front per run dir ====="
for d in results/gen3_500w_v results/gen3_500w_gardner results/gen3_500w_tf; do
  echo "--- $d ---"; $PY notebooks/analyze_front.py "$d" 2>&1 | head -12
done
