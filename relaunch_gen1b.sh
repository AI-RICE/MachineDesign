#!/bin/bash
# Bayes-side relaunch of the corrected (shared-map inner) gen-1 nested run.
# Triggered detached via setsid so a flaky laptop ssh only needs a sub-second window.
cd ~/Public/MachineDesign-5f || exit 1
pkill -9 -f "notebooks/nested.py"; sleep 3
pkill -9 ansysedt; pkill -9 Maxwell2DComEngine 2>/dev/null; sleep 3
rm -f data/nest_*.aedt* data/nseed_*.aedt*
mkdir -p results/nested_gen1b
export ANSYSEM_ROOT242=/data/Ansys/v242/Linux64/
setsid ./venv_5f/bin/python notebooks/nested.py \
  --out results/nested_gen1b --points 20:50,35:50,30:71.2 \
  --n-init 24 --n-total 48 --q 24 --n-workers 24 --n-inner 20 --ncores 1 --seed 0 \
  > results/nested_gen1b/run.log 2>&1 < /dev/null &
