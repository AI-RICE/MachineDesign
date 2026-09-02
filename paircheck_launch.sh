#!/bin/bash
# Robust launcher for the paired coarse-vs-fine ripple check (converged FEA, heavy).
cd ~/Public/MachineDesign-5f || exit 1
pkill -9 -f "gen2.py"; pkill -9 -f "paircheck.py"; sleep 2
pkill -9 ansysedt; pkill -9 Maxwell2DComEngine 2>/dev/null; sleep 3
rm -f data/pair_w*.aedt*
rm -rf results/paircheck; mkdir -p results/paircheck
export ANSYSEM_ROOT242=/data/Ansys/v242/Linux64/
setsid ./venv_5f/bin/python notebooks/paircheck.py --npts 10 --n-workers 10 --ncores 1 \
  > results/paircheck/run.log 2>&1 < /dev/null &
