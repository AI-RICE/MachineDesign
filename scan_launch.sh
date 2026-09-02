#!/bin/bash
cd ~/Public/MachineDesign-5f || exit 1
pkill -9 -f "paircheck.py"; pkill -9 -f "scancheck.py"; sleep 2
pkill -9 ansysedt; pkill -9 Maxwell2DComEngine 2>/dev/null; sleep 3
rm -f data/scan_w*.aedt*
rm -rf results/scancheck; mkdir -p results/scancheck
export ANSYSEM_ROOT242=/data/Ansys/v242/Linux64/
setsid ./venv_5f/bin/python notebooks/scancheck.py --n-workers 20 --ncores 1 \
  > results/scancheck/run.log 2>&1 < /dev/null &
