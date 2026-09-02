#!/bin/bash
# Revert deriv() output-vars (unsupported in AEDT 2024.2; Lew=0 so identical to no-term)
# and relaunch gen2 v2 at 8 workers. Idempotent sed.
cd ~/Public/MachineDesign-5f || exit 1
pkill -9 -f "gen2.py"; sleep 2
pkill -9 ansysedt; pkill -9 Maxwell2DComEngine 2>/dev/null; sleep 3
sed -i 's/ + Lew\*deriv(InputCurrent(Phase\([A-E]\)))",/",/' machine_design/design2.py
rm -f data/seed*_w*.aedt* data/r[0-9]*_w*.aedt* data/g2seed_w*.aedt*
rm -rf results/gen2_v2; mkdir -p results/gen2_v2
export ANSYSEM_ROOT242=/data/Ansys/v242/Linux64/
setsid ./venv_5f/bin/python notebooks/gen2.py \
  --out results/gen2_v2 --demands 20,35,30 \
  --n-seed 48 --n-rounds 8 --q 6 --n-gcand 256 --n-icand 64 \
  --n-workers 8 --ncores 1 --slots 60 --phases 5 --seed 0 \
  > results/gen2_v2/run.log 2>&1 < /dev/null &
