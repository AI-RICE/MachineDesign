#!/bin/bash
# Robust GATED launcher for gen3 (pathwise-Thompson MOO). Fixes the two crash modes seen
# on 2026-07-04:
#   (1) OOM-kill when the shared box was starved -> a FREE-GATE waits for RAM before launch.
#   (2) stale AEDT project locks after a crash -> full teardown (ansys + Maxwell engine +
#       stale project files) before relaunch, like gen2_launch.sh.
# Resumes from the results/gen3_500w/gen3.npz checkpoint (seed + completed rounds are kept).
# 8 workers = proven-safe footprint (~26 GB peak); 16 was too heavy under memory pressure.
cd ~/Public/MachineDesign-5f || exit 1
MIN_AVAIL_GB=60

# 1. teardown: kill any prior gen3 + ANSYS, remove stale project scratch (KEEP the npz)
pkill -9 -f "gen3.py"; sleep 2
pkill -9 ansysedt; pkill -9 Maxwell2DComEngine 2>/dev/null; sleep 3
rm -rf data/g3seed* data/seed*_w*.aedt* data/r[0-9]*_w*.aedt*
rm -f gen3.log gen3_gate.log

# 2. free-gate: do not launch until the box has headroom (avoids OOM; waits out other jobs)
while :; do
  avail=$(free -g | awk "/Mem:/{print \$7}")
  if [ "${avail:-0}" -ge "$MIN_AVAIL_GB" ]; then
    echo "$(date +%H:%M:%S) RAM ${avail}GB >= ${MIN_AVAIL_GB}GB -> launching" >> gen3_gate.log
    break
  fi
  echo "$(date +%H:%M:%S) waiting: RAM ${avail}GB < ${MIN_AVAIL_GB}GB" >> gen3_gate.log
  sleep 30
done

# 3. launch (resumes from checkpoint)
export ANSYSEM_ROOT242=/data/Ansys/v242/Linux64/
setsid env PYTHONPATH=notebooks ./venv_5f/bin/python notebooks/gen3.py \
  --out results/gen3_500w --demands 4,8,6 --fhz 50 \
  --n-seed 64 --n-rounds 8 --q 8 --n-paths 256 --n-gcand 256 --n-icand 512 \
  --n-workers 8 --ncores 1 --slots 60 --phases 5 --seed 0 \
  > gen3.log 2>&1 </dev/null &
echo "$(date +%H:%M:%S) launched pid $!" >> gen3_gate.log
