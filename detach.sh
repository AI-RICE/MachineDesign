#!/bin/bash
# detach.sh — run a command line fully detached on this box, surviving SSH disconnects.
# The recurring pain: `setsid ... &` typed inline over a flaky SSH link gets SIGHUP'd when
# the link blips before the job backgrounds. This wraps it once, robustly.
#
# Usage:   detach.sh <tag> '<command line>'
#   e.g.   detach.sh gen3 'bash gen3_launch.sh'
#          detach.sh sel  'OMP_NUM_THREADS=1 ./venv_5f/bin/python notebooks/gen3_select.py a.npz b.json'
# Effect:  runs the command with no controlling terminal, cwd = repo root, stdout+stderr ->
#          logs/<tag>.log, PID -> logs/<tag>.pid. Returns immediately.
# Check:   tail -f logs/<tag>.log   |   kill -9 $(cat logs/<tag>.pid) to stop it.
set -u
ROOT=~/Public/MachineDesign-5f
tag="${1:?usage: detach.sh <tag> '<command line>'}"; shift
cmd="${*:?need a command line}"
mkdir -p "$ROOT/logs"
cd "$ROOT" || exit 1
setsid bash -c "$cmd" </dev/null >"$ROOT/logs/$tag.log" 2>&1 &
pid=$!
echo "$pid" > "$ROOT/logs/$tag.pid"
echo "detached: tag=$tag pid=$pid cwd=$ROOT log=logs/$tag.log"
