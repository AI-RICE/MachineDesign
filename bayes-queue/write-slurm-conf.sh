#!/bin/bash
# write-slurm-conf.sh — lay down SLURM single-node config for bayes (Ubuntu 22.04, cgroup v2).
# Run as root AFTER slurm is installed (so `slurmd -C` / dirs exist):
#     sudo bash write-slurm-conf.sh [COURTESY_CAP]
# e.g. sudo bash write-slurm-conf.sh 32
# Auto-detects hostname, CPU count, and RealMemory (total - reserve). Backs up any existing
# files. Writes /etc/slurm/{slurm.conf,cgroup.conf,epilog.sh}.
#
# COURTESY_CAP (default 32): NOT a real license reservation. The ANSYS pool is a shared
# 275-seat campus server SLURM doesn't own (see read-seats.sh), so this is only a self-imposed
# cap on how many concurrent electronics_desktop checkouts we make, to be polite to the shared
# server. Cores (48) and memory (250 GB) are the real binding constraints here.
set -euo pipefail

[ "$(id -u)" -eq 0 ] || { echo "run as root (sudo)"; exit 1; }
SEATS="${1:-32}"

HOST=$(hostname -s)
CPUS=$(nproc)
TOTAL_MB=$(free -m | awk '/^Mem:/{print $2}')
RESERVE_MB=7588                       # leave ~7.5 GB for OS + slurmd
REALMEM=$(( TOTAL_MB - RESERVE_MB ))

mkdir -p /etc/slurm /var/spool/slurmctld /var/spool/slurmd /var/log/slurm
chown slurm: /var/spool/slurmctld /var/log/slurm 2>/dev/null || true
ts=$(date +%Y%m%d-%H%M%S)
for f in slurm.conf cgroup.conf epilog.sh; do
  [ -e "/etc/slurm/$f" ] && cp -a "/etc/slurm/$f" "/etc/slurm/$f.bak-$ts"
done

echo "host=$HOST cpus=$CPUS realmem=${REALMEM}MB (total ${TOTAL_MB}) courtesy_cap=$SEATS"

# ---------- slurm.conf (values interpolated) ----------
cat > /etc/slurm/slurm.conf <<EOF
ClusterName=bayes
SlurmctldHost=$HOST
SlurmUser=slurm
StateSaveLocation=/var/spool/slurmctld
SlurmdSpoolDir=/var/spool/slurmd
SlurmctldPidFile=/run/slurmctld.pid
SlurmdPidFile=/run/slurmd.pid
ProctrackType=proctrack/cgroup
TaskPlugin=task/cgroup,task/affinity
SelectType=select/cons_tres
SelectTypeParameters=CR_Core_Memory
JobAcctGatherType=jobacct_gather/cgroup
ReturnToService=2
SlurmctldLogFile=/var/log/slurm/slurmctld.log
SlurmdLogFile=/var/log/slurm/slurmd.log
# ANSYS courtesy cap (self-imposed; shared 275-seat campus pool, not a real reservation)
Licenses=electronics_desktop:$SEATS
# node + partition
NodeName=$HOST CPUs=$CPUS RealMemory=$REALMEM State=UNKNOWN
PartitionName=main Nodes=$HOST Default=YES MaxTime=INFINITE State=UP OverSubscribe=NO
Epilog=/etc/slurm/epilog.sh
EOF

# ---------- cgroup.conf (literal) ----------
cat > /etc/slurm/cgroup.conf <<'EOF'
ConstrainCores=yes
ConstrainRAMSpace=yes
ConstrainSwapSpace=yes
EOF

# ---------- epilog.sh (literal; $SLURM_JOB_USER expands at job time, not now) ----------
cat > /etc/slurm/epilog.sh <<'EOF'
#!/bin/bash
# cgroup proctrack kills the job tree; this sweeps any daemonized ANSYS strays as insurance.
pkill -9 -u "$SLURM_JOB_USER" -f 'mwrpcss|ansysedt|Maxwell2DComEngine' 2>/dev/null
exit 0
EOF

chmod 644 /etc/slurm/slurm.conf /etc/slurm/cgroup.conf
chmod 755 /etc/slurm/epilog.sh

echo "wrote /etc/slurm/{slurm.conf,cgroup.conf,epilog.sh}"
echo "next: sudo systemctl enable --now slurmctld slurmd && sinfo && scontrol show lic"
