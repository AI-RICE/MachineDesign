# bayes-queue — single-node SLURM for the shared `bayes` box

A lightweight SLURM setup so multiple users can share **bayes** (48 cores, 251 GB,
ANSYS Electronics) without stepping on each other. It enforces **memory** and **ANSYS
licenses** per job, and cleans up ANSYS solver strays automatically.

## Why

Running FEA/optimization jobs directly on the shared box caused recurring problems:

- **OOM crashes** — a runaway process (or just too many concurrent jobs) exhausted RAM
  and the kernel OOM-killed *whatever it felt like*, taking down unrelated work.
- **Orphaned `mwrpcss`/`ansysedt`** — force-killed ANSYS sessions left solver RPC
  daemons behind (we found 100+ orphans up to 120 days old) holding license seats.
- **No fairness / no queue** — everyone `ssh`'d in and launched, hoping RAM was free.

SLURM fixes all three: jobs declare `--mem` (cgroup-enforced → a runaway is killed *in its
own cgroup*, box untouched), the node is never overcommitted (memory is a consumable
resource), an epilog + cgroup proctrack reap solver strays on job exit, and everything goes
through one queue.

## Install (Ubuntu 22.04, cgroup v2 — do once, as root)

Stock `apt` SLURM is 21.08 which **predates cgroup-v2 support**, so we build current SLURM
(24.11) into `.deb`s. Full steps:

```bash
# 1. MUNGE (auth)
apt install -y munge libmunge-dev
/usr/sbin/create-munge-key -r -f
chown munge: /etc/munge/munge.key && chmod 400 /etc/munge/munge.key
systemctl enable --now munge

# 2. build SLURM 24.11 as .deb (cgroup v2 support)
apt install -y build-essential fakeroot devscripts equivs libpmix-dev libhwloc-dev \
    libyaml-dev libjson-c-dev liblz4-dev libssl-dev libdbus-1-dev libhttp-parser-dev
SLURM_VER=24.11.5           # latest 24.11.x from https://download.schedmd.com/slurm/
cd /usr/local/src
wget https://download.schedmd.com/slurm/slurm-$SLURM_VER.tar.bz2
tar xjf slurm-$SLURM_VER.tar.bz2 && cd slurm-$SLURM_VER
mk-build-deps -i -t 'apt-get -y' debian/control
debuild -b -uc -us
cd .. && dpkg -i slurm-smd_*.deb slurm-smd-slurmctld_*.deb slurm-smd-slurmd_*.deb slurm-smd-client_*.deb
apt -f install -y

# 3. slurm user + dirs
useradd -r -m -d /var/lib/slurm -s /usr/sbin/nologin slurm 2>/dev/null || true
mkdir -p /var/spool/slurmctld /var/spool/slurmd /var/log/slurm
chown slurm: /var/spool/slurmctld /var/log/slurm

# 4. write configs (auto-detects host/CPUs/RAM; arg = ANSYS courtesy cap)
bash write-slurm-conf.sh 32

# 5. start
systemctl enable --now slurmctld slurmd
sinfo && scontrol show lic
```

`write-slurm-conf.sh` lays down `/etc/slurm/{slurm.conf,cgroup.conf,epilog.sh}` (see the
script header). Key choices: `SelectTypeParameters=CR_Core_Memory` (memory a consumable),
`cgroup.conf ConstrainRAMSpace=yes` (hard per-job memory cap), and an epilog that sweeps
`mwrpcss`/`ansysedt`/`Maxwell2DComEngine` strays.

## Usage (any user, no root)

```bash
# submit a job — always declare --mem and (for ANSYS solves) a license + 1 core/solve
sbatch --mem=6G --cpus-per-task=1 --licenses=electronics_desktop:1 solve.sh
squeue -u $USER                 # your queue
scontrol show lic               # license pool usage
bash read-seats.sh              # live ANSYS seat counts from the campus FlexLM server
```

See `gen3.sbatch` for a worked example (the gen-3 optimizer as a single memory-capped job).

## Conventions

- **All ANSYS runs go through `sbatch`.** The `Licenses=electronics_desktop:32` in
  `slurm.conf` is a self-imposed courtesy cap on the shared 275-seat campus pool (SLURM
  can't reserve a pool it doesn't own) — it only works if everyone submits via SLURM.
- **1 core per solve.** ANSYS 2D solves barely benefit from more cores, and staying 1-core
  avoids the contended `anshpc` parallel-license pool (`read-seats.sh` shows it near-full).
  Parallelism comes from running *many* 1-core jobs, which SLURM schedules up to the
  core/memory/license budget.
- **Always set `--mem`.** Unset memory can be overcommitted; a realistic cap makes the
  scheduler protect the node and kills only *your* job if it runs away.

## Files
- `write-slurm-conf.sh` — generate `/etc/slurm/*` (run as root, once).
- `read-seats.sh` — print live ANSYS license seat counts (read-only).
- `gen3.sbatch` — example job (gen-3 optimizer).
- `../detach.sh` — fallback for running a command detached *without* SLURM (survives SSH
  drops); prefer `sbatch` now that SLURM is up.
