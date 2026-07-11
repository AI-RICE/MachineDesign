# bayes-queue — running ANSYS jobs on the shared `bayes` box

**`bayes` runs SLURM.** Don't `ssh` in and launch solvers by hand — **submit through
`sbatch`**. Jobs are queued and each one reserves its own memory and ANSYS license, so
nobody's run can OOM the box or starve the license pool, and the scheduler packs as many
solves as the budget allows. This page is for **using** the queue; the one-time root install
is the appendix at the end.

Box: 48 cores, 251 GB RAM, ANSYS Electronics v242. Rule of thumb: **one FEA solve = 1 core +
its own `--mem` + one license**; parallelism comes from submitting *many* such jobs.

---

## Submitting a job

Everything is an `sbatch` script that declares its resources. Minimum for one ANSYS solve:

```bash
#SBATCH --cpus-per-task=1                      # 2-D solves don't benefit from more
#SBATCH --mem=6G                               # a realistic cap (see "why --mem" below)
#SBATCH --licenses=electronics_desktop:1       # one AEDT session
```

Submit, then it runs whenever the budget frees up — no need to stay connected:

```bash
sbatch bayes-queue/sweep.sbatch        # returns a job id immediately
```

### Worked example — a rotor-sensitivity sweep (`sweep.sbatch`)

The common case is a **parameter sweep**: many independent single-point solves. Use a
**SLURM job array** — one array index per parameter value, each its own 1-core/mem/license
job, and SLURM runs them concurrently up to the budget. `sweep.sbatch` is a ready template;
adapt the two marked lines to your model:

```bash
sbatch bayes-queue/sweep.sbatch                # submits the whole array
squeue -u $USER                                # watch the array tasks
# results land in results/sweep/task-*.json ; collect when done
```

`--array=0-19%8` means 20 values, at most 8 running at once — polite on the shared box while
still 8-wide. Bump the `%8` up (toward ~40) only if RAM and licenses allow.

---

## Monitoring & control

```bash
squeue -u $USER                    # your running/pending jobs
squeue -u $USER --start            # estimated start time + WHY a job is pending
                                   #   (Resources = waiting on cores/RAM; licenses = seat cap)
scontrol show job <id>             # full detail on one job
scontrol show lic                  # electronics_desktop seats used/free (the courtesy cap)
bash bayes-queue/read-seats.sh     # live seats on the campus FlexLM server (all machines)
scancel <id>                       # cancel a job (or <id>_<k> for one array task)
```

Note: SLURM accounting is **off**, so `sacct` shows no `MaxRSS` — that's expected; the memory
*enforcement* still works, it's just not logged.

---

## Conventions (please follow)

- **Always set `--mem`.** Unset memory can be overcommitted and the kernel then OOM-kills
  *someone*. With a cap, a runaway is killed inside *its own* job and the box is untouched.
  Size it a bit above your solve's real peak.
- **1 core per solve.** ANSYS 2-D solves barely speed up with more cores, and staying 1-core
  avoids the heavily-contended `anshpc` parallel-license pool (`read-seats.sh` shows it near
  full). Want throughput? Submit more 1-core jobs (a job array), don't widen one job.
- **All ANSYS runs go through `sbatch`.** The `electronics_desktop` seat cap in `slurm.conf`
  is a *courtesy* limit on the shared 275-seat campus pool — SLURM can only honor it if
  everyone submits through the queue.

---

## Troubleshooting

- **Job vanished / `Out Of Memory` in the log, no Python traceback** → it hit its `--mem`
  cap and was cgroup-OOM-killed (contained to your job, box safe). Raise `--mem`, or find the
  leak. A classic cause is allocating over a full grid at once (chunk it).
- **Job stuck `PENDING`** → `squeue --start`. `Resources` = waiting for cores/RAM;
  `licenses` = the seat cap is full (someone else's jobs). It'll start when they free.
- **Leftover `mwrpcss` / `ansysedt` processes** → your own job's strays are reaped by the
  epilog on exit. Ones owned by *other* users you can't kill (permission denied) — that's
  the admin's / their problem, not yours; the seat cap already accounts for real usage.
- **Don't use `srun` interactively for long jobs** — it dies if your SSH drops. `sbatch` is
  fire-and-forget. (`../detach.sh` is a non-SLURM detached-runner fallback, but prefer the
  queue.)

---

## Files
- `sweep.sbatch` — job-array rotor-sensitivity sweep template (adapt & submit).
- `read-seats.sh` — print live ANSYS license seat counts (read-only, no root).
- `write-slurm-conf.sh` — generate `/etc/slurm/*` (admin, install-time — see appendix).
- `../detach.sh` — run a command detached *without* SLURM (SSH-drop-proof fallback).

---

## Appendix — one-time install (admin / root only)

Collaborators can skip this; it's done once on the node. Ubuntu 22.04, **cgroup v2**
(`stat -fc %T /sys/fs/cgroup` → `cgroup2fs`). Stock `apt` SLURM is 21.08 which **predates
cgroup-v2 support**, so build current SLURM (24.11) into `.deb`s.

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
bash bayes-queue/write-slurm-conf.sh 32

# 5. start
systemctl enable --now slurmctld slurmd
```

`write-slurm-conf.sh` lays down `/etc/slurm/{slurm.conf,cgroup.conf,epilog.sh}`: memory as a
consumable (`SelectTypeParameters=CR_Core_Memory`), a hard per-job memory cap
(`cgroup.conf ConstrainRAMSpace=yes`), and an epilog that sweeps `mwrpcss`/`ansysedt`/
`Maxwell2DComEngine` strays.

**Verify the install:**
```bash
sinfo && scontrol show lic                                  # node idle; seats listed
srun --mem=1G hostname                                      # basic
srun --mem=4G --licenses=electronics_desktop:1 true         # license path
srun --mem=100M python3 -c "bytearray(500*1024*1024)"       # MUST be OOM-killed (proves the cap)
```
If `slurmd` failed at boot with `fetch_config: DNS SRV lookup failed`, it started before the
config existed — just `systemctl restart slurmd` after step 4.
