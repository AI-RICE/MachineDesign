# Abandoned / superseded attempts (removed 2026-07-01)

Scripts and runs removed during cleanup. Recorded here so the knowledge survives the
deletion. Where a finding was folded into a memory/doc, that is noted.

## Time-horizon / "full electrical period" (Nper) investigation — ABANDONED (artifact)
- `demo_horizon.py`, `demo_res.py`, `nper_sweep.py` (+ `demo_horizon.npz`, `demo_res.npz`, `nper_sweep.json`)
- Hypothesis: simulating a full electrical period vs a fraction changes torque/ripple;
  a dq3-dependent short-window bias.
- Outcome: the apparent bias was a **discretization artifact** (spectral leakage from
  under-resolved high harmonics at PointPer=101); at PointPer>=401 it collapsed to ~0.
  Not physics. The "Nper full-period" change was abandoned.
- Recorded: `fea-fidelity` memory; `paper/h0h1.tex` caveats.

## FEA fidelity / convergence studies — DONE, findings folded in
- `pointper_sweep.py` (mean torque vs PointPer), `mesh05_ripple.py` (ripple at converged mesh)
  (+ `pointper_sweep.json`, `mesh05_ripple.{json,npz}`)
- Outcome: coarse search = 3 mm mesh / PointPer=101 (~0.5-1.5% torque bias); converged =
  0.5 mm + PointPer>=401 (~12x cost). Search coarse, re-check finalists converged.
- Recorded: `fea-fidelity` memory; `docs/setup-bayes.md`.

## ANSYS IPM example import — ABANDONED
- `ipm_inspect.py`
- Tried to open the ANSYS IPM (loaded-torque 2D) example to reuse as a baseline machine.
- Outcome: pyaedt flaky on the multi-design bundle; instead extracted TOPOLOGIES (not the
  model) into `paper/profile.tex`.

## Terminal-voltage output-variable / ddt diagnostic — DONE
- `diag_outvar.py`
- Isolated which token Maxwell's output-variable parser rejects -> `ddt()` is unsupported
  in the dq-transform output variables.
- Recorded: `profile-phase` memory (ddt/voltage note). Handled by adding the end-winding
  leakage `Lew` analytically in post-processing instead of via `ddt()`.

## Nested inner — single-process validation & timing — SUPERSEDED
- `valnested.py`, `timing.py` (+ their logs)
- valnested: validated the per-point nested inner on a known geometry SERIALLY (1 process)
  -> far too slow (~3.8 min/solve), wrong approach.
- timing: measured 1-core solve = 226 s; cores barely help a small 2D transient.
- Outcome: superseded by the parallel worker-pool gen-1 (`nested.py`) + the shared
  current-map inner (one map per geometry, thresholded per demand). See `profile-phase`.

## Fast-screen analytic inner "(b)" — NOT ADOPTED
- `validate_analytic.py`
- De-risked predicting torque/voltage analytically from a dq flux/inductance map (the cheap
  "(b)" inner alternative).
- Outcome: adopted the BO / shared-map inner instead.

## Misc throwaway / superseded runners
- `chk.py`, `smoke.py` — minimal standalone AEDT smoke tests (new-desktop sanity checks).
- `run.py` — early ReluctanceDrive-ported single-run driver; superseded by the `h0h1_par`
  worker pool + `nested.py`. (`run_optimization.py` = upstream serial runner, kept.)

## Removed result directories (regenerable H0/H1-era intermediates / orphans)
Not cited by `paper/h0h1.tex` (which uses only `results_{nolimit,volt50,volt71}`):
`results_grid`, `results_grid_50hz`, `results_h0h1`, `results_h0h1_par`, `results_volt`,
`results_volt_g2`, `results_volt_local`, `results_nolimit_60`, `results_nolimit_rep`,
`results_mock`, and superseded/smoke `results/{nested_gen1,nested_smoke,smoke3f}`.
Regenerable from the retained H0/H1 scripts if ever needed.

## Removed shell orchestration relics (H0/H1 babysitters)
`notebooks/{kill_all,kill_ansys,kill_both,reset_both,run_both,run_chain_k3,
run_chain_volt,run_chain_volt50,run_chain_volt71,run_minloss,watchdog,watchdog_both,
watchdog_volt}.sh` + one-off `cleanup_session.sh`, `relaunch_gen1b.sh`. These were the
nohup/chain launchers, kill scripts, and watchdog babysitters for the H0/H1 runs;
superseded by the `setsid` + resilient short-ssh-poll workflow. Not source.
