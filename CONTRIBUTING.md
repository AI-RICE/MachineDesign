# Contributing to MachineDesign

A short, shared "culture" so several people can do their own research on this repo without stepping on each other. It exists because we hit the failure mode it prevents: the *same* fix (the AEDT `ddt()` output-var bug) was written independently on three branches, and the core machine model `machine_design/design2.py` has drifted into a different version on every branch. The rules below keep shared code shared and personal work personal.

## The one rule

> **Shared code lives on `main` and is changed only by a small PR to `main`.
> Your research branch adds and edits only `experiments/<your-project>/`.**

If you find yourself editing `machine_design/` or `optimization/` on your own branch, stop — that's a change everyone needs, so it's a PR to `main`. Then rebase your branch on `main` and everyone gets the fix once.

## Three layers — what goes where

| layer | directory | shared? | contains |
|---|---|---|---|
| **Designs** | `machine_design/` | yes (main) | *anchor* machines, geometry generators, and the one-solve FEA evaluation |
| **Methods** | `optimization/` | yes (main) | design-**agnostic** optimizers (BO, Thompson, nested, PFN…) that talk to designs only through the interface |
| **Experiments** | `experiments/<name>/` | no (your branch) | your run configs, the profile/objective you're studying, analysis notebooks, result artifacts, write-ups |

The point of the split: **a method should run on any design, and a design should be optimizable by any method.** Neither should import the other's internals — they meet at the interface (below).

## Anchor designs — named, frozen, *derived not forked*

An **anchor** is a named, version-controlled machine definition — everything needed to instantiate and solve one physical machine:

```
class, slots, phases, winding (turns, Rs), end-winding Lew, rotor radii,
geometry generator + parameter bounds, FEA template
```

Anchors live in `machine_design/anchors.py` (or `anchors/<name>.yaml`) and are named `<class>_<phases>f_<slots>s_<turns>t`, e.g. `synrm_5f_60s_113t` (the 500 W 5-phase machine). Rules:

- **Derive, don't fork.** To study a variant (add magnets, change turns), *reference* an anchor and override only what differs — do **not** edit the anchor's class in place. A genuinely new machine is a **new anchor**, appended to the registry.
- **Anchors are append-only and shared.** A bug in an anchor's FEA (like the `ddt()` one) is fixed **once**, by a PR to `main`; everyone rebases and inherits it. No per-branch copies of `design2.py`.

## Methods and the design interface

An optimizer in `optimization/` depends only on this contract, which every anchor implements:

```python
sample_feasible_geometry() -> g            # a valid rotor (normalized params)
geometry_bounds()          -> (lb, ub)
evaluate(g, currents)      -> dict          # ONE FEA solve -> {T, ripple, flux, ...}
peak_current(currents)     -> float         # analytic constraints, no FEA
```

So `optimization/pathwise_thompson.py` never imports a specific design; you point it at `anchors.get("synrm_5f_60s_113t")` (or any other) and it runs. If you write a new method, put it here and it's instantly reusable by everyone. If you need a response the interface doesn't expose, extend the **interface** (a PR to main), not your method.

## Branch & merge workflow

- **`main` is the living trunk**, not a stale base. Keep it green; rebase onto it often.
- **Research branches are thin.** Name them `exp/<topic>` or `<initials>/<topic>` (e.g. `exp/reluctance-5f`, `vs/pfn-revisited`). They should almost never show a diff outside `experiments/<name>/`.
- **Shared-code change → PR to `main`.** Small, focused, reviewed by one other person. This is the whole discipline; everything else follows from it.
- Don't let a branch live for months without merging its shared parts back — that is how we got four divergent `design2.py`s.

## Running things

All compute goes through the SLURM queue on `bayes` — see [`bayes-queue/README.md`](bayes-queue/README.md). One FEA solve = 1 core + its own `--mem` + one license; submit sweeps as job arrays (`bayes-queue/sweep.sbatch`).

## Quick checklists

**I want to optimize an existing machine with an existing method** → in `experiments/<name>/`, write a config that picks an anchor + a method + your objective/constraints, and an sbatch to run it. Touch nothing shared.

**I want to study a new machine** → add an **anchor** (PR to `main`); if it's a variant, derive it from an existing anchor. Then use it from your experiment.

**I want a new optimization method** → add it to `optimization/` against the interface (PR to `main`). Now anyone can use it on any anchor.

**I found a bug in a shared design/method** → fix it on `main` via PR, then rebase your branch. Do not patch it only on your branch.
