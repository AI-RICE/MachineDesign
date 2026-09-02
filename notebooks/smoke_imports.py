"""Post-ruff import smoke test (no ANSYS, no licenses, no FEA).

Commit e32404f only reordered imports (I001) and dropped three dead symbols
(F401/F541/F841), so the single failure mode to rule out is an import that no
longer resolves -- or resolves in a broken ORDER. This project has two real
order hazards: torch/botorch vs. ansys.aedt.core (PyAEDT segfault) and
matplotlib.use("Agg") before pyplot.

Modules with a __main__ guard are imported outright. Script-style files execute
on import (they load npz pools, write figures), so for those we AST-parse the
import statements and import each NAMED module instead -- validating that the
reordered imports resolve without executing anything.

Usage:  PYTHONPATH=notebooks python notebooks/smoke_imports.py [commit]
        (default commit: e32404f, the ruff commit)
"""

import ast
import importlib
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
COMMIT = sys.argv[1] if len(sys.argv) > 1 else "e32404f"

# Modules that must never be imported here: they need a live AEDT desktop.
SKIP_MODULES = {"ansys", "ansys.aedt", "ansys.aedt.core", "pyaedt"}

# Orphaned trees that cannot be imported here BY DESIGN, and whose failure is not a
# regression. machine_design/parallel_calculation/ is the older standalone parallel
# driver: nothing in the 5f pipeline references it, it needs h5py (absent from
# venv_5f), and it imports its sibling as a bare top-level module, which resolves
# only when run from inside its own directory. Verified pre-existing by A/B against
# e32404f^, where the same import fails at h5py instead.
SKIP_FILE_PREFIXES = ("machine_design/parallel_calculation/",)


def changed_py_files(commit):
    out = subprocess.run(["git", "show", "--name-only", "--format=", commit], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [f for f in out.split() if f.endswith(".py") and (ROOT / f).exists()]


def declared_imports(path):
    """Top-level module names imported by a file, in source order."""
    names = []
    for node in ast.parse((ROOT / path).read_text()).body:
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.append(node.module)
    return names


def has_main_guard(path):
    src = (ROOT / path).read_text()
    return "__name__" in src and "__main__" in src


def check_agg_order(files):
    """matplotlib.use("Agg") must still precede the pyplot import."""
    bad = []
    for f in files:
        lines = (ROOT / f).read_text().splitlines()
        use = next((i for i, ln in enumerate(lines) if "matplotlib.use(" in ln), None)
        plt = next((i for i, ln in enumerate(lines) if "matplotlib.pyplot" in ln), None)
        if use is not None and plt is not None and use > plt:
            bad.append(f"{f}: use() at line {use + 1} AFTER pyplot at line {plt + 1}")
    return bad


def check_torch_aedt(files):
    """No file may import both torch-family and ansys.aedt.core (segfault hazard)."""
    bad = []
    for f in files:
        mods = declared_imports(f)
        torchy = [m for m in mods if m.split(".")[0] in {"torch", "botorch", "gpytorch"}]
        aedty = [m for m in mods if m.split(".")[0] in {"ansys", "pyaedt"}]
        if torchy and aedty:
            bad.append(f"{f}: {torchy} together with {aedty}")
    return bad


def main():
    files = changed_py_files(COMMIT)
    print(f"[smoke] {COMMIT}: {len(files)} changed .py files\n")

    fails = []

    print("=== (a) static order invariants ===")
    for label, bad in (("matplotlib Agg-before-pyplot", check_agg_order(files)), ("torch/PyAEDT separation", check_torch_aedt(files))):
        if bad:
            fails += bad
            print(f"  FAIL  {label}")
            for b in bad:
                print(f"          {b}")
        else:
            print(f"  OK    {label}")

    print("\n=== (b) importable modules (have a __main__ guard) ===")
    for f in files:
        if f.startswith(SKIP_FILE_PREFIXES):
            print(f"  SKIP  {f} (orphan tree, see SKIP_FILE_PREFIXES)")
            continue
        if not has_main_guard(f):
            continue
        mod = f[:-3].replace("/", ".").removeprefix("notebooks.")
        try:
            importlib.import_module(mod)
            print(f"  OK    import {mod}")
        except Exception as e:
            fails.append(f"{mod}: {type(e).__name__}: {e}")
            print(f"  FAIL  import {mod}: {type(e).__name__}: {e}")

    print("\n=== (c) script-style files: resolve declared imports only ===")
    for f in files:
        if has_main_guard(f) or f.startswith(SKIP_FILE_PREFIXES):
            continue
        unresolved = []
        for name in declared_imports(f):
            if name in SKIP_MODULES or name.split(".")[0] in SKIP_MODULES:
                continue
            try:
                importlib.import_module(name)
            except Exception as e:
                unresolved.append(f"{name} ({type(e).__name__}: {e})")
        if unresolved:
            fails += [f"{f}: {u}" for u in unresolved]
            print(f"  FAIL  {f}: {', '.join(unresolved)}")
        else:
            print(f"  OK    {f} ({len(declared_imports(f))} imports resolve)")

    print(f"\n[smoke] {'FAILED' if fails else 'PASSED'} -- {len(fails)} problem(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
