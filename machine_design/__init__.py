from .generators import (
    FourStupid,
    HacklGenerator_3BrokenLines,
    HacklGenerator_OneLambda,
    HacklGenerator_SixLambdas,
    HacklGenerator_TwoLambdas,
    save_params,
)
from .geometry import analyze_results, plot_barriers, rotate
from .optimization import init_points, objective, objective_single, objective_transform

# Design and load_design pull in ansys.aedt.core; tolerate its absence so that
# the lumped-reluctance sketcher and other ANSYS-free tooling import cleanly.
try:
    from .design import Design  # noqa: F401
    from .load import load_design  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - environment-dependent
    pass
