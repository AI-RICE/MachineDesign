import pickle

import numpy as np
import pandas as pd
import pytest

from machine_design.optimization.optimization import init_points, objective_transform


def test_objective_transform_normal_case():
    f1, f2 = objective_transform(TorAvg=4.5, TorRippleRms=8.0)
    assert f1 == pytest.approx(4.5)
    assert f2 == pytest.approx(-0.08)


def test_objective_transform_fallback():
    # torque null, uses fallback
    f1, f2 = objective_transform(TorAvg=np.nan, TorRippleRms=np.nan, objective_fallback=(3.0, 5.0))
    assert f1 == pytest.approx(3.0)
    assert f2 == pytest.approx(-0.05)


def test_objective_transform_no_fallback():
    # torque null, no fallback given, should raise
    with pytest.raises(ValueError):
        objective_transform(np.nan, np.nan)


def test_init_points_filters_and_flattens(tmp_path):
    # filters by method and non-null T, flattens nested params into one X row
    metadata = pd.DataFrame(
        {
            "method": ["A", "A", "B"],
            "design": [0, 1, 0],
            "T": [5.0, None, 6.0],
            "ripple": [10.0, 20.0, 15.0],
        }
    )
    metadata.to_csv(tmp_path / "metadata.csv", index=False)

    with open(tmp_path / "design_A_0.pkl", "wb") as f:
        pickle.dump((1.0, 2.0, [3.0, 4.0]), f)

    Xs, Ys = init_points(str(tmp_path), "A")

    assert Xs.shape == (1, 4)
    assert Xs[0].tolist() == pytest.approx([1.0, 2.0, 3.0, 4.0])
    assert Ys[0].tolist() == pytest.approx([5.0, -0.1])
