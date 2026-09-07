import numpy as np
import pytest

from machine_design.optimization.optimization import objective_transform


def test_objective_transform_normal_case():
    f1, f2 = objective_transform(TorAvg=4.5, TorRippleRms=8.0)
    assert f1 == pytest.approx(4.5)
    assert f2 == pytest.approx(-0.08)


def test_objective_transform_uses_fallback_when_torque_is_null():
    f1, f2 = objective_transform(TorAvg=np.nan, TorRippleRms=np.nan, objective_fallback=(3.0, 5.0))
    assert f1 == pytest.approx(3.0)
    assert f2 == pytest.approx(-0.05)


def test_objective_transform_raises_without_fallback():
    with pytest.raises(ValueError):
        objective_transform(np.nan, np.nan)
