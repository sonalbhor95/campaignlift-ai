import numpy as np

from src.metrics import qini_curve, qini_auc, uplift_by_decile


def test_qini_curve_runs():
    y = np.array([1, 0, 1, 0, 1, 0])
    t = np.array([1, 1, 0, 0, 1, 0])
    u = np.array([0.9, 0.7, 0.5, 0.3, 0.2, 0.1])
    curve = qini_curve(y, t, u)
    assert len(curve) == len(y)
    assert "incremental_outcome" in curve.columns


def test_qini_auc_returns_float():
    y = np.array([1, 0, 1, 0, 1, 0])
    t = np.array([1, 1, 0, 0, 1, 0])
    u = np.array([0.9, 0.7, 0.5, 0.3, 0.2, 0.1])
    assert isinstance(qini_auc(y, t, u), float)


def test_uplift_by_decile_runs():
    y = np.random.randint(0, 2, size=100)
    t = np.random.randint(0, 2, size=100)
    u = np.random.normal(size=100)
    result = uplift_by_decile(y, t, u)
    assert "observed_uplift" in result.columns
