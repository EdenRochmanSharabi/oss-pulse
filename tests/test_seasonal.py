"""Tests for seasonal decomposition and stationarity tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def sine_wave_series() -> pd.Series:
    """Synthetic weekly series with clear seasonality."""
    dates = pd.date_range("2020-01-01", periods=156, freq="W")
    t = np.arange(len(dates))
    trend = 0.1 * t
    seasonal = 10 * np.sin(2 * np.pi * t / 52)
    noise = np.random.default_rng(42).normal(0, 1, len(t))
    values = 50 + trend + seasonal + noise
    return pd.Series(values, index=dates, name="pr_count")


def test_stl_decompose_returns_components(sine_wave_series: pd.Series) -> None:
    from oss_pulse.analyze.seasonal import stl_decompose

    result = stl_decompose(sine_wave_series, period=52)
    assert hasattr(result, "trend")
    assert hasattr(result, "seasonal")
    assert hasattr(result, "resid")
    assert len(result.trend) == len(sine_wave_series)


def test_stationarity_returns_valid_pvalues(sine_wave_series: pd.Series) -> None:
    from oss_pulse.analyze.seasonal import test_stationarity

    result = test_stationarity(sine_wave_series)
    assert "adf_pvalue" in result
    assert "kpss_pvalue" in result
    assert 0 <= result["adf_pvalue"] <= 1
    assert 0 <= result["kpss_pvalue"] <= 1
    assert isinstance(result["is_stationary"], bool)


def test_acf_pacf_shapes(sine_wave_series: pd.Series) -> None:
    from oss_pulse.analyze.seasonal import compute_acf_pacf

    acf_vals, pacf_vals = compute_acf_pacf(sine_wave_series, nlags=30)
    assert len(acf_vals) == 31  # nlags + 1 (lag 0)
    assert len(pacf_vals) == 31
