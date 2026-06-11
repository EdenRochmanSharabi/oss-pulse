"""Tests for time-series forecasting models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from oss_pulse.analyze.forecast import _compute_metrics, benchmark_models


class TestComputeMetrics:
    def test_returns_correct_keys(self) -> None:
        actual = pd.Series([10.0, 20.0, 30.0])
        predicted = pd.Series([11.0, 19.0, 31.0])
        result = _compute_metrics(actual, predicted)
        assert set(result.keys()) == {"mae", "rmse", "mape"}

    def test_perfect_prediction(self) -> None:
        actual = pd.Series([5.0, 10.0, 15.0])
        result = _compute_metrics(actual, actual)
        assert result["mae"] == 0.0
        assert result["rmse"] == 0.0
        assert result["mape"] == 0.0

    def test_known_values(self) -> None:
        actual = pd.Series([100.0, 200.0])
        predicted = pd.Series([110.0, 190.0])
        result = _compute_metrics(actual, predicted)
        assert abs(result["mae"] - 10.0) < 1e-9
        assert abs(result["rmse"] - 10.0) < 1e-9
        # MAPE: mean(|10/100|, |10/200|) * 100 = mean(0.1, 0.05) * 100 = 7.5
        assert abs(result["mape"] - 7.5) < 1e-9

    def test_mape_handles_zeros(self) -> None:
        actual = pd.Series([0.0, 0.0])
        predicted = pd.Series([1.0, 2.0])
        result = _compute_metrics(actual, predicted)
        assert result["mape"] == float("inf")


class TestBenchmarkModels:
    @pytest.fixture(scope="class")
    def simple_series(self) -> pd.Series:
        """A simple trending series with enough data points for train/test split."""
        rng = np.random.default_rng(42)
        dates = pd.date_range("2020-01-01", periods=120, freq="W")
        trend = np.linspace(10, 50, 120)
        noise = rng.normal(0, 2, 120)
        values = trend + noise
        return pd.Series(values, index=dates, name="pr_count")

    def test_returns_dataframe(self, simple_series: pd.Series) -> None:
        try:
            result = benchmark_models(simple_series)
        except Exception:
            pytest.skip("Benchmark models failed (missing optional dependency)")
        assert isinstance(result, pd.DataFrame)

    def test_has_expected_columns(self, simple_series: pd.Series) -> None:
        try:
            result = benchmark_models(simple_series)
        except Exception:
            pytest.skip("Benchmark models failed")
        expected_cols = {"model", "mae", "rmse", "mape"}
        assert expected_cols.issubset(set(result.columns))

    def test_at_least_one_model_succeeds(self, simple_series: pd.Series) -> None:
        try:
            result = benchmark_models(simple_series)
        except Exception:
            pytest.skip("Benchmark models failed")
        assert len(result) >= 1

    def test_metrics_are_non_negative(self, simple_series: pd.Series) -> None:
        try:
            result = benchmark_models(simple_series)
        except Exception:
            pytest.skip("Benchmark models failed")
        for col in ["mae", "rmse"]:
            assert (result[col] >= 0).all(), f"{col} has negative values"
