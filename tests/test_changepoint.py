"""Tests for changepoint detection and pre/post statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from oss_pulse.analyze.changepoint import (
    annotate_ai_events,
    compute_pre_post_stats,
    detect_changepoints,
)


class TestDetectChangepoints:
    def test_clear_mean_shift(self) -> None:
        rng = np.random.default_rng(42)
        low = rng.normal(10, 1, 50)
        high = rng.normal(50, 1, 50)
        values = np.concatenate([low, high])
        series = pd.Series(values)

        cps = detect_changepoints(series, min_size=5)
        assert len(cps) >= 1
        # At least one changepoint should be near index 50 (+/- 10)
        near_50 = [cp for cp in cps if abs(cp - 50) <= 10]
        assert len(near_50) >= 1, f"Expected changepoint near 50, got {cps}"

    def test_no_changepoint_in_constant(self) -> None:
        series = pd.Series(np.ones(100))
        cps = detect_changepoints(series, min_size=5)
        # A constant series should have no changepoints (or very few spurious ones)
        assert len(cps) <= 1

    def test_binseg_method(self) -> None:
        rng = np.random.default_rng(42)
        low = rng.normal(10, 1, 50)
        high = rng.normal(50, 1, 50)
        values = np.concatenate([low, high])
        series = pd.Series(values)

        cps = detect_changepoints(series, method="binseg", min_size=5)
        assert len(cps) >= 1

    def test_invalid_method_raises(self) -> None:
        series = pd.Series(np.ones(50))
        with pytest.raises(ValueError, match="Unknown method"):
            detect_changepoints(series, method="invalid")

    def test_terminal_index_removed(self) -> None:
        rng = np.random.default_rng(42)
        values = np.concatenate([rng.normal(10, 1, 50), rng.normal(50, 1, 50)])
        series = pd.Series(values)
        cps = detect_changepoints(series, min_size=5)
        # The terminal index (len(series)) should never be in the result
        assert len(series) not in cps


class TestComputePrePostStats:
    def test_returns_correct_keys(self) -> None:
        series = pd.Series(np.arange(100, dtype=float))
        result = compute_pre_post_stats(series, 50)
        expected_keys = {
            "pre_mean",
            "pre_std",
            "pre_median",
            "pre_n",
            "post_mean",
            "post_std",
            "post_median",
            "post_n",
            "effect_size",
        }
        assert set(result.keys()) == expected_keys

    def test_pre_post_split_correct(self) -> None:
        series = pd.Series(np.arange(100, dtype=float))
        result = compute_pre_post_stats(series, 30)
        assert result["pre_n"] == 30
        assert result["post_n"] == 70

    def test_effect_size_positive_for_increase(self) -> None:
        rng = np.random.default_rng(42)
        low = rng.normal(10, 1, 50)
        high = rng.normal(50, 1, 50)
        series = pd.Series(np.concatenate([low, high]))
        result = compute_pre_post_stats(series, 50)
        assert result["effect_size"] > 0

    def test_effect_size_near_zero_for_same_dist(self) -> None:
        rng = np.random.default_rng(42)
        values = rng.normal(10, 1, 100)
        series = pd.Series(values)
        result = compute_pre_post_stats(series, 50)
        assert abs(result["effect_size"]) < 1.0


class TestAnnotateAiEvents:
    def test_matches_nearby_event(self) -> None:
        dates = pd.date_range("2020-01-01", periods=100, freq="W")
        series = pd.Series(np.arange(100, dtype=float), index=dates)

        events_df = pd.DataFrame(
            {
                "date": ["2020-12-01", "2021-06-01"],
                "event": ["Event A", "Event B"],
            }
        )
        # Changepoint at index 50 corresponds roughly to 2020-12-16
        result = annotate_ai_events([50], series, events_df)
        assert len(result) == 1
        assert result.iloc[0]["nearest_event"] == "Event A"

    def test_no_nearby_event(self) -> None:
        dates = pd.date_range("2020-01-01", periods=100, freq="W")
        series = pd.Series(np.arange(100, dtype=float), index=dates)

        # Event far away from any changepoint
        events_df = pd.DataFrame({"date": ["2025-01-01"], "event": ["Far away event"]})
        result = annotate_ai_events([50], series, events_df)
        assert len(result) == 1
        assert result.iloc[0]["nearest_event"] is None

    def test_multiple_changepoints(self) -> None:
        dates = pd.date_range("2020-01-01", periods=100, freq="W")
        series = pd.Series(np.arange(100, dtype=float), index=dates)

        events_df = pd.DataFrame(
            {
                "date": ["2020-06-01", "2021-06-01"],
                "event": ["Event A", "Event B"],
            }
        )
        result = annotate_ai_events([20, 70], series, events_df)
        assert len(result) == 2
