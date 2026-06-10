"""Tests for feature engineering and aggregation."""

from __future__ import annotations

import pandas as pd


def test_pr_features_columns(featured_events: pd.DataFrame) -> None:
    expected = {
        "time_to_merge_hours",
        "pr_size_bucket",
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
    }
    assert expected.issubset(set(featured_events.columns))


def test_pr_features_dtypes(featured_events: pd.DataFrame) -> None:
    assert featured_events["hour"].dtype in ("int64", "int32")
    assert featured_events["day_of_week"].dtype in ("int64", "int32")
    assert featured_events["month"].dtype in ("int64", "int32")
    assert featured_events["is_weekend"].dtype == bool


def test_pr_size_buckets(featured_events: pd.DataFrame) -> None:
    valid_buckets = {"small", "medium", "large"}
    actual = set(featured_events["pr_size_bucket"].dropna().unique())
    assert actual.issubset(valid_buckets)


def test_hour_range(featured_events: pd.DataFrame) -> None:
    assert featured_events["hour"].min() >= 0
    assert featured_events["hour"].max() <= 23


def test_day_of_week_range(featured_events: pd.DataFrame) -> None:
    assert featured_events["day_of_week"].min() >= 0
    assert featured_events["day_of_week"].max() <= 6


def test_merge_time_non_negative(featured_events: pd.DataFrame) -> None:
    valid = featured_events["time_to_merge_hours"].dropna()
    assert (valid >= 0).all()


def test_repo_monthly_shape(repo_monthly: pd.DataFrame) -> None:
    assert len(repo_monthly) > 0
    expected = {"repo_name", "year", "month", "pr_count", "merge_rate"}
    assert expected.issubset(set(repo_monthly.columns))


def test_repo_monthly_merge_rate_range(repo_monthly: pd.DataFrame) -> None:
    assert repo_monthly["merge_rate"].min() >= 0
    assert repo_monthly["merge_rate"].max() <= 1.0


def test_repo_monthly_no_nan_counts(repo_monthly: pd.DataFrame) -> None:
    assert repo_monthly["pr_count"].isna().sum() == 0
