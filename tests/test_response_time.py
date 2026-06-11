"""Tests for response time analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from oss_pulse.analyze.response_time import (
    compute_first_review_time,
    compute_merge_time_stats,
    merge_time_by_segment,
)


@pytest.fixture()
def merge_time_df() -> pd.DataFrame:
    """Minimal DataFrame with time_to_merge_hours, repo_name, and pr_merged_at."""
    rng = np.random.default_rng(42)
    n = 200
    repos = [f"org/repo-{i % 3}" for i in range(n)]
    merge_hours = rng.lognormal(3, 1, n)
    base_date = pd.Timestamp("2024-01-01", tz="UTC")
    merged_at = [base_date + pd.Timedelta(days=int(i)) for i in range(n)]
    return pd.DataFrame(
        {
            "repo_name": repos,
            "time_to_merge_hours": merge_hours,
            "pr_merged_at": merged_at,
        }
    )


class TestComputeMergeTimeStats:
    def test_returns_expected_columns(self, merge_time_df: pd.DataFrame) -> None:
        result = compute_merge_time_stats(merge_time_df)
        expected = {"repo_name", "median_hours", "p75_hours", "p95_hours"}
        assert expected.issubset(set(result.columns))

    def test_one_row_per_repo(self, merge_time_df: pd.DataFrame) -> None:
        result = compute_merge_time_stats(merge_time_df)
        n_repos = merge_time_df["repo_name"].nunique()
        assert len(result) == n_repos

    def test_percentile_ordering(self, merge_time_df: pd.DataFrame) -> None:
        result = compute_merge_time_stats(merge_time_df)
        for _, row in result.iterrows():
            assert row["median_hours"] <= row["p75_hours"]
            assert row["p75_hours"] <= row["p95_hours"]

    def test_trend_slope_present(self, merge_time_df: pd.DataFrame) -> None:
        result = compute_merge_time_stats(merge_time_df)
        assert "trend_slope" in result.columns

    def test_handles_nan_merge_hours(self) -> None:
        df = pd.DataFrame(
            {
                "repo_name": ["a/b", "a/b", "a/b"],
                "time_to_merge_hours": [10.0, float("nan"), 20.0],
                "pr_merged_at": pd.to_datetime(
                    ["2024-01-01", "2024-02-01", "2024-03-01"], utc=True
                ),
            }
        )
        result = compute_merge_time_stats(df)
        assert len(result) == 1
        assert result.iloc[0]["median_hours"] == 15.0


class TestMergeTimeBySegment:
    def test_groups_by_segment(self) -> None:
        df = pd.DataFrame(
            {
                "time_to_merge_hours": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
                "size_bucket": ["small", "small", "medium", "medium", "large", "large"],
            }
        )
        result = merge_time_by_segment(df, "size_bucket")
        assert len(result) == 3
        assert set(result["size_bucket"]) == {"small", "medium", "large"}

    def test_returns_percentile_columns(self) -> None:
        df = pd.DataFrame(
            {
                "time_to_merge_hours": [10.0, 20.0, 30.0, 40.0],
                "group": ["A", "A", "B", "B"],
            }
        )
        result = merge_time_by_segment(df, "group")
        assert "median_hours" in result.columns
        assert "p75_hours" in result.columns
        assert "p95_hours" in result.columns

    def test_drops_nan_in_segment(self) -> None:
        df = pd.DataFrame(
            {
                "time_to_merge_hours": [10.0, 20.0, 30.0],
                "group": ["A", None, "B"],
            }
        )
        result = merge_time_by_segment(df, "group")
        assert len(result) == 2


class TestComputeFirstReviewTime:
    def test_returns_expected_columns(self) -> None:
        pr_df = pd.DataFrame(
            {
                "repo_name": ["org/a", "org/a", "org/a"],
                "pr_number": [1, 2, 3],
                "pr_created_at": pd.to_datetime(
                    ["2024-01-01", "2024-01-02", "2024-01-03"], utc=True
                ),
            }
        )
        reviews_df = pd.DataFrame(
            {
                "repo_name": ["org/a", "org/a"],
                "pr_number": [1, 2],
                "review_timestamp": pd.to_datetime(
                    ["2024-01-01T12:00:00", "2024-01-03T06:00:00"], utc=True
                ),
            }
        )
        result = compute_first_review_time(pr_df, reviews_df)
        expected_cols = {"repo_name", "median_hours", "p75_hours", "p95_hours"}
        assert expected_cols.issubset(set(result.columns))

    def test_review_hours_non_negative(self) -> None:
        pr_df = pd.DataFrame(
            {
                "repo_name": ["org/a"] * 5,
                "pr_number": list(range(5)),
                "pr_created_at": pd.to_datetime(["2024-01-01"] * 5, utc=True),
            }
        )
        reviews_df = pd.DataFrame(
            {
                "repo_name": ["org/a"] * 5,
                "pr_number": list(range(5)),
                "review_timestamp": pd.to_datetime(
                    [
                        "2024-01-01T06:00:00",
                        "2024-01-01T12:00:00",
                        "2024-01-02T00:00:00",
                        "2024-01-02T12:00:00",
                        "2024-01-03T00:00:00",
                    ],
                    utc=True,
                ),
            }
        )
        result = compute_first_review_time(pr_df, reviews_df)
        assert (result["median_hours"] >= 0).all()
