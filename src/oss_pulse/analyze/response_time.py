"""Response time analysis for PR merge latency and first review time."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats  # type: ignore[import-untyped]


def compute_merge_time_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-repo merge time statistics with trend.

    Expects a column 'time_to_merge_hours' and 'repo_name'.
    Trend is the OLS slope of monthly median merge times.
    """
    merged = df.dropna(subset=["time_to_merge_hours"])

    agg = (
        merged.groupby("repo_name")["time_to_merge_hours"]
        .agg(
            median_hours="median",
            p75_hours=lambda x: float(np.percentile(x, 75)),
            p95_hours=lambda x: float(np.percentile(x, 95)),
        )
        .reset_index()
    )

    trends: dict[str, float] = {}
    for repo, group in merged.groupby("repo_name"):
        group = group.copy()
        group["month"] = pd.to_datetime(group["pr_merged_at"]).dt.to_period("M")
        monthly_median = group.groupby("month")["time_to_merge_hours"].median()

        if len(monthly_median) >= 3:
            x = np.arange(len(monthly_median), dtype=float)
            slope: float = float(stats.linregress(x, monthly_median.values).slope)
        else:
            slope = 0.0
        trends[str(repo)] = slope

    agg["trend_slope"] = agg["repo_name"].map(trends).fillna(0.0)
    return agg


def compute_first_review_time(
    df: pd.DataFrame, reviews_df: pd.DataFrame
) -> pd.DataFrame:
    """Compute time from PR creation to first review, aggregated per repo.

    df must have 'repo_name', 'pr_number', 'pr_created_at'.
    reviews_df must have 'repo_name', 'pr_number', 'review_timestamp'.
    """
    reviews_sorted = reviews_df.sort_values("review_timestamp")
    first_reviews = reviews_sorted.groupby(["repo_name", "pr_number"], as_index=False)[
        "review_timestamp"
    ].first()

    merged = (
        df[["repo_name", "pr_number", "pr_created_at"]]
        .drop_duplicates()
        .merge(first_reviews, on=["repo_name", "pr_number"], how="inner")
    )

    created = pd.to_datetime(merged["pr_created_at"], utc=True)
    reviewed = pd.to_datetime(merged["review_timestamp"], utc=True)
    merged["first_review_hours"] = (reviewed - created).dt.total_seconds() / 3600.0

    merged = merged[merged["first_review_hours"] >= 0]

    result = (
        merged.groupby("repo_name")["first_review_hours"]
        .agg(
            median_hours="median",
            p75_hours=lambda x: float(np.percentile(x, 75)),
            p95_hours=lambda x: float(np.percentile(x, 95)),
        )
        .reset_index()
    )

    return result


def merge_time_by_segment(df: pd.DataFrame, segment_col: str) -> pd.DataFrame:
    """Compute merge time percentiles grouped by an arbitrary segment column."""
    filtered = df.dropna(subset=["time_to_merge_hours", segment_col])

    result = (
        filtered.groupby(segment_col)["time_to_merge_hours"]
        .agg(
            median_hours="median",
            p75_hours=lambda x: float(np.percentile(x, 75)),
            p95_hours=lambda x: float(np.percentile(x, 95)),
        )
        .reset_index()
    )

    return result


if __name__ == "__main__":
    data_path = Path("data/processed/pr_events_clean.parquet")
    df = pd.read_parquet(data_path)

    if "time_to_merge_hours" not in df.columns:
        created = pd.to_datetime(df["pr_created_at"], utc=True)
        merged_at = pd.to_datetime(df["pr_merged_at"], utc=True)
        df["time_to_merge_hours"] = (merged_at - created).dt.total_seconds() / 3600.0

    merge_stats = compute_merge_time_stats(df)
    print("Merge time statistics (per repo):")
    print(merge_stats.head(10).to_string(index=False))
    print(f"\nTotal repos analyzed: {len(merge_stats)}")
