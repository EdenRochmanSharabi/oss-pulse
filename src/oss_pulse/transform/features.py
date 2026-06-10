"""Feature engineering for PR-level and repo-level aggregations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def build_pr_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add temporal and size features to each PR event row."""
    out = df.copy()

    has_merge_time = out["pr_merged_at"].notna() & out["pr_created_at"].notna()
    out["time_to_merge_hours"] = np.where(
        has_merge_time,
        (out["pr_merged_at"] - out["pr_created_at"]).dt.total_seconds() / 3600,
        np.nan,
    )

    total_lines = out["additions"] + out["deletions"]
    out["pr_size_bucket"] = pd.cut(
        total_lines,
        bins=[-1, 49, 500, np.inf],
        labels=["small", "medium", "large"],
    )

    out["hour"] = out["pr_created_at"].dt.hour
    out["day_of_week"] = out["pr_created_at"].dt.dayofweek
    out["month"] = out["pr_created_at"].dt.month
    out["is_weekend"] = out["day_of_week"] >= 5

    return out


def build_repo_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate PR metrics by repo and calendar month."""
    out = df.copy()
    out["year"] = out["pr_created_at"].dt.year
    out["_month"] = out["pr_created_at"].dt.month

    grouped = out.groupby(["repo_name", "year", "_month"])
    agg = grouped.agg(
        pr_count=("pr_number", "nunique"),
        merged_count=("pr_outcome", lambda x: (x == "merged").sum()),
        closed_count=("pr_outcome", lambda x: (x == "closed").sum()),
        abandoned_count=("pr_outcome", lambda x: (x == "abandoned").sum()),
        unique_contributors=("author", "nunique"),
        median_merge_time_hours=("time_to_merge_hours", "median"),
        bot_pr_count=("is_bot", "sum"),
    ).reset_index()

    agg = agg.rename(columns={"_month": "month"})
    agg["merge_rate"] = agg["merged_count"] / agg["pr_count"].replace(0, np.nan)
    agg["bot_ratio"] = agg["bot_pr_count"] / agg["pr_count"].replace(0, np.nan)
    agg = agg.drop(columns=["bot_pr_count"])

    return agg


def build_repo_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate PR metrics by repo and ISO calendar week."""
    out = df.copy()
    iso = out["pr_created_at"].dt.isocalendar()
    out["year"] = iso["year"].astype(int)
    out["week"] = iso["week"].astype(int)
    out["year_week"] = (
        out["year"].astype(str) + "-W" + out["week"].astype(str).str.zfill(2)
    )

    grouped = out.groupby(["repo_name", "year_week"])
    agg = grouped.agg(
        pr_count=("pr_number", "nunique"),
        merged_count=("pr_outcome", lambda x: (x == "merged").sum()),
        closed_count=("pr_outcome", lambda x: (x == "closed").sum()),
        abandoned_count=("pr_outcome", lambda x: (x == "abandoned").sum()),
        unique_contributors=("author", "nunique"),
        median_merge_time_hours=("time_to_merge_hours", "median"),
        bot_pr_count=("is_bot", "sum"),
    ).reset_index()

    agg["merge_rate"] = agg["merged_count"] / agg["pr_count"].replace(0, np.nan)
    agg["bot_ratio"] = agg["bot_pr_count"] / agg["pr_count"].replace(0, np.nan)
    agg = agg.drop(columns=["bot_pr_count"])

    return agg


if __name__ == "__main__":
    classified_path = Path("data/processed/pr_events_classified.parquet")
    output_dir = Path("data/processed")

    df = pd.read_parquet(classified_path)
    df = build_pr_features(df)

    df.to_parquet(output_dir / "pr_events_featured.parquet", index=False)
    print(f"PR features: {len(df)} rows")

    monthly = build_repo_monthly(df)
    monthly.to_parquet(output_dir / "repo_monthly.parquet", index=False)
    print(f"Repo monthly: {len(monthly)} rows")

    weekly = build_repo_weekly(df)
    weekly.to_parquet(output_dir / "repo_weekly.parquet", index=False)
    print(f"Repo weekly: {len(weekly)} rows")
