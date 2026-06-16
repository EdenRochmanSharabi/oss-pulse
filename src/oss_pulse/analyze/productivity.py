"""Productivity and merge-rate analysis segmented by author type."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

AUTHOR_GROUPS = ["first-timer", "regular", "maintainer"]


def compute_productivity_timeseries(
    pr_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute monthly PRs-per-contributor for each author_class.

    Returns a DataFrame with columns: year_month, author_class,
    n_prs, n_authors, prs_per_person.
    """
    human = pr_df[~pr_df["is_bot"]].copy()
    unique_prs = human.drop_duplicates(subset=["repo_name", "pr_number"])
    unique_prs["year_month"] = unique_prs["pr_created_at"].dt.to_period("M")

    records: list[dict[str, object]] = []
    for group in AUTHOR_GROUPS:
        g = unique_prs[unique_prs["author_class"] == group]
        monthly = g.groupby("year_month").agg(
            n_prs=("pr_number", "count"),
            n_authors=("author", "nunique"),
        )
        for period, row in monthly.iterrows():
            records.append({
                "year_month": str(period),
                "author_class": group,
                "n_prs": int(row["n_prs"]),
                "n_authors": int(row["n_authors"]),
                "prs_per_person": round(row["n_prs"] / row["n_authors"], 3),
            })

    return pd.DataFrame(records)


def compute_merge_rate_timeseries(
    pr_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute monthly merge rate for each author_class.

    Returns a DataFrame with columns: year_month, author_class,
    n_prs, n_merged, merge_rate.
    """
    human = pr_df[~pr_df["is_bot"]].copy()
    unique_prs = human.drop_duplicates(subset=["repo_name", "pr_number"])
    unique_prs["year_month"] = unique_prs["pr_created_at"].dt.to_period("M")

    records: list[dict[str, object]] = []
    for group in AUTHOR_GROUPS:
        g = unique_prs[unique_prs["author_class"] == group]
        monthly = g.groupby("year_month").agg(
            n_prs=("pr_number", "count"),
            n_merged=("pr_outcome", lambda x: int((x == "merged").sum())),
        )
        for period, row in monthly.iterrows():
            n_prs = int(row["n_prs"])
            n_merged = int(row["n_merged"])
            records.append({
                "year_month": str(period),
                "author_class": group,
                "n_prs": n_prs,
                "n_merged": n_merged,
                "merge_rate": round(n_merged / n_prs, 4) if n_prs > 0 else 0.0,
            })

    return pd.DataFrame(records)


def compute_productivity_stats(pr_df: pd.DataFrame) -> dict[str, Any]:
    """Compute summary stats for stats.json: productivity and merge rate
    by author_class, comparing pre-2023 vs post-2023 eras."""
    human = pr_df[~pr_df["is_bot"]].copy()
    unique_prs = human.drop_duplicates(subset=["repo_name", "pr_number"])
    unique_prs["year_month"] = unique_prs["pr_created_at"].dt.to_period("M")

    result: dict[str, Any] = {}

    for group in AUTHOR_GROUPS:
        g = unique_prs[unique_prs["author_class"] == group]
        pre = g[g["pr_created_at"] < "2023-01-01"]
        post = g[g["pr_created_at"] >= "2023-01-01"]

        pre_monthly = pre.groupby("year_month").agg(
            n_prs=("pr_number", "count"), n_auth=("author", "nunique"),
        )
        post_monthly = post.groupby("year_month").agg(
            n_prs=("pr_number", "count"), n_auth=("author", "nunique"),
        )

        pre_prod = float((pre_monthly["n_prs"] / pre_monthly["n_auth"]).median())
        post_prod = float((post_monthly["n_prs"] / post_monthly["n_auth"]).median())

        pre_merge = float((pre["pr_outcome"] == "merged").mean())
        post_merge = float((post["pr_outcome"] == "merged").mean())

        pct_change = ((post_prod - pre_prod) / pre_prod * 100) if pre_prod > 0 else 0.0

        result[group] = {
            "pre_2023": {
                "prs_per_person_month": round(pre_prod, 2),
                "merge_rate_pct": round(pre_merge * 100, 1),
            },
            "post_2023": {
                "prs_per_person_month": round(post_prod, 2),
                "merge_rate_pct": round(post_merge * 100, 1),
            },
            "productivity_change_pct": round(pct_change, 1),
        }

    return result


def run_productivity_analysis(
    data_path: Path | None = None,
    stats_path: Path | None = None,
) -> dict[str, Any]:
    """Run the full productivity analysis and update stats.json."""
    if data_path is None:
        data_path = Path("data/processed/pr_events_featured.parquet")
    if stats_path is None:
        stats_path = Path("data/processed/stats.json")

    print("Loading data...")
    df = pd.read_parquet(data_path)
    print(f"  Loaded {len(df):,} PR events")

    print("Computing productivity by author type...")
    result = compute_productivity_stats(df)

    for group in AUTHOR_GROUPS:
        g = result[group]
        pre = g["pre_2023"]
        post = g["post_2023"]
        print(
            f"  {group:>12s}: "
            f"{pre['prs_per_person_month']:.2f} -> {post['prs_per_person_month']:.2f} "
            f"({g['productivity_change_pct']:+.0f}%), "
            f"merge rate {pre['merge_rate_pct']:.1f}% -> {post['merge_rate_pct']:.1f}%"
        )

    stats_data: dict[str, Any] = {}
    if stats_path.exists():
        with open(stats_path) as f:
            stats_data = json.load(f)

    stats_data["productivity_by_author_type"] = result

    with open(stats_path, "w") as f:
        json.dump(stats_data, f, indent=2)
    print(f"\nUpdated {stats_path}")

    return result


if __name__ == "__main__":
    run_productivity_analysis()
