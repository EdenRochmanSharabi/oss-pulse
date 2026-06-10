"""Composite health index for open-source project vitality scoring."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

DEFAULT_WEIGHTS: dict[str, float] = {
    "response_time": 0.25,
    "merge_rate": 0.20,
    "diversity": 0.20,
    "trend": 0.15,
    "bus_factor": 0.20,
}


def _gini(values: np.ndarray) -> float:
    """Compute the Gini coefficient of a 1-D array of non-negative values."""
    arr = np.sort(values.astype(float))
    n = len(arr)
    if n == 0 or arr.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float(
        (2.0 * np.sum(index * arr) - (n + 1) * np.sum(arr)) / (n * np.sum(arr))
    )


def _normalize_scores(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Min-max normalize specified columns to the 0-100 range."""
    out = df.copy()
    for col in columns:
        col_min = out[col].min()
        col_max = out[col].max()
        if col_max == col_min:
            out[col] = 50.0
        else:
            out[col] = 100.0 * (out[col] - col_min) / (col_max - col_min)
    return out


def compute_health_components(repo_monthly: pd.DataFrame) -> pd.DataFrame:
    """Compute five raw health-component scores per repo, then normalize to 0-100."""
    repos = repo_monthly.groupby("repo_name")
    records: list[dict[str, object]] = []

    for repo_name, grp in repos:
        grp_sorted = grp.sort_values(["year", "month"])

        median_merge = grp_sorted["median_merge_time_hours"].median()
        response_time_score = (
            1.0 / (1.0 + median_merge) if pd.notna(median_merge) else 0.0
        )

        merge_rate_score = float(grp_sorted["merge_rate"].mean() * 100)

        diversity_score = float(grp_sorted["unique_contributors"].mean())

        last_12 = grp_sorted.tail(12)
        if len(last_12) >= 2:
            x = np.arange(len(last_12), dtype=float)
            slope: float = float(
                scipy_stats.linregress(x, last_12["pr_count"].values).slope
            )
            trend_score = slope
        else:
            trend_score = 0.0

        contributor_counts = grp_sorted["unique_contributors"].values.astype(float)
        bus_factor_score = (
            1.0 - _gini(contributor_counts) if len(contributor_counts) > 1 else 0.5
        )

        records.append(
            {
                "repo_name": repo_name,
                "response_time_score": response_time_score,
                "merge_rate_score": merge_rate_score,
                "diversity_score": diversity_score,
                "trend_score": trend_score,
                "bus_factor_score": bus_factor_score,
            }
        )

    components = pd.DataFrame(records)
    score_cols = [
        "response_time_score",
        "merge_rate_score",
        "diversity_score",
        "trend_score",
        "bus_factor_score",
    ]
    components = _normalize_scores(components, score_cols)
    return components


def compute_health_index(
    components: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> pd.Series:  # type: ignore[type-arg]
    """Compute a weighted-average health index from normalized component scores."""
    w = weights if weights is not None else DEFAULT_WEIGHTS

    weight_map: dict[str, str] = {
        "response_time": "response_time_score",
        "merge_rate": "merge_rate_score",
        "diversity": "diversity_score",
        "trend": "trend_score",
        "bus_factor": "bus_factor_score",
    }

    health: pd.Series = pd.Series(0.0, index=components.index)  # type: ignore[type-arg]
    for component, col_name in weight_map.items():
        health = health + components[col_name] * w[component]

    return health


def rank_repos(health_df: pd.DataFrame) -> pd.DataFrame:
    """Sort repos by health_index descending and assign integer ranks."""
    out = health_df.sort_values("health_index", ascending=False).reset_index(drop=True)
    out["rank"] = range(1, len(out) + 1)
    return out


if __name__ == "__main__":
    data_path = Path("data/processed/repo_monthly.parquet")
    df = pd.read_parquet(data_path)

    print(f"Loaded {len(df)} monthly records for {df['repo_name'].nunique()} repos")

    components = compute_health_components(df)
    components["health_index"] = compute_health_index(components)
    ranked = rank_repos(components)

    print("\nTop 10 repos by health index:")
    display_cols = ["rank", "repo_name", "health_index"]
    print(ranked[display_cols].head(10).to_string(index=False))
