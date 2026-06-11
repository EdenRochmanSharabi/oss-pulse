"""Comparative statistical analysis across repo groups and ecosystems."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any, cast

import pandas as pd
from scipy import stats  # type: ignore[import-untyped]


def compare_groups(df: pd.DataFrame, metric: str, group_col: str) -> dict[str, Any]:
    """Compare a metric across groups using Kruskal-Wallis and pairwise Mann-Whitney U.

    Pairwise p-values are corrected with Bonferroni.
    """
    groups: dict[str, pd.Series] = {}
    for name, group_df in df.groupby(group_col):
        values = group_df[metric].dropna()
        if len(values) >= 2:
            groups[str(name)] = values

    if len(groups) < 2:
        return {
            "kruskal_stat": float("nan"),
            "kruskal_pvalue": float("nan"),
            "pairwise": [],
        }

    group_arrays = list(groups.values())
    kruskal_stat, kruskal_pvalue = stats.kruskal(*group_arrays)

    group_names = list(groups.keys())
    n_comparisons = len(list(combinations(group_names, 2)))

    pairwise: list[dict[str, Any]] = []
    for g1, g2 in combinations(group_names, 2):
        u_stat, pvalue = stats.mannwhitneyu(
            groups[g1], groups[g2], alternative="two-sided"
        )
        pvalue_corrected = min(pvalue * n_comparisons, 1.0)
        pairwise.append(
            {
                "group1": g1,
                "group2": g2,
                "u_stat": float(u_stat),
                "pvalue": float(pvalue),
                "pvalue_corrected": pvalue_corrected,
            }
        )

    return {
        "kruskal_stat": float(kruskal_stat),
        "kruskal_pvalue": float(kruskal_pvalue),
        "pairwise": pairwise,
    }


def compare_ecosystems(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate key metrics by programming language for ecosystem comparison.

    Expected columns: language, pr_count, merge_rate,
    median_merge_time_hours, unique_contributors.
    """
    metric_cols = [
        "pr_count",
        "merge_rate",
        "median_merge_time_hours",
        "unique_contributors",
    ]
    available = [c for c in metric_cols if c in monthly_df.columns]

    agg_funcs: dict[str, list[str]] = {col: ["mean", "median"] for col in available}

    result = monthly_df.groupby("language").agg(agg_funcs)
    multi_cols = cast(list[tuple[str, str]], result.columns.tolist())
    result.columns = pd.Index([f"{col}_{stat}" for col, stat in multi_cols])
    result = result.reset_index()

    return result


if __name__ == "__main__":
    data_path = Path("data/processed/repo_monthly.parquet")
    repos_path = Path("data/raw/top_repos.parquet")
    df = pd.read_parquet(data_path)

    if repos_path.exists():
        repos = pd.read_parquet(repos_path)[["repo_name", "org_type"]]
        df = df.merge(repos, on="repo_name", how="left")

    if "org_type" not in df.columns:
        print("No org_type column, skipping.")
        raise SystemExit(0)

    print("Comparing org_types on merge_rate:")
    comparison = compare_groups(df, metric="merge_rate", group_col="org_type")

    print(
        f"  Kruskal-Wallis H = {comparison['kruskal_stat']:.4f}, "
        f"p = {comparison['kruskal_pvalue']:.4f}"
    )

    print("\n  Pairwise Mann-Whitney U tests (Bonferroni corrected):")
    for pair in comparison["pairwise"]:
        print(
            f"    {pair['group1']} vs {pair['group2']}: "
            f"U = {pair['u_stat']:.1f}, "
            f"p = {pair['pvalue']:.4f}, "
            f"p_corrected = {pair['pvalue_corrected']:.4f}"
        )
