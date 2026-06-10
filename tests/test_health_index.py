"""Tests for OSS health index computation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def mock_repo_monthly() -> pd.DataFrame:
    """Minimal monthly data for health index tests."""
    rng = np.random.default_rng(42)
    rows = []
    for repo_idx in range(5):
        repo = f"org-{repo_idx:03d}/project-{repo_idx:03d}"
        for month in range(1, 13):
            n_contributors = int(rng.integers(5, 50))
            rows.append(
                {
                    "repo_name": repo,
                    "year": 2025,
                    "month": month,
                    "pr_count": int(rng.integers(10, 200)),
                    "merge_rate": float(rng.uniform(0.3, 0.9)),
                    "unique_contributors": n_contributors,
                    "median_merge_time_hours": float(rng.lognormal(3, 1)),
                    "bot_ratio": float(rng.uniform(0, 0.3)),
                    "merged_count": int(rng.integers(5, 150)),
                    "closed_count": int(rng.integers(0, 30)),
                    "abandoned_count": int(rng.integers(0, 10)),
                }
            )
    return pd.DataFrame(rows)


def test_health_components_range(mock_repo_monthly: pd.DataFrame) -> None:
    from oss_pulse.analyze.health_index import compute_health_components

    components = compute_health_components(mock_repo_monthly)
    score_cols = [c for c in components.columns if c.endswith("_score")]
    assert len(score_cols) >= 4
    for col in score_cols:
        assert components[col].min() >= 0, f"{col} has values < 0"
        assert components[col].max() <= 100, f"{col} has values > 100"


def test_health_index_weighted_average(mock_repo_monthly: pd.DataFrame) -> None:
    from oss_pulse.analyze.health_index import (
        compute_health_components,
        compute_health_index,
    )

    components = compute_health_components(mock_repo_monthly)
    index = compute_health_index(components)
    assert len(index) == len(components)
    assert index.min() >= 0
    assert index.max() <= 100


def test_ranking_is_sorted(mock_repo_monthly: pd.DataFrame) -> None:
    from oss_pulse.analyze.health_index import (
        compute_health_components,
        compute_health_index,
        rank_repos,
    )

    components = compute_health_components(mock_repo_monthly)
    components["health_index"] = compute_health_index(components)
    ranked = rank_repos(components)
    assert ranked["health_index"].is_monotonic_decreasing


def test_default_weights_sum_to_one() -> None:
    from oss_pulse.analyze.health_index import DEFAULT_WEIGHTS

    assert abs(sum(DEFAULT_WEIGHTS.values()) - 1.0) < 1e-9
