"""Extra tests for weekly aggregation and feature completeness."""

from __future__ import annotations

import pandas as pd


def test_repo_weekly_columns(featured_events: pd.DataFrame) -> None:
    from oss_pulse.transform.features import build_repo_weekly

    weekly = build_repo_weekly(featured_events)
    expected = {"repo_name", "year_week", "pr_count", "merge_rate"}
    assert expected.issubset(set(weekly.columns))
    assert len(weekly) > 0


def test_repo_weekly_merge_rate(featured_events: pd.DataFrame) -> None:
    from oss_pulse.transform.features import build_repo_weekly

    weekly = build_repo_weekly(featured_events)
    valid = weekly["merge_rate"].dropna()
    assert (valid >= 0).all()
    assert (valid <= 1).all()


def test_repo_monthly_has_all_repos(
    featured_events: pd.DataFrame, repo_monthly: pd.DataFrame
) -> None:
    expected_repos = featured_events["repo_name"].nunique()
    actual_repos = repo_monthly["repo_name"].nunique()
    assert actual_repos == expected_repos
