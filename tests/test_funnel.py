"""Tests for contributor retention funnel analysis."""

from __future__ import annotations

import pandas as pd
import pytest

from oss_pulse.analyze.funnel import (
    build_contributor_funnel,
    compute_retention_rates,
    funnel_by_segment,
)


@pytest.fixture()
def funnel_pr_df() -> pd.DataFrame:
    """Minimal PR data with is_bot, author, and pr_number columns."""
    rows = []
    # Create several authors with varying PR counts:
    # user-0: 25 PRs (regular/maintainer)
    # user-1: 12 PRs
    # user-2: 6 PRs
    # user-3: 2 PRs
    # user-4: 1 PR
    # bot: 10 PRs (should be excluded)
    authors = {
        "user-0": 25,
        "user-1": 12,
        "user-2": 6,
        "user-3": 2,
        "user-4": 1,
    }
    pr_num = 0
    for author, count in authors.items():
        for _ in range(count):
            pr_num += 1
            rows.append(
                {
                    "author": author,
                    "pr_number": pr_num,
                    "is_bot": False,
                    "repo_name": "org/repo",
                }
            )
    # Bot PRs
    for _ in range(10):
        pr_num += 1
        rows.append(
            {
                "author": "dependabot[bot]",
                "pr_number": pr_num,
                "is_bot": True,
                "repo_name": "org/repo",
            }
        )
    return pd.DataFrame(rows)


class TestBuildContributorFunnel:
    def test_returns_dataframe_with_expected_columns(
        self, funnel_pr_df: pd.DataFrame
    ) -> None:
        result = build_contributor_funnel(funnel_pr_df)
        assert isinstance(result, pd.DataFrame)
        assert "stage" in result.columns
        assert "count" in result.columns
        assert "percentage" in result.columns

    def test_first_stage_percentage_is_100(self, funnel_pr_df: pd.DataFrame) -> None:
        result = build_contributor_funnel(funnel_pr_df)
        # First stage (1st_pr) should have 100% since all authors have >= 1 PR
        first_stage = result[result["stage"] == "1st_pr"]
        assert first_stage.iloc[0]["percentage"] == 100.0

    def test_funnel_is_monotonically_decreasing(
        self, funnel_pr_df: pd.DataFrame
    ) -> None:
        result = build_contributor_funnel(funnel_pr_df)
        counts = result["count"].tolist()
        for i in range(1, len(counts)):
            assert counts[i] <= counts[i - 1]

    def test_bots_excluded(self, funnel_pr_df: pd.DataFrame) -> None:
        result = build_contributor_funnel(funnel_pr_df)
        first_stage = result[result["stage"] == "1st_pr"]
        # 5 human authors, bot excluded
        assert first_stage.iloc[0]["count"] == 5

    def test_expected_stage_counts(self, funnel_pr_df: pd.DataFrame) -> None:
        result = build_contributor_funnel(funnel_pr_df)
        stage_counts = dict(zip(result["stage"], result["count"], strict=True))
        # user-0(25), user-1(12), user-2(6), user-3(2), user-4(1) = 5 human authors
        assert stage_counts["1st_pr"] == 5  # all >= 1
        assert stage_counts["2nd_pr"] == 4  # user-0,1,2,3 >= 2
        assert stage_counts["5th_pr"] == 3  # user-0,1,2 >= 5
        assert stage_counts["10th_pr"] == 2  # user-0,1 >= 10
        assert stage_counts["regular"] == 1  # user-0 >= 20


class TestComputeRetentionRates:
    def test_adds_retention_rate_column(self, funnel_pr_df: pd.DataFrame) -> None:
        funnel = build_contributor_funnel(funnel_pr_df)
        result = compute_retention_rates(funnel)
        assert "retention_rate" in result.columns

    def test_first_retention_is_none(self, funnel_pr_df: pd.DataFrame) -> None:
        funnel = build_contributor_funnel(funnel_pr_df)
        result = compute_retention_rates(funnel)
        assert result.iloc[0]["retention_rate"] is None

    def test_retention_values_between_0_and_1(self, funnel_pr_df: pd.DataFrame) -> None:
        funnel = build_contributor_funnel(funnel_pr_df)
        result = compute_retention_rates(funnel)
        for i in range(1, len(result)):
            rate = result.iloc[i]["retention_rate"]
            if rate is not None:
                assert 0 <= rate <= 1.0


class TestFunnelBySegment:
    def test_groups_by_segment(self) -> None:
        rows = []
        pr_num = 0
        for org in ["company", "community"]:
            for author_idx in range(3):
                for _ in range(5):
                    pr_num += 1
                    rows.append(
                        {
                            "author": f"{org}-user-{author_idx}",
                            "pr_number": pr_num,
                            "is_bot": False,
                            "org_type": org,
                        }
                    )
        df = pd.DataFrame(rows)
        result = funnel_by_segment(df, "org_type")
        assert len(result) > 0
        assert "org_type" in result.columns
        segments = result["org_type"].unique()
        assert set(segments) == {"company", "community"}

    def test_empty_segment_returns_empty(self) -> None:
        # All bots, so human_df is empty
        result = funnel_by_segment(
            pd.DataFrame(
                {
                    "author": ["bot"],
                    "pr_number": [1],
                    "is_bot": [True],
                    "org_type": ["company"],
                }
            ),
            "org_type",
        )
        assert len(result) == 0
