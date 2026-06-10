"""Tests for comparative statistical analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from oss_pulse.analyze.comparative import compare_ecosystems, compare_groups


class TestCompareGroups:
    @pytest.fixture()
    def group_df(self) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        return pd.DataFrame(
            {
                "metric_val": np.concatenate(
                    [
                        rng.normal(10, 2, 30),
                        rng.normal(15, 2, 30),
                        rng.normal(20, 2, 30),
                    ]
                ),
                "group": ["A"] * 30 + ["B"] * 30 + ["C"] * 30,
            }
        )

    def test_returns_correct_keys(self, group_df: pd.DataFrame) -> None:
        result = compare_groups(group_df, metric="metric_val", group_col="group")
        assert "kruskal_stat" in result
        assert "kruskal_pvalue" in result
        assert "pairwise" in result

    def test_kruskal_stat_positive(self, group_df: pd.DataFrame) -> None:
        result = compare_groups(group_df, metric="metric_val", group_col="group")
        assert result["kruskal_stat"] > 0

    def test_pvalue_in_valid_range(self, group_df: pd.DataFrame) -> None:
        result = compare_groups(group_df, metric="metric_val", group_col="group")
        assert 0 <= result["kruskal_pvalue"] <= 1

    def test_pairwise_count(self, group_df: pd.DataFrame) -> None:
        result = compare_groups(group_df, metric="metric_val", group_col="group")
        # 3 groups => C(3,2) = 3 pairwise comparisons
        assert len(result["pairwise"]) == 3

    def test_pairwise_has_expected_keys(self, group_df: pd.DataFrame) -> None:
        result = compare_groups(group_df, metric="metric_val", group_col="group")
        for pair in result["pairwise"]:
            assert "group1" in pair
            assert "group2" in pair
            assert "u_stat" in pair
            assert "pvalue" in pair
            assert "pvalue_corrected" in pair

    def test_bonferroni_correction_applied(self, group_df: pd.DataFrame) -> None:
        result = compare_groups(group_df, metric="metric_val", group_col="group")
        for pair in result["pairwise"]:
            # Corrected p-value should be >= raw p-value and <= 1
            assert pair["pvalue_corrected"] >= pair["pvalue"]
            assert pair["pvalue_corrected"] <= 1.0

    def test_fewer_than_two_groups_returns_nan(self) -> None:
        df = pd.DataFrame({"metric_val": [1.0, 2.0], "group": ["A", "A"]})
        result = compare_groups(df, metric="metric_val", group_col="group")
        assert pd.isna(result["kruskal_stat"])
        assert pd.isna(result["kruskal_pvalue"])
        assert result["pairwise"] == []


class TestCompareEcosystems:
    def test_returns_correct_columns(self) -> None:
        df = pd.DataFrame(
            {
                "language": ["Python", "Python", "Go", "Go"],
                "pr_count": [100, 200, 150, 250],
                "merge_rate": [0.7, 0.8, 0.6, 0.9],
            }
        )
        result = compare_ecosystems(df)
        assert "language" in result.columns
        # Should have mean and median for available metric columns
        assert "pr_count_mean" in result.columns
        assert "merge_rate_median" in result.columns

    def test_groups_by_language(self) -> None:
        df = pd.DataFrame(
            {
                "language": ["Python"] * 5 + ["Go"] * 5,
                "pr_count": list(range(10)),
                "merge_rate": [0.5] * 10,
            }
        )
        result = compare_ecosystems(df)
        assert len(result) == 2
        assert set(result["language"]) == {"Python", "Go"}
