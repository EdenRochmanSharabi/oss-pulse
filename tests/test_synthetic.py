"""Tests for synthetic dataset generator."""

from __future__ import annotations

import pandas as pd
import pytest

from oss_pulse.extract.synthetic import generate_synthetic_dataset


class TestGenerateSyntheticDataset:
    @pytest.fixture(scope="class")
    def dataset(self) -> dict[str, pd.DataFrame]:
        return generate_synthetic_dataset(n_repos=3, years=1, seed=123)

    def test_returns_correct_keys(self, dataset: dict[str, pd.DataFrame]) -> None:
        assert set(dataset.keys()) == {"repos", "pr_events", "pr_reviews"}

    def test_dataframes_non_empty(self, dataset: dict[str, pd.DataFrame]) -> None:
        for key, df in dataset.items():
            assert len(df) > 0, f"{key} DataFrame is empty"

    def test_repos_expected_columns(self, dataset: dict[str, pd.DataFrame]) -> None:
        expected = {
            "repo_name",
            "org",
            "org_type",
            "language",
            "category",
            "contributors",
            "event_count",
            "is_dying",
        }
        assert expected.issubset(set(dataset["repos"].columns))

    def test_pr_events_expected_columns(self, dataset: dict[str, pd.DataFrame]) -> None:
        expected = {
            "repo_name",
            "pr_number",
            "action",
            "state",
            "merged",
            "pr_created_at",
            "author",
            "additions",
            "deletions",
            "changed_files",
            "event_timestamp",
        }
        assert expected.issubset(set(dataset["pr_events"].columns))

    def test_pr_reviews_expected_columns(
        self, dataset: dict[str, pd.DataFrame]
    ) -> None:
        expected = {"repo_name", "pr_number", "reviewer", "review_timestamp"}
        assert expected.issubset(set(dataset["pr_reviews"].columns))

    def test_repo_count_matches_input(self, dataset: dict[str, pd.DataFrame]) -> None:
        assert len(dataset["repos"]) == 3

    def test_merged_values_are_strings(self, dataset: dict[str, pd.DataFrame]) -> None:
        assert dataset["pr_events"]["merged"].isin(["true", "false"]).all()

    def test_seed_reproducibility(self) -> None:
        d1 = generate_synthetic_dataset(n_repos=2, years=1, seed=77)
        d2 = generate_synthetic_dataset(n_repos=2, years=1, seed=77)
        pd.testing.assert_frame_equal(d1["repos"], d2["repos"])
        pd.testing.assert_frame_equal(d1["pr_events"], d2["pr_events"])
