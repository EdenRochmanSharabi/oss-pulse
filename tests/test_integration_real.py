"""Integration tests using real GitHub data (deepseek-ai/awesome-deepseek-integration).

This file runs the full pipeline end-to-end on a real 450-PR dataset
committed as a test fixture. If this test breaks, the pipeline is
incompatible with real GitHub API data.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "real_repo_sample.parquet"


@pytest.fixture(scope="module")
def raw_real() -> pd.DataFrame:
    return pd.read_parquet(FIXTURE)


@pytest.fixture(scope="module")
def cleaned_real(raw_real: pd.DataFrame) -> pd.DataFrame:
    from oss_pulse.transform.clean import clean_pr_events

    return clean_pr_events(raw_real)


@pytest.fixture(scope="module")
def classified_real(cleaned_real: pd.DataFrame) -> pd.DataFrame:
    from oss_pulse.transform.classify import apply_classifications

    return apply_classifications(cleaned_real)


@pytest.fixture(scope="module")
def featured_real(classified_real: pd.DataFrame) -> pd.DataFrame:
    from oss_pulse.transform.features import build_pr_features

    return build_pr_features(classified_real)


@pytest.fixture(scope="module")
def monthly_real(featured_real: pd.DataFrame) -> pd.DataFrame:
    from oss_pulse.transform.features import build_repo_monthly

    return build_repo_monthly(featured_real)


@pytest.fixture(scope="module")
def weekly_real(featured_real: pd.DataFrame) -> pd.DataFrame:
    from oss_pulse.transform.features import build_repo_weekly

    return build_repo_weekly(featured_real)


class TestCleanReal:
    def test_no_rows_lost(
        self, raw_real: pd.DataFrame, cleaned_real: pd.DataFrame
    ) -> None:
        assert len(cleaned_real) == len(raw_real)

    def test_timestamps_are_utc(self, cleaned_real: pd.DataFrame) -> None:
        assert cleaned_real["pr_created_at"].dt.tz is not None

    def test_state_normalized(self, cleaned_real: pd.DataFrame) -> None:
        assert "merged" not in cleaned_real["state"].values
        assert set(cleaned_real["state"].unique()).issubset({"closed", "open"})

    def test_no_null_authors(self, cleaned_real: pd.DataFrame) -> None:
        assert cleaned_real["author"].isna().sum() == 0


class TestClassifyReal:
    def test_has_outcome_column(self, classified_real: pd.DataFrame) -> None:
        assert "pr_outcome" in classified_real.columns

    def test_outcome_values(self, classified_real: pd.DataFrame) -> None:
        valid = {"merged", "closed", "abandoned", "open"}
        assert set(classified_real["pr_outcome"].unique()).issubset(valid)

    def test_has_author_class(self, classified_real: pd.DataFrame) -> None:
        assert "author_class" in classified_real.columns

    def test_author_class_values(self, classified_real: pd.DataFrame) -> None:
        valid = {"bot", "first-timer", "regular", "maintainer"}
        assert set(classified_real["author_class"].unique()).issubset(valid)

    def test_has_first_timers(self, classified_real: pd.DataFrame) -> None:
        assert (classified_real["author_class"] == "first-timer").any()


class TestFeaturesReal:
    def test_time_to_merge_exists(self, featured_real: pd.DataFrame) -> None:
        assert "time_to_merge_hours" in featured_real.columns
        merged = featured_real[featured_real["pr_outcome"] == "merged"]
        assert merged["time_to_merge_hours"].notna().any()

    def test_size_buckets(self, featured_real: pd.DataFrame) -> None:
        valid = {"small", "medium", "large"}
        actual = set(featured_real["pr_size_bucket"].dropna().unique())
        assert actual.issubset(valid)

    def test_temporal_features(self, featured_real: pd.DataFrame) -> None:
        assert featured_real["hour"].between(0, 23).all()
        assert featured_real["day_of_week"].between(0, 6).all()
        assert featured_real["month"].between(1, 12).all()


class TestAggregationsReal:
    def test_monthly_has_data(self, monthly_real: pd.DataFrame) -> None:
        assert len(monthly_real) > 0
        assert monthly_real["pr_count"].sum() > 0

    def test_monthly_merge_rate_valid(self, monthly_real: pd.DataFrame) -> None:
        valid = monthly_real["merge_rate"].dropna()
        assert (valid >= 0).all()
        assert (valid <= 1).all()

    def test_weekly_has_data(self, weekly_real: pd.DataFrame) -> None:
        assert len(weekly_real) > 0


class TestAnalysisReal:
    def test_seasonal_decomposition(self, weekly_real: pd.DataFrame) -> None:
        from oss_pulse.analyze.seasonal import stl_decompose

        series = pd.Series(
            weekly_real["pr_count"].values,
            index=pd.date_range("2024-01-01", periods=len(weekly_real), freq="W"),
        )
        if len(series) >= 52:
            result = stl_decompose(series, period=26)
            assert hasattr(result, "trend")

    def test_health_index(self, monthly_real: pd.DataFrame) -> None:
        from oss_pulse.analyze.health_index import (
            compute_health_components,
            compute_health_index,
        )

        components = compute_health_components(monthly_real)
        index = compute_health_index(components)
        assert len(index) > 0
        assert index.between(0, 100).all()

    def test_survival_data(self, featured_real: pd.DataFrame) -> None:
        from oss_pulse.analyze.abandonment import build_survival_data

        surv = build_survival_data(featured_real)
        assert len(surv) > 0
        assert (surv["duration_days"] > 0).all()
        assert surv["event"].isin([0, 1]).all()

    def test_contributor_funnel(self, classified_real: pd.DataFrame) -> None:
        from oss_pulse.analyze.funnel import build_contributor_funnel

        funnel = build_contributor_funnel(classified_real)
        assert len(funnel) > 0
        assert funnel.iloc[0]["percentage"] == 100.0
