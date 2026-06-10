"""Shared fixtures using synthetic data for fast tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture(scope="session")
def synthetic_data() -> dict[str, pd.DataFrame]:
    from oss_pulse.extract.synthetic import generate_synthetic_dataset

    return generate_synthetic_dataset(n_repos=5, years=2, seed=99)


@pytest.fixture(scope="session")
def raw_pr_events(synthetic_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return synthetic_data["pr_events"]


@pytest.fixture(scope="session")
def raw_reviews(synthetic_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return synthetic_data["pr_reviews"]


@pytest.fixture(scope="session")
def cleaned_events(raw_pr_events: pd.DataFrame) -> pd.DataFrame:
    from oss_pulse.transform.clean import clean_pr_events

    return clean_pr_events(raw_pr_events)


@pytest.fixture(scope="session")
def classified_events(cleaned_events: pd.DataFrame) -> pd.DataFrame:
    from oss_pulse.transform.classify import apply_classifications

    return apply_classifications(cleaned_events)


@pytest.fixture(scope="session")
def featured_events(classified_events: pd.DataFrame) -> pd.DataFrame:
    from oss_pulse.transform.features import build_pr_features

    return build_pr_features(classified_events)


@pytest.fixture(scope="session")
def repo_monthly(featured_events: pd.DataFrame) -> pd.DataFrame:
    from oss_pulse.transform.features import build_repo_monthly

    return build_repo_monthly(featured_events)


@pytest.fixture(scope="session")
def repo_config() -> pd.DataFrame:
    config_path = Path("config/repos.csv")
    if config_path.exists():
        return pd.read_csv(config_path)
    return pd.DataFrame(columns=["repo", "org", "org_type", "language", "category"])
