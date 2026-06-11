"""Tests for repo discovery and organization-type classification."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from oss_pulse.extract.repos import (
    COMPANY_ORGS,
    FOUNDATION_ORGS,
    classify_org,
    load_repo_config,
    merge_repo_metadata,
)


class TestClassifyOrg:
    def test_company_orgs(self) -> None:
        for org in ["google", "facebook", "microsoft", "apple", "openai", "anthropic"]:
            assert classify_org(org) == "company", f"{org} should be company"

    def test_foundation_orgs(self) -> None:
        for org in ["apache", "linux", "cncf", "kubernetes", "python", "rust-lang"]:
            assert classify_org(org) == "foundation", f"{org} should be foundation"

    def test_community_orgs(self) -> None:
        for org in ["some-random-org", "unknown-project", "my-repo"]:
            assert classify_org(org) == "community", f"{org} should be community"

    def test_case_insensitive(self) -> None:
        assert classify_org("Google") == "company"
        assert classify_org("APACHE") == "foundation"
        assert classify_org("Microsoft") == "company"

    def test_all_company_orgs_classified(self) -> None:
        for org in COMPANY_ORGS:
            assert classify_org(org) == "company"

    def test_all_foundation_orgs_classified(self) -> None:
        for org in FOUNDATION_ORGS:
            assert classify_org(org) == "foundation"


class TestLoadRepoConfig:
    def test_reads_csv(self) -> None:
        config_path = Path("config/repos.csv")
        if not config_path.exists():
            pytest.skip("config/repos.csv not found")
        df = load_repo_config(config_path)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        expected_cols = {"repo", "org", "org_type", "language", "category"}
        assert expected_cols.issubset(set(df.columns))

    def test_reads_custom_path(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "test_repos.csv"
        csv_path.write_text(
            "repo,org,org_type,language,category\ntest/repo,test,community,Python,ml\n"
        )
        df = load_repo_config(csv_path)
        assert len(df) == 1
        assert df.iloc[0]["repo"] == "test/repo"


class TestMergeRepoMetadata:
    def test_merge_basic(self) -> None:
        discovered = pd.DataFrame(
            {
                "repo_name": ["google/proj", "unknown/proj"],
                "org": ["google", "unknown"],
                "org_type": ["company", "community"],
            }
        )
        manual = pd.DataFrame(
            {
                "repo": ["google/proj"],
                "org_type": ["company"],
                "language": ["Python"],
                "category": ["ml"],
            }
        )
        result = merge_repo_metadata(discovered, manual)
        assert "language" in result.columns
        assert "category" in result.columns
        assert len(result) == 2

    def test_override_org_type(self) -> None:
        discovered = pd.DataFrame(
            {
                "repo_name": ["myorg/proj"],
                "org": ["myorg"],
                "org_type": ["community"],
            }
        )
        manual = pd.DataFrame(
            {
                "repo_name": ["myorg/proj"],
                "org_type": ["foundation"],
                "language": ["Go"],
                "category": ["infra"],
            }
        )
        result = merge_repo_metadata(discovered, manual)
        assert result.iloc[0]["org_type"] == "foundation"
