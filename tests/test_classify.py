"""Tests for bot detection, author classification, and PR outcome classification."""

from __future__ import annotations

import pandas as pd

from oss_pulse.transform.classify import classify_author, classify_pr_outcome, is_bot


class TestIsBot:
    def test_known_bots(self) -> None:
        assert is_bot("dependabot[bot]")
        assert is_bot("renovate[bot]")
        assert is_bot("greenkeeper[bot]")
        assert is_bot("github-actions[bot]")

    def test_non_bots(self) -> None:
        assert not is_bot("torvalds")
        assert not is_bot("guido")
        assert not is_bot("user-123")

    def test_bot_suffix_pattern(self) -> None:
        assert is_bot("some-custom[bot]")

    def test_empty_string(self) -> None:
        assert not is_bot("")


class TestClassifyAuthor:
    def test_bot_override(self) -> None:
        assert classify_author("dependabot[bot]", 500) == "bot"

    def test_first_timer(self) -> None:
        assert classify_author("newbie", 1) == "first-timer"

    def test_regular(self) -> None:
        assert classify_author("contributor", 5) == "regular"

    def test_maintainer(self) -> None:
        assert classify_author("core-dev", 50) == "maintainer"

    def test_zero_prs(self) -> None:
        assert classify_author("ghost", 0) == "first-timer"


class TestClassifyPrOutcome:
    def test_merged(self) -> None:
        result = classify_pr_outcome(
            "closed",
            "true",
            pd.Timestamp("2025-01-01", tz="UTC"),
            pd.Timestamp("2025-06-01", tz="UTC"),
        )
        assert result == "merged"

    def test_closed_not_merged(self) -> None:
        result = classify_pr_outcome(
            "closed",
            "false",
            pd.Timestamp("2025-01-01", tz="UTC"),
            pd.Timestamp("2025-06-01", tz="UTC"),
        )
        assert result == "closed"

    def test_abandoned(self) -> None:
        result = classify_pr_outcome(
            "open",
            "false",
            pd.Timestamp("2024-01-01", tz="UTC"),
            pd.Timestamp("2025-06-01", tz="UTC"),
        )
        assert result == "abandoned"

    def test_still_open(self) -> None:
        result = classify_pr_outcome(
            "open",
            "false",
            pd.Timestamp("2025-05-15", tz="UTC"),
            pd.Timestamp("2025-06-01", tz="UTC"),
        )
        assert result == "open"
