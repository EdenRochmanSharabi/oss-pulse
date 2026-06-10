"""Tests for data cleaning and normalization."""

from __future__ import annotations

import pandas as pd

from oss_pulse.transform.clean import (
    DEDUP_KEYS,
    NUMERIC_COLS,
    TIMESTAMP_COLS,
    clean_pr_events,
    load_or_clean,
)


class TestCleanPrEvents:
    def test_dedup_removes_duplicates(self, raw_pr_events: pd.DataFrame) -> None:
        dup = pd.concat([raw_pr_events, raw_pr_events.iloc[:5]], ignore_index=True)
        cleaned = clean_pr_events(dup)
        assert len(cleaned) <= len(raw_pr_events)
        assert cleaned.duplicated(subset=DEDUP_KEYS).sum() == 0

    def test_timestamps_are_utc_datetime(self, cleaned_events: pd.DataFrame) -> None:
        for col in TIMESTAMP_COLS:
            if col in cleaned_events.columns:
                assert pd.api.types.is_datetime64_any_dtype(cleaned_events[col])
                non_null = cleaned_events[col].dropna()
                if len(non_null) > 0:
                    assert non_null.dt.tz is not None

    def test_numeric_columns_are_int(self, cleaned_events: pd.DataFrame) -> None:
        for col in NUMERIC_COLS:
            if col in cleaned_events.columns:
                assert cleaned_events[col].dtype in (
                    "int64",
                    "int32",
                ), f"{col} is {cleaned_events[col].dtype}"

    def test_null_pr_created_at_dropped(self) -> None:
        df = pd.DataFrame(
            {
                "repo_name": ["a/b", "c/d"],
                "pr_number": [1, 2],
                "action": ["opened", "opened"],
                "pr_created_at": ["2025-01-01T00:00:00Z", None],
                "additions": ["10", "20"],
                "deletions": ["5", "10"],
                "changed_files": ["2", "3"],
            }
        )
        result = clean_pr_events(df)
        assert len(result) == 1

    def test_index_is_reset(self, raw_pr_events: pd.DataFrame) -> None:
        cleaned = clean_pr_events(raw_pr_events)
        assert list(cleaned.index) == list(range(len(cleaned)))


class TestLoadOrClean:
    def test_cleans_from_raw_and_caches(
        self, raw_pr_events: pd.DataFrame, tmp_path: object
    ) -> None:
        from pathlib import Path

        tmp = Path(str(tmp_path))
        raw_path = tmp / "raw.parquet"
        processed_path = tmp / "processed" / "clean.parquet"

        raw_pr_events.to_parquet(raw_path, index=False)

        result = load_or_clean(raw_path, processed_path)
        assert len(result) > 0
        assert processed_path.exists()

    def test_loads_cached_file(
        self, cleaned_events: pd.DataFrame, tmp_path: object
    ) -> None:
        from pathlib import Path

        tmp = Path(str(tmp_path))
        raw_path = tmp / "raw.parquet"
        processed_path = tmp / "clean.parquet"

        cleaned_events.to_parquet(processed_path, index=False)

        # raw_path does not need to exist since processed_path is present
        result = load_or_clean(raw_path, processed_path)
        assert len(result) == len(cleaned_events)
