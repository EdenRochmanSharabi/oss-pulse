"""Data cleaning and normalization for raw PR event data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TIMESTAMP_COLS: list[str] = [
    "pr_created_at",
    "pr_merged_at",
    "pr_closed_at",
    "event_timestamp",
]

NUMERIC_COLS: list[str] = [
    "additions",
    "deletions",
    "changed_files",
]

DEDUP_KEYS: list[str] = [
    "repo_name",
    "pr_number",
    "action",
]


def clean_pr_events(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw PR events: dedup, normalize timestamps, coerce types."""
    out = df.copy()

    out = out.drop_duplicates(subset=DEDUP_KEYS, keep="last")

    for col in TIMESTAMP_COLS:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], utc=True, errors="coerce")

    out = out.dropna(subset=["pr_created_at"])

    for col in NUMERIC_COLS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)

    out = out.reset_index(drop=True)
    return out


def load_or_clean(raw_path: Path, processed_path: Path) -> pd.DataFrame:
    """Load cached clean data or clean from raw and cache as parquet."""
    if processed_path.exists():
        return pd.read_parquet(processed_path)

    df = pd.read_parquet(raw_path)
    df = clean_pr_events(df)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(processed_path, index=False)
    return df


if __name__ == "__main__":
    raw = Path("data/raw/pr_events.parquet")
    processed = Path("data/processed/pr_events_clean.parquet")
    result = load_or_clean(raw, processed)
    print(f"Clean PR events: {len(result)} rows, {len(result.columns)} columns")
