"""Author and PR outcome classification."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

BOT_PATTERNS: list[str] = [
    "dependabot",
    "renovate",
    "greenkeeper",
    "codecov",
    "github-actions",
    "snyk",
    "imgbot",
    "allcontributors",
    "stale",
    "mergify",
    "pre-commit-ci",
    "depfu",
    "whitesource-bolt",
    "mend-bolt",
    "netlify",
    "vercel",
    "sonarcloud",
    "deepsource",
    "lgtm-com",
    "pyup",
]


def is_bot(username: str) -> bool:
    """Check whether a username belongs to a known bot."""
    lower = username.lower()
    if "[bot]" in lower:
        return True
    return any(pattern in lower for pattern in BOT_PATTERNS)


def classify_author(username: str, pr_count: int) -> str:
    """Classify an author as bot, first-timer, regular, or maintainer."""
    if is_bot(username):
        return "bot"
    if pr_count <= 1:
        return "first-timer"
    if pr_count <= 10:
        return "regular"
    return "maintainer"


def classify_pr_outcome(
    state: str,
    merged: str,
    last_activity: pd.Timestamp,
    reference_date: pd.Timestamp,
) -> str:
    """Classify a PR as merged, closed, abandoned, or open."""
    if merged == "true":
        return "merged"
    if state == "closed":
        return "closed"
    if state == "open" and (reference_date - last_activity).days > 90:
        return "abandoned"
    return "open"


def apply_classifications(df: pd.DataFrame) -> pd.DataFrame:
    """Apply author and outcome classifications to a clean PR events DataFrame."""
    out = df.copy()

    author_pr_counts = out.groupby("author")["pr_number"].nunique()

    out["is_bot"] = out["author"].apply(is_bot)
    out["author_class"] = out.apply(
        lambda row: classify_author(
            row["author"], author_pr_counts.get(row["author"], 0)
        ),
        axis=1,
    )

    reference_date = out["event_timestamp"].max()
    last_activity = out.groupby(["repo_name", "pr_number"])[
        "event_timestamp"
    ].transform("max")

    out["pr_outcome"] = out.apply(
        lambda row: classify_pr_outcome(
            row["state"],
            row["merged"],
            last_activity[row.name],
            reference_date,
        ),
        axis=1,
    )

    return out


if __name__ == "__main__":
    clean_path = Path("data/processed/pr_events_clean.parquet")
    output_path = Path("data/processed/pr_events_classified.parquet")

    df = pd.read_parquet(clean_path)
    df = apply_classifications(df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Classified PR events: {len(df)} rows")
    print(f"Outcome distribution:\n{df['pr_outcome'].value_counts()}")
    print(f"Author class distribution:\n{df['author_class'].value_counts()}")
