"""Full data processing pipeline: combine, clean, classify, filter, feature engineer.

This is the single entry point for processing raw repo parquets into
the analysis-ready dataset. Handles:
1. Combine per-repo parquets
2. Clean (dedup, timestamps, types)
3. Classify (bots, author type, PR outcome)
4. Feature engineering
5. Filter incomplete months
6. Filter non-code and curated repos
7. Save processed parquets
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

REPO_DIR = Path("data/raw/repos")
PROCESSED_DIR = Path("data/processed")

NON_CODE_LANGUAGES = {None, "Markdown"}
CURATED_REPOS = [
    "papers-we-love/papers-we-love",
    "jaywcjlove/awesome-mac",
    "Snailclimb/JavaGuide",
    "yangshun/tech-interview-handbook",
    "EbookFoundation/free-programming-books",
    "ripienaar/free-for-dev",
    "GrowingGit/GitHub-Chinese-Top-Charts",
    "mattpocock/skills",
    "bregman-arie/devops-exercises",
    "avelino/awesome-go",
    "awesome-selfhosted/awesome-selfhosted",
    "torvalds/linux",
]


def main() -> None:
    print("=== 1. Combining raw parquets ===")
    dfs = []
    for f in REPO_DIR.glob("*.parquet"):
        df = pd.read_parquet(f)
        if len(df) > 0:
            dfs.append(df)
    raw = pd.concat(dfs, ignore_index=True)
    print(f"  {len(raw):,} PRs, {raw['repo_name'].nunique()} repos")

    print("\n=== 2. Cleaning ===")
    from oss_pulse.transform.clean import clean_pr_events

    cleaned = clean_pr_events(raw)
    print(f"  {len(cleaned):,} PRs")

    print("\n=== 3. Classifying ===")
    from oss_pulse.transform.classify import apply_classifications

    classified = apply_classifications(cleaned)
    print(f"  Outcomes: {classified['pr_outcome'].value_counts().to_dict()}")

    print("\n=== 4. Feature engineering ===")
    from oss_pulse.transform.features import (
        build_pr_features,
        build_repo_monthly,
        build_repo_weekly,
    )

    featured = build_pr_features(classified)

    print("\n=== 5. Filter incomplete month ===")
    featured["pr_created_at"] = pd.to_datetime(
        featured["pr_created_at"], utc=True
    )
    max_date = featured["pr_created_at"].max()
    cutoff = max_date.replace(day=1)
    featured = featured[featured["pr_created_at"] < cutoff]
    print(f"  Filtered to before {cutoff.date()}: {len(featured):,} PRs")

    print("\n=== 6. Filter non-code repos ===")
    repos = pd.read_parquet("data/raw/top_repos.parquet")
    non_code = repos[
        repos["language"].isin(NON_CODE_LANGUAGES)
        | repos["language"].isna()
    ]["repo_name"].tolist()
    exclude = set(non_code + CURATED_REPOS)
    before = featured["repo_name"].nunique()
    featured = featured[~featured["repo_name"].isin(exclude)]
    after = featured["repo_name"].nunique()
    print(f"  Excluded {before - after} non-code/curated repos")

    print("\n=== 7. Saving ===")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    featured.to_parquet(
        PROCESSED_DIR / "pr_events_featured.parquet", index=False
    )

    monthly = build_repo_monthly(featured)
    monthly.to_parquet(PROCESSED_DIR / "repo_monthly.parquet", index=False)

    weekly = build_repo_weekly(featured)
    weekly.to_parquet(PROCESSED_DIR / "repo_weekly.parquet", index=False)

    print(f"\n=== Done ===")
    h = featured[~featured["is_bot"]]
    print(f"  PRs: {len(featured):,}")
    print(f"  Repos: {featured['repo_name'].nunique()}")
    print(f"  Contributors: {h['author'].nunique():,}")
    print(f"  Merge rate: {(featured['pr_outcome']=='merged').mean():.1%}")
    m = featured[featured["pr_outcome"] == "merged"]
    print(f"  Median merge time: {m['time_to_merge_hours'].median():.1f}h")


if __name__ == "__main__":
    main()
