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

CLASSIFICATION_PATH = PROCESSED_DIR / "repo_classification.json"


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
    if CLASSIFICATION_PATH.exists():
        import json as _json
        with open(CLASSIFICATION_PATH) as f:
            classification = _json.load(f)
        exclude = set(classification.get("non_code", []))
        print(f"  Loaded {len(exclude)} non-code repos from repo_classification.json")
    else:
        repos = pd.read_parquet("data/raw/top_repos.parquet")
        non_code = repos[
            repos["language"].isin(NON_CODE_LANGUAGES)
            | repos["language"].isna()
        ]["repo_name"].tolist()
        exclude = set(non_code)
        print(f"  No classification file; falling back to language filter ({len(exclude)} repos)")
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

    print("\n=== 8. Generating stats.json ===")
    _generate_stats(featured, PROCESSED_DIR / "stats.json")

    print("\n=== Done ===")


def _generate_stats(featured: pd.DataFrame, out_path: Path) -> None:
    """Generate stats.json: the single source of truth for all numbers."""
    import json

    h = featured[~featured["is_bot"]]
    merged = featured[featured["pr_outcome"] == "merged"]
    ret = h.groupby("author")["pr_number"].nunique()

    def _era(start: str, end: str) -> dict[str, object]:
        e = h[(h["pr_created_at"] >= start) & (h["pr_created_at"] < end)]
        if len(e) == 0:
            return {}
        n_m = e["pr_created_at"].dt.to_period("M").nunique()
        ft = e[e["author_class"] == "first-timer"]
        return {
            "contributors_per_month": round(
                e.groupby(e["pr_created_at"].dt.to_period("M"))["author"]
                .nunique()
                .mean()
            ),
            "firsttimers_per_month": round(len(ft) / n_m),
            "ft_rejection_rate": round(
                ft["pr_outcome"].isin(["closed", "abandoned"]).mean() * 100, 1
            ),
            "merge_rate": round(
                (e["pr_outcome"] == "merged").mean() * 100, 1
            ),
            "prs_per_month": round(len(e) / n_m),
        }

    def _tool_era(start: str, end: str) -> dict[str, object]:
        e = h[(h["pr_created_at"] >= start) & (h["pr_created_at"] < end)]
        if len(e) == 0:
            return {}
        ft = e[e["author_class"] == "first-timer"]
        n_m = e["pr_created_at"].dt.to_period("M").nunique()
        return {
            "merge_rate": round(
                (e["pr_outcome"] == "merged").mean() * 100, 1
            ),
            "ft_merge_rate": round(
                (ft["pr_outcome"] == "merged").mean() * 100, 1
            ),
            "prs_per_month": round(len(e) / n_m),
        }

    stats = {
        "dataset": {
            "total_prs": len(featured),
            "n_repos": featured["repo_name"].nunique(),
            "n_contributors": h["author"].nunique(),
            "date_from": str(featured["pr_created_at"].min().date()),
            "date_to": str(featured["pr_created_at"].max().date()),
        },
        "outcomes": {
            "merge_rate": round(
                (featured["pr_outcome"] == "merged").mean() * 100, 1
            ),
            "closed_rate": round(
                (featured["pr_outcome"] == "closed").mean() * 100, 1
            ),
            "abandoned_rate": round(
                (featured["pr_outcome"] == "abandoned").mean() * 100, 1
            ),
            "open_rate": round(
                (featured["pr_outcome"] == "open").mean() * 100, 1
            ),
        },
        "merge_time": {
            "median_hours": round(
                merged["time_to_merge_hours"].median(), 1
            ),
        },
        "funnel": {
            "total": len(ret),
            "2nd_pr": int((ret >= 2).sum()),
            "2nd_pr_pct": round((ret >= 2).mean() * 100, 1),
            "5th_pr": int((ret >= 5).sum()),
            "5th_pr_pct": round((ret >= 5).mean() * 100, 1),
            "regular": int((ret >= 20).sum()),
            "regular_pct": round((ret >= 20).mean() * 100, 1),
            "dropout_1st_pct": round(
                (1 - (ret >= 2).mean()) * 100
            ),
        },
        "ai_eras": {
            "2016_2019": _era("2016-01-01", "2020-01-01"),
            "2020_2021": _era("2020-01-01", "2022-01-01"),
            "2022_copilot": _era("2022-01-01", "2023-01-01"),
            "2023_chatgpt": _era("2023-01-01", "2024-01-01"),
            "2024_cursor": _era("2024-01-01", "2025-01-01"),
            "2025_agentic": _era("2025-01-01", "2026-06-01"),
        },
        "tool_eras": {
            "pre_copilot": _tool_era("2016-01-01", "2022-06-01"),
            "copilot": _tool_era("2022-06-01", "2023-03-01"),
            "chatgpt_gpt4": _tool_era("2023-03-01", "2024-03-01"),
            "cursor": _tool_era("2024-03-01", "2025-02-01"),
            "agentic": _tool_era("2025-02-01", "2026-06-01"),
        },
    }

    with open(out_path, "w") as fp:
        json.dump(stats, fp, indent=2)

    print(f"  Saved {out_path}")
    print(f"  PRs: {stats['dataset']['total_prs']:,}")
    print(f"  Repos: {stats['dataset']['n_repos']}")
    print(f"  Contributors: {stats['dataset']['n_contributors']:,}")
    print(f"  Merge rate: {stats['outcomes']['merge_rate']}%")
    print(f"  Median merge: {stats['merge_time']['median_hours']}h")


if __name__ == "__main__":
    main()
