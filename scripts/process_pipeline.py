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
    top_repos_path = Path("data/raw/top_repos.parquet")
    top_repos = pd.read_parquet(top_repos_path) if top_repos_path.exists() else None
    _generate_stats(featured, PROCESSED_DIR / "stats.json", top_repos=top_repos)

    print("\n=== Done ===")


def _generate_stats(
    featured: pd.DataFrame,
    out_path: Path,
    top_repos: pd.DataFrame | None = None,
) -> None:
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

    # ── Hacktoberfest: October spike vs non-October average ───────────
    featured_with_month = featured.copy()
    featured_with_month["_year"] = featured_with_month["pr_created_at"].dt.year
    featured_with_month["_month"] = featured_with_month["pr_created_at"].dt.month

    monthly_counts = (
        featured_with_month.groupby(["_year", "_month"])
        .size()
        .reset_index(name="pr_count")
    )
    oct_counts = monthly_counts[monthly_counts["_month"] == 10]
    non_oct_counts = monthly_counts[monthly_counts["_month"] != 10]
    non_oct_avg = non_oct_counts["pr_count"].mean()

    if non_oct_avg > 0 and len(oct_counts) > 0:
        oct_counts = oct_counts.copy()
        oct_counts["spike_pct"] = (
            (oct_counts["pr_count"] - non_oct_avg) / non_oct_avg * 100
        )
        peak_row = oct_counts.loc[oct_counts["spike_pct"].idxmax()]
        peak_spike_pct = round(float(peak_row["spike_pct"]))
        peak_year = int(peak_row["_year"])
    else:
        peak_spike_pct = 0
        peak_year = 0

    oct_prs = featured_with_month[featured_with_month["_month"] == 10]
    non_oct_prs = featured_with_month[featured_with_month["_month"] != 10]
    oct_merge_rate = round(
        (oct_prs["pr_outcome"] == "merged").mean() * 100, 1
    ) if len(oct_prs) > 0 else 0.0
    non_oct_merge_rate = round(
        (non_oct_prs["pr_outcome"] == "merged").mean() * 100, 1
    ) if len(non_oct_prs) > 0 else 0.0

    stats["hacktoberfest"] = {
        "peak_spike_pct": peak_spike_pct,
        "peak_year": peak_year,
        "oct_merge_rate": oct_merge_rate,
        "non_oct_merge_rate": non_oct_merge_rate,
    }

    # ── Language comparison: merge rate per language ────────────────────
    if top_repos is not None and "language" in top_repos.columns:
        lang_map = top_repos[["repo_name", "language"]].drop_duplicates("repo_name")
        with_lang = featured.merge(lang_map, on="repo_name", how="left")
        with_lang = with_lang[with_lang["language"].notna()]

        lang_merge = (
            with_lang.groupby("language")
            .apply(
                lambda g: pd.Series({
                    "merge_rate": round(
                        (g["pr_outcome"] == "merged").mean() * 100, 1
                    ),
                    "n_prs": len(g),
                }),
                include_groups=False,
            )
            .reset_index()
        )
        lang_merge = lang_merge.sort_values("merge_rate", ascending=False)

        # Kruskal-Wallis test on per-repo merge rates across languages
        from scipy import stats as scipy_stats  # type: ignore[import-untyped]

        repo_merge = (
            with_lang.groupby(["repo_name", "language"])
            .apply(
                lambda g: (g["pr_outcome"] == "merged").mean(),
                include_groups=False,
            )
            .reset_index(name="repo_merge_rate")
        )
        groups = [
            grp["repo_merge_rate"].values
            for _, grp in repo_merge.groupby("language")
            if len(grp) >= 2
        ]
        if len(groups) >= 2:
            kw_stat, kw_p = scipy_stats.kruskal(*groups)
        else:
            kw_stat, kw_p = float("nan"), float("nan")

        languages_list = [
            {"lang": row["language"], "merge_rate": row["merge_rate"]}
            for _, row in lang_merge.iterrows()
        ]
        stats["language_comparison"] = {
            "top_language": languages_list[0]["lang"] if languages_list else "",
            "top_merge_rate": languages_list[0]["merge_rate"] if languages_list else 0,
            "kruskal_h": round(kw_stat, 1) if not pd.isna(kw_stat) else None,
            "kruskal_p": round(kw_p, 6) if not pd.isna(kw_p) else None,
            "languages": languages_list,
        }

    # ── Counterfactual: ETS forecast from pre-Copilot data ──────────────
    from statsmodels.tsa.holtwinters import ExponentialSmoothing as _ETS

    COPILOT_DATE = pd.Timestamp("2022-06-01", tz="UTC")

    unique_prs = h.drop_duplicates(subset=["repo_name", "pr_number"])
    unique_prs_ym = unique_prs.copy()
    unique_prs_ym["ym"] = unique_prs_ym["pr_created_at"].dt.to_period("M").dt.to_timestamp("s", how="S").dt.tz_localize("UTC")

    cf_monthly = unique_prs_ym.groupby("ym").agg(
        pr_count=("pr_number", "count"),
        unique_authors=("author", "nunique"),
    ).sort_index()
    cf_monthly.index = pd.DatetimeIndex(cf_monthly.index, freq="MS")

    ft_ym = unique_prs_ym[unique_prs_ym["author_class"] == "first-timer"]
    ft_monthly = ft_ym.groupby("ym").agg(ft_count=("pr_number", "count")).sort_index()
    ft_monthly.index = pd.DatetimeIndex(ft_monthly.index, freq="MS")

    rej_monthly = unique_prs_ym.groupby("ym").agg(
        total=("pr_number", "count"),
        rejected=("pr_outcome", lambda x: (x == "closed").sum()),
    ).sort_index()
    rej_monthly.index = pd.DatetimeIndex(rej_monthly.index, freq="MS")
    rej_monthly["rejection_rate"] = rej_monthly["rejected"] / rej_monthly["total"]

    ft_rej = ft_ym.groupby("ym").agg(
        total=("pr_number", "count"),
        rejected=("pr_outcome", lambda x: (x == "closed").sum()),
    ).sort_index()
    ft_rej.index = pd.DatetimeIndex(ft_rej.index, freq="MS")
    ft_rej["ft_rejection"] = ft_rej["rejected"] / ft_rej["total"]

    size_monthly = unique_prs_ym[unique_prs_ym["additions"] > 0].groupby("ym").agg(
        median_size=("additions", "median"),
    ).sort_index()
    size_monthly.index = pd.DatetimeIndex(size_monthly.index, freq="MS")

    cf_specs = [
        ("pr_volume", cf_monthly["pr_count"]),
        ("unique_contributors", cf_monthly["unique_authors"]),
        ("first_timers", ft_monthly["ft_count"]),
        ("rejection_rate", rej_monthly["rejection_rate"]),
        ("ft_rejection_rate", ft_rej["ft_rejection"]),
        ("median_pr_size", size_monthly["median_size"]),
    ]

    counterfactual = {}
    for name, series in cf_specs:
        s = series.dropna().asfreq("MS").ffill()
        pre = s[s.index < COPILOT_DATE]
        post = s[s.index >= COPILOT_DATE]
        if len(pre) < 12 or len(post) < 3:
            continue
        try:
            model = _ETS(pre, trend="add", seasonal=None, initialization_method="estimated")
            fitted = model.fit(optimized=True)
            forecast = fitted.forecast(steps=len(post))
            predicted_mean = float(forecast.mean())
            actual_mean = float(post.mean())
            if predicted_mean > 0:
                excess_pct = round((actual_mean / predicted_mean - 1) * 100)
            else:
                excess_pct = 0
            counterfactual[name] = {
                "predicted": round(predicted_mean, 1),
                "actual": round(actual_mean, 1),
                "excess_pct": excess_pct,
            }
        except Exception:
            pass

    if counterfactual:
        stats["counterfactual"] = counterfactual

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
