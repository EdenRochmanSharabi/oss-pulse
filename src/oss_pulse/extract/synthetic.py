"""Realistic synthetic PR dataset generator for E2E testing without BigQuery."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_RNG = np.random.default_rng(42)

BOT_USERNAMES = [
    "dependabot[bot]",
    "renovate[bot]",
    "greenkeeper[bot]",
    "codecov[bot]",
    "github-actions[bot]",
]

LANGUAGES = ["Python", "JavaScript", "TypeScript", "Go", "Rust", "C++", "Java"]
ORG_TYPES = ["company", "foundation", "community"]
CATEGORIES = ["ml", "frontend", "infra", "language", "web", "tooling", "data"]


def generate_synthetic_dataset(
    n_repos: int = 20,
    years: int = 5,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Generate a realistic synthetic PR dataset.

    Returns dict with keys: repos, pr_events, pr_reviews.
    Also writes parquet files to data/raw/.
    """
    rng = np.random.default_rng(seed)
    repos_df = _generate_repos(n_repos, rng)
    pr_events_df = _generate_pr_events(repos_df, years, rng)
    pr_reviews_df = _generate_reviews(pr_events_df, rng)

    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    repos_df.to_parquet(output_dir / "top_repos.parquet", index=False)
    pr_events_df.to_parquet(output_dir / "pr_events.parquet", index=False)
    pr_reviews_df.to_parquet(output_dir / "pr_reviews.parquet", index=False)

    return {
        "repos": repos_df,
        "pr_events": pr_events_df,
        "pr_reviews": pr_reviews_df,
    }


def _generate_repos(n: int, rng: np.random.Generator) -> pd.DataFrame:
    repos = []
    for i in range(n):
        org_type = ORG_TYPES[i % len(ORG_TYPES)]
        lang = LANGUAGES[i % len(LANGUAGES)]
        cat = CATEGORIES[i % len(CATEGORIES)]
        org_name = f"org-{i:03d}"
        repo_name = f"{org_name}/project-{i:03d}"

        is_dying = i >= n - 3

        repos.append(
            {
                "repo_name": repo_name,
                "org": org_name,
                "org_type": org_type,
                "language": lang,
                "category": cat,
                "contributors": int(rng.integers(50, 2000)),
                "event_count": int(rng.integers(1000, 50000)),
                "is_dying": is_dying,
            }
        )
    return pd.DataFrame(repos)


def _generate_pr_events(
    repos_df: pd.DataFrame,
    years: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    end_date = pd.Timestamp("2026-01-01", tz="UTC")
    start_date = end_date - pd.DateOffset(years=years)
    all_events = []

    for _, repo in repos_df.iterrows():
        repo_name: str = repo["repo_name"]
        is_dying: bool = repo["is_dying"]
        base_rate = rng.uniform(2, 15)
        n_contributors = int(rng.integers(20, 200))
        contributors = [f"user-{rng.integers(0, 10000)}" for _ in range(n_contributors)]

        date_range = pd.date_range(start_date, end_date, freq="D", tz="UTC")

        for day in date_range:
            day_of_week = day.dayofweek
            month = day.month
            year_frac = (day - start_date).days / 365.25

            rate = base_rate

            # weekly seasonality: fewer PRs on weekends
            if day_of_week >= 5:
                rate *= 0.3

            # yearly seasonality: Hacktoberfest spike
            if month == 10:
                rate *= 1.8

            # summer dip
            if month in (7, 8):
                rate *= 0.75

            # growth trend
            rate *= 1 + 0.1 * year_frac

            # AI effect: step increase after mid-2022
            if day >= pd.Timestamp("2022-07-01", tz="UTC"):
                rate *= 1.25

            # dying repos: decline in last 40% of timespan
            if is_dying and year_frac > years * 0.6:
                decay = max(0.05, 1 - (year_frac - years * 0.6) / (years * 0.5))
                rate *= decay

            n_prs = rng.poisson(max(0.1, rate))
            for _ in range(n_prs):
                is_bot = rng.random() < 0.15
                if is_bot:
                    author = rng.choice(BOT_USERNAMES)
                else:
                    author = rng.choice(contributors)

                hour = int(rng.normal(14, 4)) % 24
                pr_created = day + pd.Timedelta(
                    hours=hour, minutes=int(rng.integers(0, 60))
                )

                merge_time_hours = float(rng.lognormal(3, 1.5))
                merge_time_hours = min(merge_time_hours, 24 * 180)

                fate_roll = rng.random()
                if fate_roll < 0.65:
                    state = "closed"
                    merged = True
                    merged_at = pr_created + pd.Timedelta(hours=merge_time_hours)
                    closed_at = merged_at
                elif fate_roll < 0.85:
                    state = "closed"
                    merged = False
                    closed_at = pr_created + pd.Timedelta(
                        hours=float(rng.lognormal(4, 1))
                    )
                    merged_at = pd.NaT
                else:
                    state = "open"
                    merged = False
                    merged_at = pd.NaT
                    closed_at = pd.NaT

                additions = int(rng.lognormal(3, 2))
                deletions = int(rng.lognormal(2, 2))
                changed_files = int(rng.lognormal(1, 1)) + 1

                all_events.append(
                    {
                        "repo_name": repo_name,
                        "pr_number": len(all_events) + 1,
                        "action": "closed" if state == "closed" else "opened",
                        "state": state,
                        "merged": str(merged).lower(),
                        "pr_created_at": pr_created.isoformat(),
                        "pr_merged_at": merged_at.isoformat()
                        if pd.notna(merged_at)
                        else None,
                        "pr_closed_at": closed_at.isoformat()
                        if pd.notna(closed_at)
                        else None,
                        "pr_updated_at": (
                            closed_at if pd.notna(closed_at) else pr_created
                        ).isoformat(),
                        "author": author,
                        "additions": str(additions),
                        "deletions": str(deletions),
                        "changed_files": str(changed_files),
                        "event_actor": author,
                        "event_timestamp": pr_created.isoformat(),
                    }
                )

    return pd.DataFrame(all_events)


def _generate_reviews(
    pr_events_df: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    reviews = []
    merged_prs = pr_events_df[pr_events_df["merged"] == "true"]

    for _, pr in merged_prs.iterrows():
        created = pd.Timestamp(pr["pr_created_at"])
        merged = pd.Timestamp(pr["pr_merged_at"])
        if pd.isna(merged):
            continue

        review_delay_hours = float(rng.lognormal(2, 1))
        review_time = created + pd.Timedelta(hours=review_delay_hours)
        if review_time > merged:
            review_time = created + (merged - created) * 0.5

        reviewer = f"reviewer-{rng.integers(0, 500)}"
        reviews.append(
            {
                "repo_name": pr["repo_name"],
                "pr_number": pr["pr_number"],
                "reviewer": reviewer,
                "review_timestamp": review_time.isoformat(),
            }
        )

    return pd.DataFrame(reviews)


if __name__ == "__main__":
    data = generate_synthetic_dataset()
    for name, df in data.items():
        print(f"{name}: {len(df)} rows")
