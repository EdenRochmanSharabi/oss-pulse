"""CI check: verify README numbers match the processed dataset.

Extracts key numbers from the data, then checks the README contains
those numbers. Fails with a clear error if any number is stale.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd


def load_data() -> dict[str, str]:
    """Compute key numbers from processed data."""
    f = pd.read_parquet("data/processed/pr_events_featured.parquet")
    f["pr_created_at"] = pd.to_datetime(f["pr_created_at"], utc=True)
    h = f[~f["is_bot"]]

    total_prs = len(f)
    n_repos = f["repo_name"].nunique()
    n_contributors = h["author"].nunique()
    merge_rate = (f["pr_outcome"] == "merged").mean()
    merged = f[f["pr_outcome"] == "merged"]
    median_merge_h = merged["time_to_merge_hours"].median()

    ret = h.groupby("author")["pr_number"].nunique()
    funnel_2nd = (ret >= 2).mean()

    return {
        "total_prs": f"{total_prs:,}",
        "n_repos": str(n_repos),
        "n_contributors": f"{n_contributors:,}",
        "merge_rate_pct": f"{merge_rate:.1%}".replace("%", ""),
        "median_merge_hours": f"{median_merge_h:.1f}",
        "funnel_2nd_pct": f"{funnel_2nd:.0%}".replace("%", ""),
    }


def check_readme(numbers: dict[str, str]) -> list[str]:
    """Check README contains the expected numbers."""
    readme = Path("README.md").read_text()
    errors: list[str] = []

    checks = [
        ("total_prs", numbers["total_prs"], "Total PRs"),
        ("n_repos", numbers["n_repos"], "Number of repos"),
        ("n_contributors", numbers["n_contributors"], "Unique contributors"),
        ("merge_rate_pct", numbers["merge_rate_pct"], "Merge rate"),
        ("median_merge_hours", numbers["median_merge_hours"], "Median merge time"),
    ]

    for key, expected, label in checks:
        if expected not in readme:
            errors.append(
                f"  {label}: expected '{expected}' not found in README"
            )

    return errors


def main() -> None:
    data_path = Path("data/processed/pr_events_featured.parquet")
    if not data_path.exists():
        print("SKIP: no processed data found (run make process first)")
        sys.exit(0)

    print("Checking README numbers against processed data...")
    numbers = load_data()

    print("Expected values:")
    for k, v in numbers.items():
        print(f"  {k}: {v}")

    errors = check_readme(numbers)

    if errors:
        print(f"\nFAILED: {len(errors)} number(s) in README don't match data:")
        for e in errors:
            print(e)
        sys.exit(1)
    else:
        print("\nPASSED: all README numbers match the data")


if __name__ == "__main__":
    main()
