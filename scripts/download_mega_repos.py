"""Download mega-repos by splitting into yearly chunks.

GitHub returns 502 after ~200-300 consecutive pages. Mega-repos
(40k+ PRs) always hit this limit. Solution: query each year
separately, then combine.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from oss_pulse.extract.github_api_v2 import (
    PR_QUERY,
    _get_token,
    _log,
    _run_query,
)

OUTPUT_DIR = Path("data/raw/repos")
YEARS = range(2016, 2027)

def get_missing_repos() -> list[str]:
    """Get all repos not yet cached."""
    import pandas as pd
    from pathlib import Path
    repos = pd.read_parquet("data/raw/top_repos.parquet")
    cached = {
        f.stem.replace("__", "/")
        for f in Path("data/raw/repos").iterdir()
        if f.suffix == ".parquet"
    }
    return [r for r in repos["repo_name"] if r not in cached]


def extract_repo_year(
    owner: str,
    name: str,
    year: int,
    token: str,
) -> pd.DataFrame:
    """Extract PRs for a single repo for a single year."""
    since = f"{year}-01-01"
    until = f"{year + 1}-01-01"
    all_prs: list[dict[str, object]] = []
    cursor: str | None = None
    page = 0

    while True:
        variables: dict[str, object] = {"owner": owner, "name": name}
        if cursor:
            variables["cursor"] = cursor

        data = _run_query(PR_QUERY, variables, token)
        repo_data = data["data"]["repository"]

        if repo_data is None:
            break

        prs = repo_data["pullRequests"]
        rate_limit = data["data"]["rateLimit"]
        remaining = rate_limit.get("remaining", 5000)
        if remaining < 50:
            _log(f"    Rate limit low ({remaining}), sleeping 60s")
            time.sleep(60)
        time.sleep(2.0)

        for node in prs["nodes"]:
            created = node["createdAt"]
            if not created:
                continue
            if created < since:
                continue
            if created >= until:
                return pd.DataFrame(all_prs)

            author_data = node.get("author")
            author = author_data["login"] if author_data else None

            first_review = None
            first_reviewer = None
            if node["reviews"]["nodes"]:
                first_review = node["reviews"]["nodes"][0]["createdAt"]
                rd = node["reviews"]["nodes"][0].get("author")
                if rd:
                    first_reviewer = rd.get("login")

            all_prs.append(
                {
                    "repo_name": f"{owner}/{name}",
                    "pr_number": node["number"],
                    "action": "closed"
                    if node["state"] != "OPEN"
                    else "opened",
                    "state": node["state"].lower(),
                    "merged": str(node["merged"]).lower(),
                    "pr_created_at": created,
                    "pr_merged_at": node["mergedAt"],
                    "pr_closed_at": node["closedAt"],
                    "pr_updated_at": node["updatedAt"],
                    "author": author,
                    "additions": str(node["additions"] or 0),
                    "deletions": str(node["deletions"] or 0),
                    "changed_files": str(
                        node["changedFiles"] or 0
                    ),
                    "event_actor": author,
                    "event_timestamp": created,
                    "first_review_at": first_review,
                    "first_reviewer": first_reviewer,
                }
            )

        page += 1
        if not prs["pageInfo"]["hasNextPage"]:
            break
        cursor = prs["pageInfo"]["endCursor"]

        if page % 10 == 0:
            _log(f"    Page {page}, {len(all_prs)} PRs")

    return pd.DataFrame(all_prs)


def download_mega_repo(repo_name: str, token: str) -> None:
    owner, name = repo_name.split("/")
    cache_path = OUTPUT_DIR / f"{owner}__{name}.parquet"

    if cache_path.exists():
        _log(f"{repo_name}: already cached, skipping")
        return

    chunk_dir = OUTPUT_DIR / f"{owner}__{name}_chunks"
    chunk_dir.mkdir(exist_ok=True)

    all_dfs: list[pd.DataFrame] = []
    for year in YEARS:
        chunk_path = chunk_dir / f"{year}.parquet"
        if chunk_path.exists():
            df = pd.read_parquet(chunk_path)
            _log(f"  {year}: cached ({len(df)} PRs)")
            all_dfs.append(df)
            continue

        _log(f"  {year}...")
        try:
            df = extract_repo_year(owner, name, year, token)
            df.to_parquet(chunk_path, index=False)
            all_dfs.append(df)
            _log(f"  {year}: {len(df)} PRs")
        except Exception as e:
            _log(f"  {year}: FAILED ({e})")

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_parquet(cache_path, index=False)
        _log(f"{repo_name}: DONE ({len(combined)} total PRs)")


def main() -> None:
    token = _get_token()
    missing = get_missing_repos()
    _log(f"Downloading {len(missing)} missing repos by yearly chunks")

    for repo in missing:
        _log(f"\n{'='*50}")
        _log(f"Downloading {repo} by yearly chunks")
        download_mega_repo(repo, token)

    _log("\nAll repos processed!")


if __name__ == "__main__":
    main()
