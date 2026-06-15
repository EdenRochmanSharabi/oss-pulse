"""Download mega-repos using GitHub Search API with date filtering.

The Search API allows created:YYYY-MM-DD..YYYY-MM-DD, avoiding the
need to paginate through all historical PRs. Splits by month when
a year has >1000 PRs.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, UTC
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from oss_pulse.extract.github_api_v2 import _log

OUTPUT_DIR = Path("data/raw/repos")
REST_URL = "https://api.github.com/search/issues"


def _get_token() -> str:
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        token = os.popen("gh auth token").read().strip()
    return token


def search_prs(
    repo: str,
    date_from: str,
    date_to: str,
    token: str,
) -> list[dict[str, object]]:
    """Search PRs in a repo within a date range. Paginates up to 1000."""
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
    q = f"repo:{repo} is:pr created:{date_from}..{date_to}"
    all_items: list[dict[str, object]] = []
    page = 1

    while True:
        for attempt in range(5):
            try:
                resp = requests.get(
                    REST_URL,
                    params={"q": q, "per_page": 100, "page": page, "sort": "created", "order": "asc"},
                    headers=headers,
                    timeout=30,
                )
                if resp.status_code == 403:
                    reset = int(resp.headers.get("x-ratelimit-reset", 0))
                    wait = max(reset - int(time.time()), 10)
                    _log(f"    Rate limited, waiting {wait}s")
                    time.sleep(wait)
                    continue
                if resp.status_code in (502, 503):
                    time.sleep(2 ** (attempt + 1))
                    continue
                resp.raise_for_status()
                break
            except requests.exceptions.ConnectionError:
                time.sleep(2 ** (attempt + 1))
        else:
            _log(f"    Failed after 5 retries for page {page}")
            break

        data = resp.json()
        items = data.get("items", [])
        if not items:
            break

        for item in items:
            pr = item.get("pull_request", {})
            all_items.append({
                "repo_name": repo,
                "pr_number": item["number"],
                "action": "closed" if item["state"] == "closed" else "opened",
                "state": "merged" if pr.get("merged_at") else item["state"],
                "merged": "true" if pr.get("merged_at") else "false",
                "pr_created_at": item["created_at"],
                "pr_merged_at": pr.get("merged_at"),
                "pr_closed_at": item["closed_at"],
                "pr_updated_at": item["updated_at"],
                "author": item["user"]["login"] if item.get("user") else None,
                "additions": "0",
                "deletions": "0",
                "changed_files": "0",
                "event_actor": item["user"]["login"] if item.get("user") else None,
                "event_timestamp": item["created_at"],
                "first_review_at": None,
                "first_reviewer": None,
            })

        if len(items) < 100:
            break
        page += 1
        if page > 10:
            break
        time.sleep(2)

    return all_items


def download_repo(repo: str, token: str) -> None:
    owner, name = repo.split("/")
    cache_path = OUTPUT_DIR / f"{owner}__{name}.parquet"
    if cache_path.exists():
        _log(f"{repo}: already cached")
        return

    chunk_dir = OUTPUT_DIR / f"{owner}__{name}_chunks"
    chunk_dir.mkdir(exist_ok=True)

    all_dfs: list[pd.DataFrame] = []
    for year in range(2016, 2027):
        chunk_path = chunk_dir / f"{year}.parquet"
        if chunk_path.exists():
            df = pd.read_parquet(chunk_path)
            _log(f"  {year}: cached ({len(df)} PRs)")
            all_dfs.append(df)
            continue

        count_resp = requests.get(
            REST_URL,
            params={"q": f"repo:{repo} is:pr created:{year}-01-01..{year}-12-31", "per_page": 1},
            headers={"Authorization": f"token {token}"},
            timeout=10,
        )
        total = count_resp.json().get("total_count", 0)
        time.sleep(2)

        if total == 0:
            _log(f"  {year}: 0 PRs")
            all_dfs.append(pd.DataFrame())
            pd.DataFrame().to_parquet(chunk_path, index=False)
            continue

        if total <= 1000:
            _log(f"  {year}: {total} PRs (single query)")
            items = search_prs(repo, f"{year}-01-01", f"{year}-12-31", token)
            df = pd.DataFrame(items)
            df.to_parquet(chunk_path, index=False)
            all_dfs.append(df)
        else:
            _log(f"  {year}: {total} PRs (splitting by month)")
            year_items: list[dict[str, object]] = []
            for month in range(1, 13):
                if month == 12:
                    d_from = f"{year}-12-01"
                    d_to = f"{year}-12-31"
                else:
                    d_from = f"{year}-{month:02d}-01"
                    d_to = f"{year}-{month + 1:02d}-01"
                items = search_prs(repo, d_from, d_to, token)
                if items:
                    year_items.extend(items)
                    _log(f"    {year}-{month:02d}: {len(items)} PRs")
                time.sleep(2)
            df = pd.DataFrame(year_items)
            df.to_parquet(chunk_path, index=False)
            all_dfs.append(df)

        _log(f"  {year}: {len(all_dfs[-1])} PRs saved")

    valid = [d for d in all_dfs if len(d) > 0]
    if valid:
        combined = pd.concat(valid, ignore_index=True)
        combined = combined.drop_duplicates(subset=["repo_name", "pr_number"], keep="last")
        combined.to_parquet(cache_path, index=False)
        _log(f"{repo}: DONE ({len(combined)} total PRs)")
    else:
        _log(f"{repo}: no PRs found")


def main() -> None:
    token = _get_token()
    repos = pd.read_parquet("data/raw/top_repos.parquet")
    cached = {
        f.stem.replace("__", "/")
        for f in OUTPUT_DIR.iterdir()
        if f.suffix == ".parquet"
    }
    missing = [r for r in repos["repo_name"] if r not in cached]
    _log(f"Downloading {len(missing)} missing repos via Search API")

    for repo in missing:
        _log(f"\n{'=' * 50}")
        _log(f"{repo}")
        download_repo(repo, token)

    _log("\nAll repos processed!")


if __name__ == "__main__":
    main()
