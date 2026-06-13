"""GitHub GraphQL PR extractor v2: robust, observable, resumable.

Improvements over v1:
- Writes a live status.json after every repo (machine-readable monitoring)
- Logs with timestamps and structured format
- ETA calculation based on running average
- Tracks failed repos for retry in a separate pass
- Summary report at the end
"""

from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

GRAPHQL_URL = "https://api.github.com/graphql"
STATUS_FILE = Path("data/extraction_status.json")

PR_QUERY = """
query($owner: String!, $name: String!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequests(
      first: 100, after: $cursor,
      orderBy: {field: CREATED_AT, direction: ASC}
    ) {
      totalCount
      pageInfo {
        hasNextPage
        endCursor
      }
      nodes {
        number
        state
        merged
        createdAt
        mergedAt
        closedAt
        updatedAt
        additions
        deletions
        changedFiles
        author {
          login
        }
        reviews(first: 1) {
          nodes {
            createdAt
            author {
              login
            }
          }
        }
      }
    }
  }
  rateLimit {
    remaining
    resetAt
  }
}
"""


def _get_token() -> str:
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        result = os.popen("gh auth token").read().strip()  # noqa: S605
        if result:
            return result
        msg = "Set GITHUB_TOKEN or install gh CLI"
        raise RuntimeError(msg)
    return token


def _log(msg: str) -> None:
    ts = datetime.now(UTC).strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _run_query(
    query: str,
    variables: dict[str, Any],
    token: str,
    max_retries: int = 10,
) -> dict[str, Any]:
    headers = {"Authorization": f"bearer {token}"}
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                GRAPHQL_URL,
                json={"query": query, "variables": variables},
                headers=headers,
                timeout=60,
            )
            if resp.status_code in (502, 503, 504, 429):
                wait = min(2 ** (attempt + 1), 120)
                _log(
                    f"  HTTP {resp.status_code}, retry "
                    f"{attempt + 1}/{max_retries} in {wait}s"
                )
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()
            if "errors" in data:
                msg = str(data["errors"])
                raise RuntimeError(msg)
            return data
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.ChunkedEncodingError,
        ):
            wait = min(2 ** (attempt + 1), 120)
            _log(f"  Connection error, retry {attempt + 1}/{max_retries} in {wait}s")
            time.sleep(wait)
    msg = f"Failed after {max_retries} retries"
    raise RuntimeError(msg)


def _write_status(status: dict[str, Any]) -> None:
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATUS_FILE.write_text(json.dumps(status, indent=2, default=str))


def extract_repo_prs(
    owner: str,
    name: str,
    token: str,
    since: str = "2016-01-01",
) -> pd.DataFrame:
    """Extract all PRs for a single repo via GraphQL pagination."""
    all_prs: list[dict[str, object]] = []
    cursor: str | None = None
    page = 0

    while True:
        variables: dict[str, Any] = {"owner": owner, "name": name}
        if cursor:
            variables["cursor"] = cursor

        data = _run_query(PR_QUERY, variables, token)
        repo_data = data["data"]["repository"]

        if repo_data is None:
            _log(f"  Repository {owner}/{name} not found")
            break

        prs = repo_data["pullRequests"]
        rate_limit = data["data"]["rateLimit"]

        remaining = rate_limit.get("remaining", 5000)
        if remaining < 50:
            reset_at = rate_limit.get("resetAt", "")
            _log(f"  Rate limit low ({remaining}), sleeping 60s (resets {reset_at})")
            time.sleep(60)

        time.sleep(2.0)

        for node in prs["nodes"]:
            created = node["createdAt"]
            if created and created < since:
                continue

            first_review = None
            first_reviewer = None
            if node["reviews"]["nodes"]:
                first_review = node["reviews"]["nodes"][0]["createdAt"]
                reviewer_data = node["reviews"]["nodes"][0].get("author")
                if reviewer_data:
                    first_reviewer = reviewer_data.get("login")

            author_data = node.get("author")
            author = author_data["login"] if author_data else None

            all_prs.append(
                {
                    "repo_name": f"{owner}/{name}",
                    "pr_number": node["number"],
                    "action": "closed" if node["state"] != "OPEN" else "opened",
                    "state": node["state"].lower(),
                    "merged": str(node["merged"]).lower(),
                    "pr_created_at": created,
                    "pr_merged_at": node["mergedAt"],
                    "pr_closed_at": node["closedAt"],
                    "pr_updated_at": node["updatedAt"],
                    "author": author,
                    "additions": str(node["additions"] or 0),
                    "deletions": str(node["deletions"] or 0),
                    "changed_files": str(node["changedFiles"] or 0),
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
            total = prs.get("totalCount", "?")
            _log(f"  Page {page}, {len(all_prs)} PRs (total in repo: {total})")

    return pd.DataFrame(all_prs)


def extract_all_repos(
    repos_df: pd.DataFrame,
    output_dir: Path = Path("data/raw/repos"),
    since: str = "2016-01-01",
    max_prs_per_repo: int = 30000,
) -> pd.DataFrame:
    """Extract PRs for all repos with live status reporting."""
    token = _get_token()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_dfs: list[pd.DataFrame] = []
    failed: list[dict[str, str]] = []
    deferred: list[str] = []
    repo_times: list[float] = []

    sorted_repos = repos_df.sort_values("event_count", ascending=True)
    total = len(sorted_repos)
    start_time = time.time()

    for idx, (_i, row) in enumerate(sorted_repos.iterrows()):
        repo_name: str = row["repo_name"]
        event_count: int = int(row.get("event_count", 0))
        owner, name = repo_name.split("/", 1)
        cache_path = output_dir / f"{owner}__{name}.parquet"

        if cache_path.exists():
            df = pd.read_parquet(cache_path)
            all_dfs.append(df)
            _log(f"[{idx + 1}/{total}] {repo_name}: cached ({len(df)} PRs)")
            continue

        if event_count > max_prs_per_repo:
            deferred.append(repo_name)
            _log(f"[{idx + 1}/{total}] {repo_name}: deferred ({event_count:,} events)")
            continue

        repo_start = time.time()
        _log(f"[{idx + 1}/{total}] {repo_name}...")

        try:
            df = extract_repo_prs(owner, name, token, since=since)
            df.to_parquet(cache_path, index=False)
            all_dfs.append(df)
            elapsed = time.time() - repo_start
            repo_times.append(elapsed)
            _log(f"[{idx + 1}/{total}] {repo_name}: {len(df)} PRs in {elapsed:.0f}s")
        except Exception as e:
            elapsed = time.time() - repo_start
            failed.append({"repo": repo_name, "error": str(e)})
            _log(f"[{idx + 1}/{total}] {repo_name}: FAILED ({e})")

        completed = len(all_dfs) + len(failed)
        remaining = total - completed - len(deferred)
        avg_time = sum(repo_times) / len(repo_times) if repo_times else 60
        eta_seconds = remaining * avg_time
        eta_min = eta_seconds / 60

        _write_status(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "completed": completed,
                "failed": len(failed),
                "deferred": len(deferred),
                "remaining": remaining,
                "total": total,
                "pct": round(100 * completed / total, 1),
                "total_prs": sum(len(d) for d in all_dfs),
                "disk_mb": round(
                    sum(f.stat().st_size for f in output_dir.iterdir()) / 1e6, 1
                ),
                "avg_seconds_per_repo": round(avg_time, 1),
                "eta_minutes": round(eta_min, 1),
                "current_repo": repo_name,
                "last_failed": failed[-1] if failed else None,
                "process_uptime_minutes": round((time.time() - start_time) / 60, 1),
            }
        )

    total_time = (time.time() - start_time) / 60
    _log(f"\n{'=' * 60}")
    _log(f"EXTRACTION COMPLETE in {total_time:.1f} minutes")
    _log(f"  Repos completed: {len(all_dfs)}")
    _log(f"  Repos failed:    {len(failed)}")
    _log(f"  Repos deferred:  {len(deferred)}")
    _log(f"  Total PRs:       {sum(len(d) for d in all_dfs):,}")

    if failed:
        _log("\nFailed repos:")
        failed_path = output_dir.parent / "failed_repos.json"
        for f in failed:
            _log(f"  {f['repo']}: {f['error']}")
        failed_path.write_text(json.dumps(failed, indent=2))
        _log(f"  Saved to {failed_path}")

    if deferred:
        _log(f"\nDeferred repos (>{max_prs_per_repo} events):")
        for d in deferred:
            _log(f"  {d}")

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_parquet(output_dir.parent / "pr_events.parquet", index=False)
        _log(f"\nCombined dataset: {len(combined):,} PRs saved")
        return combined
    return pd.DataFrame()


if __name__ == "__main__":
    repos_path = Path("data/raw/top_repos.parquet")
    if not repos_path.exists():
        _log("Run repo discovery first")
        raise SystemExit(1)

    repos = pd.read_parquet(repos_path)
    _log(f"Starting extraction for {len(repos)} repos")
    _log("Monitor progress: cat data/extraction_status.json")
    result = extract_all_repos(repos)
