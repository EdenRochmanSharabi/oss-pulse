"""Extract PR data via GitHub GraphQL API as a BigQuery-free alternative."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

GRAPHQL_URL = "https://api.github.com/graphql"

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


def _run_query(
    query: str,
    variables: dict[str, Any],
    token: str,
    max_retries: int = 5,
) -> dict[str, Any]:
    headers = {"Authorization": f"bearer {token}"}
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                GRAPHQL_URL,
                json={"query": query, "variables": variables},
                headers=headers,
                timeout=30,
            )
            if resp.status_code in (502, 503, 429):
                wait = 2 ** (attempt + 1)
                print(f"  {resp.status_code}, retrying in {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()
            if "errors" in data:
                msg = str(data["errors"])
                raise RuntimeError(msg)
            return data
        except requests.exceptions.ConnectionError:
            wait = 2 ** (attempt + 1)
            print(f"  Connection error, retrying in {wait}s...")
            time.sleep(wait)
    msg = f"Failed after {max_retries} retries"
    raise RuntimeError(msg)


def _handle_rate_limit(rate_limit: dict[str, Any]) -> None:
    remaining = rate_limit.get("remaining", 1)
    if remaining < 50:
        reset_at = rate_limit.get("resetAt", "")
        print(f"  Rate limit low ({remaining}), waiting for reset at {reset_at}...")
        time.sleep(60)


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
            print(f"  Repository {owner}/{name} not found or inaccessible")
            break

        prs = repo_data["pullRequests"]
        rate_limit = data["data"]["rateLimit"]
        _handle_rate_limit(rate_limit)
        time.sleep(1.0)

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
            print(f"  Page {page}, {len(all_prs)} PRs so far...")

    return pd.DataFrame(all_prs)


def extract_all_repos(
    repos_df: pd.DataFrame,
    output_dir: Path = Path("data/raw/repos"),
    since: str = "2016-01-01",
    max_prs_per_repo: int = 30000,
) -> pd.DataFrame:
    """Extract PRs for all repos, saving progress per repo.

    Repos are sorted by event_count (smallest first) to maximize
    the number of completed repos early. Repos exceeding max_prs_per_repo
    are deferred to a second pass.
    """
    token = _get_token()
    output_dir.mkdir(parents=True, exist_ok=True)
    all_dfs: list[pd.DataFrame] = []
    deferred: list[str] = []

    sorted_repos = repos_df.sort_values("event_count", ascending=True)

    for _i, row in sorted_repos.iterrows():
        repo_name: str = row["repo_name"]
        event_count: int = int(row["event_count"])
        owner, name = repo_name.split("/", 1)
        cache_path = output_dir / f"{owner}__{name}.parquet"

        if cache_path.exists():
            print(f"[{len(all_dfs) + 1}/{len(repos_df)}] {repo_name}: cached")
            df = pd.read_parquet(cache_path)
            all_dfs.append(df)
            continue

        if event_count > max_prs_per_repo:
            deferred.append(repo_name)
            print(f"[deferred] {repo_name} ({event_count:,} events)")
            continue

        print(
            f"[{len(all_dfs) + 1}/{len(repos_df)}] {repo_name}...", end=" ", flush=True
        )
        try:
            df = extract_repo_prs(owner, name, token, since=since)
            df.to_parquet(cache_path, index=False)
            all_dfs.append(df)
            print(f"{len(df)} PRs")
        except Exception as e:
            print(f"ERROR: {e}")

    if deferred:
        print(f"\n--- Deferred {len(deferred)} large repos ---")
        for repo_name in deferred:
            print(f"  {repo_name}")

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_parquet(output_dir.parent / "pr_events.parquet", index=False)
        return combined
    return pd.DataFrame()


if __name__ == "__main__":
    repos_path = Path("data/raw/top_repos.parquet")
    if not repos_path.exists():
        print("Run repo discovery first")
        raise SystemExit(1)

    repos = pd.read_parquet(repos_path)
    print(f"Extracting PRs for {len(repos)} repos via GitHub API...")
    result = extract_all_repos(repos)
    n_repos = result["repo_name"].nunique()
    print(f"\nDone: {len(result):,} total PRs across {n_repos} repos")
