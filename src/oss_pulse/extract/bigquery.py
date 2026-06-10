"""BigQuery client for extracting PR data from GH Archive."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from oss_pulse.extract import queries


class BigQueryClient:
    """Thin wrapper around google-cloud-bigquery with parquet caching."""

    def __init__(self, credentials_path: Path | None = None) -> None:
        try:
            from google.cloud import bigquery  # type: ignore[import-not-found]
        except ImportError as exc:
            msg = "Install bigquery extra: pip install -e '.[bigquery]'"
            raise ImportError(msg) from exc

        kwargs: dict[str, Any] = {}
        if credentials_path is not None:
            from google.oauth2 import service_account  # type: ignore[import-not-found]

            creds = service_account.Credentials.from_service_account_file(
                str(credentials_path)
            )
            kwargs["credentials"] = creds
            kwargs["project"] = creds.project_id

        self._client: Any = bigquery.Client(**kwargs)

    def run_query(
        self,
        sql: str,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        from google.cloud import bigquery

        job_config = bigquery.QueryJobConfig()
        if params:
            job_config.query_parameters = [
                _to_query_param(k, v) for k, v in params.items()
            ]
        job = self._client.query(sql, job_config=job_config)
        return job.to_dataframe()  # type: ignore[no-any-return]

    def extract_top_repos(
        self,
        n: int = 200,
        cache_path: Path = Path("data/raw/top_repos.parquet"),
    ) -> pd.DataFrame:
        if cache_path.exists():
            return pd.read_parquet(cache_path)
        df = self.run_query(queries.TOP_REPOS_QUERY, {"top_n": n})
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        return df

    def extract_pr_events(
        self,
        repos: list[str],
        start: str = "20160101",
        end: str = "20261231",
        cache_path: Path = Path("data/raw/pr_events.parquet"),
    ) -> pd.DataFrame:
        if cache_path.exists():
            return pd.read_parquet(cache_path)
        df = self.run_query(
            queries.PR_EVENTS_QUERY,
            {"repo_list": repos, "start_date": start, "end_date": end},
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        return df

    def extract_reviews(
        self,
        repos: list[str],
        start: str = "20160101",
        end: str = "20261231",
        cache_path: Path = Path("data/raw/pr_reviews.parquet"),
    ) -> pd.DataFrame:
        if cache_path.exists():
            return pd.read_parquet(cache_path)
        df = self.run_query(
            queries.PR_REVIEW_EVENTS_QUERY,
            {"repo_list": repos, "start_date": start, "end_date": end},
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        return df


def _to_query_param(name: str, value: Any) -> Any:
    from google.cloud import bigquery

    if isinstance(value, int):
        return bigquery.ScalarQueryParameter(name, "INT64", value)
    if isinstance(value, str):
        return bigquery.ScalarQueryParameter(name, "STRING", value)
    if isinstance(value, list):
        return bigquery.ArrayQueryParameter(name, "STRING", value)
    msg = f"Unsupported param type: {type(value)}"
    raise TypeError(msg)


if __name__ == "__main__":
    cred_path = Path("data/credentials/service_account.json")
    client = BigQueryClient(cred_path if cred_path.exists() else None)
    repos_df = client.extract_top_repos()
    repo_names = repos_df["repo_name"].tolist()
    client.extract_pr_events(repo_names)
    client.extract_reviews(repo_names)
    print(f"Extracted PR events for {len(repo_names)} repos")
