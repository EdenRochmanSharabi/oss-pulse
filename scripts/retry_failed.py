"""Retry extraction for repos that failed in the first pass.

Uses github_api_v2 with max_retries=10, no size limit, 2s delay.
Skips repos already cached in data/raw/repos/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from oss_pulse.extract.github_api_v2 import _get_token, _log, extract_repo_prs

OUTPUT_DIR = Path("data/raw/repos")


def main() -> None:
    repos = pd.read_parquet("data/raw/top_repos.parquet")
    token = _get_token()

    cached = {
        f.stem.replace("__", "/")
        for f in OUTPUT_DIR.iterdir()
        if f.suffix == ".parquet"
    }

    missing = repos[~repos["repo_name"].isin(cached)]
    _log(f"Retrying {len(missing)} missing repos (max_retries=10, delay=2s)")

    for idx, row in missing.iterrows():
        repo_name: str = row["repo_name"]
        owner, name = repo_name.split("/", 1)
        cache_path = OUTPUT_DIR / f"{owner}__{name}.parquet"

        if cache_path.exists():
            continue

        _log(f"[{idx}] {repo_name}...")
        try:
            df = extract_repo_prs(owner, name, token, since="2016-01-01")
            df.to_parquet(cache_path, index=False)
            _log(f"[{idx}] {repo_name}: {len(df)} PRs")
        except Exception as e:
            _log(f"[{idx}] {repo_name}: FAILED ({e})")


if __name__ == "__main__":
    main()
