"""Retry extraction until ALL repos are downloaded.

Loops forever, retrying failed repos each round with increasing
delay between rounds. Skips already-cached repos.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from oss_pulse.extract.github_api_v2 import _get_token, _log, extract_repo_prs

OUTPUT_DIR = Path("data/raw/repos")


def get_missing(repos: pd.DataFrame) -> list[str]:
    cached = {
        f.stem.replace("__", "/")
        for f in OUTPUT_DIR.iterdir()
        if f.suffix == ".parquet"
    }
    return [r for r in repos["repo_name"] if r not in cached]


def main() -> None:
    repos = pd.read_parquet("data/raw/top_repos.parquet")
    token = _get_token()
    round_num = 0

    while True:
        missing = get_missing(repos)
        if not missing:
            _log("All repos downloaded!")
            break

        round_num += 1
        _log(f"=== Round {round_num}: {len(missing)} repos remaining ===")

        succeeded = 0
        for repo_name in missing:
            owner, name = repo_name.split("/", 1)
            cache_path = OUTPUT_DIR / f"{owner}__{name}.parquet"

            if cache_path.exists():
                continue

            _log(f"  {repo_name}...")
            try:
                df = extract_repo_prs(owner, name, token, since="2016-01-01")
                df.to_parquet(cache_path, index=False)
                _log(f"  {repo_name}: {len(df)} PRs")
                succeeded += 1
            except Exception as e:
                _log(f"  {repo_name}: FAILED ({e})")

        remaining = len(get_missing(repos))
        _log(
            f"=== Round {round_num} done: "
            f"{succeeded} succeeded, {remaining} remaining ==="
        )

        if remaining == 0:
            _log("All repos downloaded!")
            break

        wait = min(300, 60 * round_num)
        _log(f"Waiting {wait}s before next round...")
        time.sleep(wait)


if __name__ == "__main__":
    main()
