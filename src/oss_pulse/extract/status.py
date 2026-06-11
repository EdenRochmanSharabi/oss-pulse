"""Quick status check for running extraction.

Usage: python -m oss_pulse.extract.status
"""

from __future__ import annotations

import json
from pathlib import Path


def show_status() -> None:
    status_file = Path("data/extraction_status.json")
    if not status_file.exists():
        print("No extraction running")
        return

    s = json.loads(status_file.read_text())

    bar_width = 30
    filled = int(bar_width * s["pct"] / 100)
    bar = "=" * filled + "-" * (bar_width - filled)

    done = s["completed"]
    total = s["total"]
    failed = s["failed"]
    deferred = s["deferred"]

    print(f"[{bar}] {s['pct']}%")
    print(f"  Repos:  {done}/{total} done, {failed} failed, {deferred} deferred")
    print(f"  PRs:    {s['total_prs']:,} extracted ({s['disk_mb']} MB)")
    print(f"  ETA:    {s['eta_minutes']:.0f} min")
    print(f"  Speed:  {s['avg_seconds_per_repo']:.0f}s/repo")
    print(f"  Uptime: {s['process_uptime_minutes']:.0f} min")
    print(f"  Now:    {s['current_repo']}")

    last = s.get("last_failed")
    if last:
        err = last["error"][:60]
        print(f"  Last error: {last['repo']}: {err}")


if __name__ == "__main__":
    show_status()
