"""Changepoint detection and AI-event annotation for PR time series."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import ruptures as rpt


def detect_changepoints(
    series: pd.Series[float],
    method: str = "pelt",
    min_size: int = 12,
) -> list[int]:
    """Detect changepoints in a numeric time series.

    Returns indices of changepoints, excluding the terminal index that
    ruptures always appends (== len(series)).
    """
    values: np.ndarray = np.asarray(series.dropna().values, dtype=float)

    if method == "pelt":
        algo = rpt.Pelt(model="rbf", min_size=min_size).fit(values)
        breakpoints: list[int] = algo.predict(pen=10)
    elif method == "binseg":
        algo = rpt.Binseg(model="rbf", min_size=min_size).fit(values)
        breakpoints = algo.predict(pen=10)
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'pelt' or 'binseg'.")

    if breakpoints and breakpoints[-1] == len(values):
        breakpoints = breakpoints[:-1]

    return breakpoints


def annotate_ai_events(
    changepoints: list[int],
    series: pd.Series[float],
    events_df: pd.DataFrame,
) -> pd.DataFrame:
    """Match each changepoint to the nearest event within +/-4 weeks.

    events_df must have at least columns 'date' (parseable) and 'event'
    (description string).
    """
    event_dates = pd.to_datetime(events_df["date"])
    window = pd.Timedelta(weeks=4)
    records: list[dict[str, Any]] = []

    for cp_idx in changepoints:
        idx_val = series.index[cp_idx]
        try:
            cp_date = pd.Timestamp(str(idx_val))
        except Exception:
            continue

        mask = (event_dates >= cp_date - window) & (event_dates <= cp_date + window)
        nearby = events_df.loc[mask].copy()

        if nearby.empty:
            records.append(
                {
                    "changepoint_idx": cp_idx,
                    "changepoint_date": cp_date,
                    "nearest_event": None,
                    "nearest_event_date": None,
                    "days_apart": None,
                }
            )
            continue

        nearby_dates = pd.to_datetime(nearby["date"])
        days_diff = (nearby_dates - cp_date).abs()
        closest_pos = days_diff.idxmin()

        records.append(
            {
                "changepoint_idx": cp_idx,
                "changepoint_date": cp_date,
                "nearest_event": nearby.loc[closest_pos, "event"],
                "nearest_event_date": pd.Timestamp(
                    str(nearby.loc[closest_pos, "date"])
                ),
                "days_apart": int(days_diff.loc[closest_pos].days),
            }
        )

    return pd.DataFrame(records)


def compute_pre_post_stats(
    series: pd.Series[float],
    changepoint_idx: int,
) -> dict[str, Any]:
    """Compute summary statistics and Cohen's d for the split at changepoint_idx."""
    pre = series.iloc[:changepoint_idx].dropna()
    post = series.iloc[changepoint_idx:].dropna()

    pre_mean = float(pre.mean())
    post_mean = float(post.mean())
    pre_std = float(pre.std(ddof=1)) if len(pre) > 1 else 0.0
    post_std = float(post.std(ddof=1)) if len(post) > 1 else 0.0

    pooled_std = np.sqrt(
        ((len(pre) - 1) * pre_std**2 + (len(post) - 1) * post_std**2)
        / max(len(pre) + len(post) - 2, 1)
    )
    effect_size = (post_mean - pre_mean) / pooled_std if pooled_std > 0 else 0.0

    return {
        "pre_mean": pre_mean,
        "pre_std": pre_std,
        "pre_median": float(pre.median()),
        "pre_n": len(pre),
        "post_mean": post_mean,
        "post_std": post_std,
        "post_median": float(post.median()),
        "post_n": len(post),
        "effect_size": effect_size,
    }


if __name__ == "__main__":
    data_path = Path("data/processed/repo_weekly.parquet")
    events_path = Path("config/events.csv")

    df = pd.read_parquet(data_path)
    first_repo: str = df["repo_name"].iloc[0]
    repo_df = df[df["repo_name"] == first_repo].sort_values("year_week")
    series = repo_df.set_index("year_week")["pr_count"]

    print(f"Repo: {first_repo}")
    print(f"Series length: {len(series)}")

    cps = detect_changepoints(series)
    print(f"\nChangepoints detected: {len(cps)} at indices {cps}")

    for cp in cps:
        stats = compute_pre_post_stats(series, cp)
        print(f"\n  Changepoint at index {cp}:")
        print(f"    Pre:  mean={stats['pre_mean']:.2f}, std={stats['pre_std']:.2f}")
        print(f"    Post: mean={stats['post_mean']:.2f}, std={stats['post_std']:.2f}")
        print(f"    Cohen's d: {stats['effect_size']:.3f}")

    if events_path.exists():
        events_df = pd.read_csv(events_path)
        annotation_results = annotate_ai_events(cps, series, events_df)
        print(f"\nAnnotated changepoints:\n{annotation_results.to_string(index=False)}")
    else:
        print(f"\nEvents file not found at {events_path}, skipping annotation.")

    # Update stats.json
    primary_cp = cps[0] if cps else None
    primary_stats: dict[str, Any] = {}
    if primary_cp is not None:
        primary_stats = compute_pre_post_stats(series, primary_cp)
        primary_date = str(series.index[primary_cp])
    else:
        primary_date = None

    stats_path = Path("data/processed/stats.json")
    stats_data: dict[str, Any] = {}
    if stats_path.exists():
        with open(stats_path) as f:
            stats_data = json.load(f)

    stats_data["changepoint"] = {
        "n_changepoints": len(cps),
        "primary_changepoint_date": primary_date,
        "pre_mean": round(primary_stats.get("pre_mean", float("nan")), 3),
        "post_mean": round(primary_stats.get("post_mean", float("nan")), 3),
        "cohens_d": round(primary_stats.get("effect_size", float("nan")), 3),
    }

    with open(stats_path, "w") as f:
        json.dump(stats_data, f, indent=2)
    print(f"\nUpdated {stats_path}")
