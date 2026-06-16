"""Contributor retention funnel analysis."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

FUNNEL_STAGES: list[tuple[str, int]] = [
    ("1st_pr", 1),
    ("2nd_pr", 2),
    ("5th_pr", 5),
    ("10th_pr", 10),
    ("regular", 20),
]


def build_contributor_funnel(pr_df: pd.DataFrame) -> pd.DataFrame:
    """Count contributors reaching each engagement stage (non-bots only)."""
    human_df = pr_df[~pr_df["is_bot"]].copy()
    author_counts = human_df.groupby("author")["pr_number"].nunique()
    total_authors = len(author_counts)

    records: list[dict[str, object]] = []
    for stage_name, threshold in FUNNEL_STAGES:
        count = int((author_counts >= threshold).sum())
        percentage = 100.0 * count / total_authors if total_authors > 0 else 0.0
        records.append(
            {"stage": stage_name, "count": count, "percentage": round(percentage, 2)}
        )

    return pd.DataFrame(records)


def funnel_by_segment(pr_df: pd.DataFrame, segment: str) -> pd.DataFrame:
    """Build contributor funnels grouped by a segment column."""
    human_df = pr_df[~pr_df["is_bot"]].copy()

    segments: list[pd.DataFrame] = []
    for seg_value, seg_group in human_df.groupby(segment):
        author_counts = seg_group.groupby("author")["pr_number"].nunique()
        total_authors = len(author_counts)

        for stage_name, threshold in FUNNEL_STAGES:
            count = int((author_counts >= threshold).sum())
            percentage = 100.0 * count / total_authors if total_authors > 0 else 0.0
            segments.append(
                pd.DataFrame(
                    [
                        {
                            segment: seg_value,
                            "stage": stage_name,
                            "count": count,
                            "percentage": round(percentage, 2),
                        }
                    ]
                )
            )

    return pd.concat(segments, ignore_index=True) if segments else pd.DataFrame()


def compute_retention_rates(funnel_df: pd.DataFrame) -> pd.DataFrame:
    """Add stage-over-stage retention rates to a funnel DataFrame."""
    out = funnel_df.copy()
    out["retention_rate"] = None

    for i in range(1, len(out)):
        prev_count = out.iloc[i - 1]["count"]
        if prev_count > 0:
            out.loc[out.index[i], "retention_rate"] = round(
                float(out.iloc[i]["count"]) / float(prev_count), 4
            )

    return out


if __name__ == "__main__":
    data_path = Path("data/processed/pr_events_featured.parquet")
    df = pd.read_parquet(data_path)

    print(f"Loaded {len(df)} classified PR events")

    funnel = build_contributor_funnel(df)
    funnel = compute_retention_rates(funnel)

    print("\nContributor Funnel:")
    print(funnel.to_string(index=False))

    if "org_type" in df.columns:
        print("\nFunnel by org type:")
        seg_funnel = funnel_by_segment(df, "org_type")
        seg_funnel = compute_retention_rates(seg_funnel)
        print(seg_funnel.to_string(index=False))
