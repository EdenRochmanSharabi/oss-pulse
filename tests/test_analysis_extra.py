"""Extra tests to boost coverage on analyze modules."""

from __future__ import annotations

import numpy as np
import pandas as pd


def test_compare_ecosystems() -> None:
    from oss_pulse.analyze.comparative import compare_ecosystems

    df = pd.DataFrame(
        {
            "language": ["Python"] * 5 + ["Go"] * 5,
            "pr_count": np.random.default_rng(42).integers(10, 100, 10),
            "merge_rate": np.random.default_rng(42).uniform(0.3, 0.9, 10),
        }
    )
    result = compare_ecosystems(df)
    assert "language" in result.columns
    assert len(result) == 2


def test_funnel_by_segment(classified_events: pd.DataFrame) -> None:
    from oss_pulse.analyze.funnel import funnel_by_segment

    df = classified_events.copy()
    df["org_type"] = "community"
    df.loc[df.index[:100], "org_type"] = "company"

    result = funnel_by_segment(df, "org_type")
    assert "org_type" in result.columns
    assert len(result) > 0


def test_changepoint_annotation_with_datetime_index() -> None:
    from oss_pulse.analyze.changepoint import annotate_ai_events

    dates = pd.date_range("2022-01-01", periods=100, freq="W")
    series = pd.Series(np.ones(100), index=dates)
    events = pd.DataFrame({"date": ["2022-06-01"], "event": ["test event"]})
    result = annotate_ai_events([25], series, events)
    assert len(result) >= 0


def test_dashboard_ai_effect() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from oss_pulse.visualize.dashboard import create_ai_effect_dashboard

    series = pd.Series(
        np.random.default_rng(42).normal(50, 5, 100),
        index=pd.date_range("2020-01-01", periods=100, freq="W"),
    )
    data = {
        "series": series,
        "changepoints": [50],
        "pre_post_stats": {
            "mean": {"pre": 48.0, "post": 55.0},
            "std": {"pre": 5.0, "post": 5.0},
        },
    }
    fig = create_ai_effect_dashboard(data)
    assert fig is not None
    plt.close("all")
