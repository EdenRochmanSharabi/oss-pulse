"""Composite dashboard figures combining multiple plot types."""

from __future__ import annotations

from typing import Any

import matplotlib.figure as mfigure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from oss_pulse.visualize.style import PALETTE, setup_style


def create_overview_dashboard(data: dict[str, Any]) -> mfigure.Figure:
    """Create a 2x2 overview dashboard.

    Expected data keys:
        weekly_trend: pd.DataFrame with 'date' and 'pr_count' columns
        health_ranking: pd.DataFrame with 'repo_name' and 'health_index' columns
        merge_times: pd.Series of merge durations
        activity_df: pd.DataFrame with 'hour' and 'day_of_week' columns
    """
    setup_style()

    fig, axes = plt.subplots(2, 2, figsize=(18, 11))

    weekly: pd.DataFrame = data["weekly_trend"]
    ax_trend = axes[0, 0]
    ax_trend.plot(
        weekly["date"], weekly["pr_count"], color=PALETTE["primary"], linewidth=1.5
    )
    ax_trend.fill_between(
        weekly["date"], weekly["pr_count"], alpha=0.1, color=PALETTE["primary"]
    )
    ax_trend.set_title("PR Volume Trend", fontsize=12, fontweight="bold")
    ax_trend.set_xlabel("Date")
    ax_trend.set_ylabel("PR Count")

    activity: pd.DataFrame = data["activity_df"]
    ax_heat = axes[0, 1]
    pivot = activity.pivot_table(
        index="day_of_week",
        columns="hour",
        aggfunc="size",
        fill_value=0,
    )
    pivot = pivot.reindex(index=range(7), columns=range(24), fill_value=0)
    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    sns.heatmap(
        pivot,
        ax=ax_heat,
        cmap="Blues",
        cbar=True,
        linewidths=0.3,
        linecolor=PALETTE["bg"],
    )
    ax_heat.set_yticklabels(day_labels, rotation=0)
    ax_heat.set_title("Activity Heatmap", fontsize=12, fontweight="bold")
    ax_heat.set_xlabel("Hour")
    ax_heat.set_ylabel("Day")

    health: pd.DataFrame = data["health_ranking"]
    top10 = health.nlargest(10, "health_index").sort_values("health_index")
    ax_bar = axes[1, 0]
    ax_bar.barh(
        top10["repo_name"],
        top10["health_index"],
        color=PALETTE["success"],
        edgecolor=PALETTE["bg"],
    )
    ax_bar.set_title("Top 10 Health Index", fontsize=12, fontweight="bold")
    ax_bar.set_xlabel("Health Index")

    merge_times: pd.Series = data["merge_times"]  # type: ignore[assignment]
    ax_hist = axes[1, 1]
    ax_hist.hist(
        merge_times,
        bins=30,
        color=PALETTE["warning"],
        edgecolor=PALETTE["bg"],
        alpha=0.85,
    )
    ax_hist.axvline(
        merge_times.median(),
        color=PALETTE["accent"],
        linestyle="--",
        linewidth=1.2,
        label=f"Median: {merge_times.median():.1f}",
    )
    ax_hist.set_title("Merge Time Distribution", fontsize=12, fontweight="bold")
    ax_hist.set_xlabel("Hours to Merge")
    ax_hist.set_ylabel("Frequency")
    ax_hist.legend(loc="upper right")

    fig.suptitle("OSS Pulse Overview Dashboard", fontsize=16, fontweight="bold", y=1.01)
    return fig


def create_ai_effect_dashboard(data: dict[str, Any]) -> mfigure.Figure:
    """Create a 1x2 AI-effect analysis dashboard.

    Expected data keys:
        series: pd.Series with DatetimeIndex
        changepoints: list of datetime-like changepoint locations
        pre_post_stats: dict mapping metric names to {"pre": float, "post": float}
    """
    setup_style()

    fig, (ax_ts, ax_bar) = plt.subplots(1, 2, figsize=(18, 7))

    series: pd.Series = data["series"]  # type: ignore[assignment]
    changepoints: list[Any] = data["changepoints"]

    ax_ts.plot(series.index, series.values, color=PALETTE["primary"], linewidth=1.3)
    for cp in changepoints:
        ax_ts.axvline(
            x=cp, color=PALETTE["accent"], linestyle="--", linewidth=1.2, alpha=0.8
        )
    ax_ts.set_title("Time Series with Changepoints", fontsize=12, fontweight="bold")
    ax_ts.set_xlabel("Date")
    ax_ts.set_ylabel("Value")

    pre_post: dict[str, dict[str, float]] = data["pre_post_stats"]
    metric_names = list(pre_post.keys())
    pre_vals = [pre_post[m]["pre"] for m in metric_names]
    post_vals = [pre_post[m]["post"] for m in metric_names]

    x = np.arange(len(metric_names))
    bar_width = 0.35
    ax_bar.bar(
        x - bar_width / 2, pre_vals, bar_width, color=PALETTE["secondary"], label="Pre"
    )
    ax_bar.bar(
        x + bar_width / 2, post_vals, bar_width, color=PALETTE["primary"], label="Post"
    )
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(metric_names, rotation=45, ha="right")
    ax_bar.set_title("Pre/Post Comparison", fontsize=12, fontweight="bold")
    ax_bar.set_ylabel("Value")
    ax_bar.legend(loc="upper right")

    fig.suptitle("AI Effect Dashboard", fontsize=16, fontweight="bold", y=1.01)
    return fig


if __name__ == "__main__":
    print("Dashboard module ready. Use create_overview_dashboard() with data dict.")
