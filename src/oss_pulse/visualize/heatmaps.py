"""Heatmap visualizations for activity patterns and correlations."""

from __future__ import annotations

import matplotlib.figure as mfigure
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from oss_pulse.visualize.style import PALETTE, setup_style

_DAY_LABELS: list[str] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def plot_activity_heatmap(
    df: pd.DataFrame, title: str = "PR Activity"
) -> mfigure.Figure:
    """Plot hour-of-day vs day-of-week activity heatmap.

    Expects columns 'hour' (0-23) and 'day_of_week' (0-6, Monday=0).
    """
    setup_style()

    pivot = df.pivot_table(
        index="day_of_week",
        columns="hour",
        aggfunc="size",
        fill_value=0,
    )
    pivot = pivot.reindex(index=range(7), columns=range(24), fill_value=0)

    fig, ax = plt.subplots(figsize=(16, 6))
    sns.heatmap(
        pivot,
        ax=ax,
        cmap="Blues",
        linewidths=0.5,
        linecolor=PALETTE["bg"],
        cbar_kws={"label": "Count"},
    )
    ax.set_yticklabels(_DAY_LABELS, rotation=0)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Day of Week")
    ax.set_title(title, fontsize=14, fontweight="bold")
    return fig


def plot_monthly_heatmap(
    df: pd.DataFrame,
    repos: list[str] | None = None,
    title: str = "Monthly PR Volume",
) -> mfigure.Figure:
    """Plot repo x year-month heatmap of PR counts.

    Expects columns: repo_name, year, month, pr_count.
    """
    setup_style()

    data = df.copy()
    if repos is not None:
        data = data[data["repo_name"].isin(repos)]

    data["year_month"] = (
        data["year"].astype(str) + "-" + data["month"].astype(str).str.zfill(2)
    )

    pivot = data.pivot_table(
        index="repo_name",
        columns="year_month",
        values="pr_count",
        aggfunc="sum",
        fill_value=0,
    )
    pivot = pivot[sorted(pivot.columns)]

    fig, ax = plt.subplots(figsize=(16, max(4, len(pivot) * 0.6)))
    sns.heatmap(
        pivot,
        ax=ax,
        cmap="YlOrRd",
        annot=True,
        fmt=".0f",
        linewidths=0.5,
        linecolor=PALETTE["bg"],
        cbar_kws={"label": "PR Count"},
    )
    ax.set_xlabel("Year-Month")
    ax.set_ylabel("Repository")
    ax.set_title(title, fontsize=14, fontweight="bold")
    return fig


def plot_correlation_matrix(
    df: pd.DataFrame, metrics: list[str], title: str = "Correlation Matrix"
) -> mfigure.Figure:
    """Plot annotated correlation heatmap for the specified metric columns."""
    setup_style()

    corr = df[metrics].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        linecolor=PALETTE["bg"],
        square=True,
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    return fig
