"""Comparison and distribution plots: box, violin, funnel, survival, ROC."""

from __future__ import annotations

from typing import Any

import matplotlib.figure as mfigure
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from oss_pulse.visualize.style import PALETTE, setup_style


def plot_boxplot_comparison(
    df: pd.DataFrame,
    metric: str,
    group: str,
    title: str | None = None,
) -> mfigure.Figure:
    """Grouped boxplot comparing distributions of `metric` across `group`."""
    setup_style()

    fig, ax = plt.subplots()
    sns.boxplot(
        data=df,
        x=group,
        y=metric,
        ax=ax,
        color=PALETTE["primary"],
        linewidth=0.8,
    )
    ax.set_title(title or f"{metric} by {group}", fontsize=14, fontweight="bold")
    ax.set_xlabel(group)
    ax.set_ylabel(metric)
    return fig


def plot_violin_comparison(
    df: pd.DataFrame,
    metric: str,
    group: str,
    title: str | None = None,
) -> mfigure.Figure:
    """Grouped violin plot comparing distributions of `metric` across `group`."""
    setup_style()

    fig, ax = plt.subplots()
    sns.violinplot(
        data=df,
        x=group,
        y=metric,
        ax=ax,
        color=PALETTE["primary"],
        linewidth=0.8,
        inner="box",
    )
    ax.set_title(title or f"{metric} by {group}", fontsize=14, fontweight="bold")
    ax.set_xlabel(group)
    ax.set_ylabel(metric)
    return fig


def plot_funnel(
    funnel_df: pd.DataFrame, title: str = "Contributor Funnel"
) -> mfigure.Figure:
    """Horizontal bar chart showing funnel stages with a gradient effect.

    Expects columns: 'stage' (str) and 'count' (numeric).
    """
    setup_style()

    sorted_df = funnel_df.sort_values("count", ascending=False).reset_index(drop=True)
    n_stages = len(sorted_df)

    base_colors = [
        PALETTE["primary"],
        PALETTE["secondary"],
        PALETTE["success"],
        PALETTE["warning"],
        PALETTE["accent"],
    ]
    colors = [base_colors[i % len(base_colors)] for i in range(n_stages)]

    fig, ax = plt.subplots(figsize=(12, max(4, n_stages * 0.8)))
    bars = ax.barh(
        sorted_df["stage"],
        sorted_df["count"],
        color=colors,
        edgecolor=PALETTE["bg"],
        linewidth=1.5,
    )

    for bar, count in zip(bars, sorted_df["count"], strict=True):
        ax.text(
            bar.get_width() + sorted_df["count"].max() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{count:,.0f}",
            va="center",
            fontsize=10,
        )

    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_title(title, fontsize=14, fontweight="bold")
    return fig


def plot_survival_curves(
    curves: list[tuple[Any, str]], title: str = "Survival Curves"
) -> mfigure.Figure:
    """Plot Kaplan-Meier survival curves on shared axes.

    Each tuple is (KaplanMeierFitter, label).
    """
    setup_style()

    cycle_colors = [
        PALETTE["primary"],
        PALETTE["accent"],
        PALETTE["success"],
        PALETTE["warning"],
        PALETTE["secondary"],
    ]

    fig, ax = plt.subplots()
    for i, (kmf, label) in enumerate(curves):
        color = cycle_colors[i % len(cycle_colors)]
        kmf.plot_survival_function(ax=ax, label=label, color=color, linewidth=1.5)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival Probability")
    ax.legend()
    return fig


def plot_roc_curve(
    y_true: Any,
    y_scores: dict[str, Any],
    title: str = "ROC Curve",
) -> mfigure.Figure:
    """Plot ROC curves for multiple models with AUC in the legend.

    y_scores maps model_name to predicted probabilities.
    """
    from sklearn.metrics import auc, roc_curve

    setup_style()

    cycle_colors = [
        PALETTE["primary"],
        PALETTE["accent"],
        PALETTE["success"],
        PALETTE["warning"],
        PALETTE["secondary"],
    ]

    fig, ax = plt.subplots()

    for i, (name, scores) in enumerate(y_scores.items()):
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        color = cycle_colors[i % len(cycle_colors)]
        ax.plot(
            fpr, tpr, color=color, linewidth=1.5, label=f"{name} (AUC = {roc_auc:.3f})"
        )

    ax.plot(
        [0, 1],
        [0, 1],
        color=PALETTE["grid"],
        linestyle="--",
        linewidth=1.0,
        label="Random",
    )
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    return fig
