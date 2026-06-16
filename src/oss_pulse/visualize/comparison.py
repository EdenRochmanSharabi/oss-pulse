"""Comparison and distribution plots: box, violin, funnel, survival, ROC,
first-timer recommendations."""

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
    ax.set_xlim((0.0, 1.0))
    ax.set_ylim((0.0, 1.05))
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    return fig


def plot_first_timer_recommendations(
    tree_results: dict[str, Any],
    languages: list[str] | None = None,
) -> mfigure.Figure:
    """Horizontal bar chart of top repos for first-time contributors per language.

    ``tree_results`` is the ``first_timer_trees`` dict from stats.json.
    Each language panel shows up to 5 repos coloured by first-timer merge rate.
    """
    setup_style()

    if languages is None:
        languages = [
            lang for lang in [
                "Python", "TypeScript", "Go", "Rust", "C++",
                "Java", "C", "Ruby", "C#", "JavaScript",
            ]
            if lang in tree_results and tree_results[lang].get("recommended_repos")
        ]

    n_langs = len(languages)
    if n_langs == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
        return fig

    fig, axes = plt.subplots(
        n_langs, 1,
        figsize=(14, max(6, n_langs * 2.5)),
        squeeze=False,
    )

    cmap = plt.cm.RdYlGn  # type: ignore[attr-defined]

    for idx, lang in enumerate(languages):
        ax = axes[idx, 0]
        recs = tree_results[lang]["recommended_repos"]

        repos = [r["repo"] for r in recs]
        rates = [r["ft_merge_rate"] for r in recs]
        stars = [r["stars"] for r in recs]

        norm_rates = [r / 100.0 for r in rates]
        colors = [cmap(nr) for nr in norm_rates]

        bars = ax.barh(repos[::-1], rates[::-1], color=colors[::-1], edgecolor="none")

        for bar, star_count in zip(bars, stars[::-1]):
            ax.text(
                bar.get_width() + 0.5,
                bar.get_y() + bar.get_height() / 2,
                f"{star_count:,} stars",
                va="center",
                fontsize=8,
                color=PALETTE["secondary"],
            )

        ax.set_xlim(0, 105)
        ax.set_xlabel("First-Timer Merge Rate (%)" if idx == n_langs - 1 else "")
        ax.set_title(f"{lang}", fontsize=11, fontweight="bold", loc="left")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Best Repos for First-Time Contributors by Language",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    return fig
