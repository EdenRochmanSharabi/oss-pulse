"""Time-series visualization: decomposition, forecasts, and autocorrelation."""

from __future__ import annotations

from typing import Any

import matplotlib.figure as mfigure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from oss_pulse.visualize.style import PALETTE, setup_style


def plot_decomposition(result: Any, title: str = "STL Decomposition") -> mfigure.Figure:
    """Plot a 4-panel STL decomposition (observed, trend, seasonal, residual)."""
    setup_style()

    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)

    components: list[tuple[str, Any, str]] = [
        ("Observed", result.observed, PALETTE["primary"]),
        ("Trend", result.trend, PALETTE["success"]),
        ("Seasonal", result.seasonal, PALETTE["warning"]),
        ("Residual", result.resid, PALETTE["accent"]),
    ]

    for ax, (label, data, color) in zip(axes, components, strict=True):
        ax.plot(data, color=color, linewidth=1.2)
        ax.set_ylabel(label, fontsize=11)
        ax.tick_params(axis="both", labelsize=9)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    return fig


def plot_forecast(
    actual: pd.Series,
    predicted: pd.Series,
    ci_lower: pd.Series | None = None,
    ci_upper: pd.Series | None = None,
    title: str = "Forecast",
) -> mfigure.Figure:
    """Plot actual vs predicted with optional confidence interval band."""
    setup_style()

    fig, ax = plt.subplots()

    ax.plot(
        actual.index,
        actual.to_numpy(),
        color=PALETTE["primary"],
        linewidth=1.5,
        label="Actual",
    )
    ax.plot(
        predicted.index,
        predicted.to_numpy(),
        color=PALETTE["accent"],
        linewidth=1.5,
        linestyle="--",
        label="Predicted",
    )

    if ci_lower is not None and ci_upper is not None:
        ax.fill_between(
            predicted.index,
            ci_lower.to_numpy(),
            ci_upper.to_numpy(),
            color=PALETTE["accent"],
            alpha=0.2,
            label="95% CI",
        )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()
    return fig


def plot_acf_pacf(
    acf_vals: Any, pacf_vals: Any, title: str = "Autocorrelation"
) -> mfigure.Figure:
    """Plot ACF and PACF as side-by-side bar charts."""
    setup_style()

    fig, (ax_acf, ax_pacf) = plt.subplots(2, 1, figsize=(16, 9))

    lags_acf = np.arange(len(acf_vals))
    ax_acf.bar(lags_acf, acf_vals, width=0.4, color=PALETTE["primary"], alpha=0.8)
    ax_acf.axhline(y=0, color=PALETTE["secondary"], linewidth=0.8)
    ax_acf.set_title("ACF", fontsize=12)
    ax_acf.set_xlabel("Lag")
    ax_acf.set_ylabel("Correlation")

    lags_pacf = np.arange(len(pacf_vals))
    ax_pacf.bar(lags_pacf, pacf_vals, width=0.4, color=PALETTE["accent"], alpha=0.8)
    ax_pacf.axhline(y=0, color=PALETTE["secondary"], linewidth=0.8)
    ax_pacf.set_title("PACF", fontsize=12)
    ax_pacf.set_xlabel("Lag")
    ax_pacf.set_ylabel("Correlation")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.0)
    return fig
