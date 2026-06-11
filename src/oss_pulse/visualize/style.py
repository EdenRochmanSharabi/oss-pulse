"""Shared style configuration and figure-saving utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.figure as mfigure
import matplotlib.pyplot as plt

PALETTE: dict[str, str] = {
    "primary": "#2563eb",
    "secondary": "#64748b",
    "accent": "#dc2626",
    "success": "#16a34a",
    "warning": "#d97706",
    "bg": "#ffffff",
    "grid": "#e2e8f0",
}


def setup_style() -> None:
    """Configure matplotlib rcParams for an academic-clean look."""
    plt.style.use("seaborn-v0_8-whitegrid")

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 11,
            "figure.figsize": (16, 9),
            "figure.facecolor": PALETTE["bg"],
            "axes.facecolor": PALETTE["bg"],
            "axes.edgecolor": PALETTE["secondary"],
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.5,
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "legend.loc": "upper left",
            "legend.borderaxespad": 0.0,
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    )


def save_fig(
    fig: mfigure.Figure, name: str, dpi: int = 300, fmt: str = "svg"
) -> Path:
    """Save figure to output/figures/{name}.{fmt}."""
    output_dir = Path("output/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    path = output_dir / f"{name}.{fmt}"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    return path
