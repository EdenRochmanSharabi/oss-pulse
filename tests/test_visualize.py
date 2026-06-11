"""Smoke tests for all visualization functions."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.figure as mfigure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _close_all_figures() -> None:
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


class TestStyle:
    def test_setup_style_does_not_crash(self) -> None:
        from oss_pulse.visualize.style import setup_style

        setup_style()

    def test_save_fig_creates_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from oss_pulse.visualize.style import save_fig

        # Monkeypatch the output directory so save_fig writes to tmp_path
        monkeypatch.chdir(tmp_path)
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        path = save_fig(fig, "test_plot")
        assert path.exists()
        assert path.suffix == ".png"


class TestTimeseries:
    def test_plot_decomposition(self) -> None:
        from oss_pulse.visualize.timeseries import plot_decomposition

        n = 100
        index = pd.date_range("2020-01-01", periods=n, freq="W")
        mock_result = SimpleNamespace(
            observed=pd.Series(np.random.default_rng(42).normal(0, 1, n), index=index),
            trend=pd.Series(np.linspace(0, 10, n), index=index),
            seasonal=pd.Series(np.sin(np.linspace(0, 4 * np.pi, n)), index=index),
            resid=pd.Series(np.random.default_rng(42).normal(0, 0.5, n), index=index),
        )
        fig = plot_decomposition(mock_result)
        assert isinstance(fig, mfigure.Figure)

    def test_plot_forecast(self) -> None:
        from oss_pulse.visualize.timeseries import plot_forecast

        dates = pd.date_range("2020-01-01", periods=50, freq="W")
        actual = pd.Series(np.linspace(10, 50, 50), index=dates)
        predicted = pd.Series(np.linspace(12, 48, 50), index=dates)
        fig = plot_forecast(actual, predicted)
        assert isinstance(fig, mfigure.Figure)

    def test_plot_forecast_with_ci(self) -> None:
        from oss_pulse.visualize.timeseries import plot_forecast

        dates = pd.date_range("2020-01-01", periods=50, freq="W")
        actual = pd.Series(np.linspace(10, 50, 50), index=dates)
        predicted = pd.Series(np.linspace(12, 48, 50), index=dates)
        ci_lower = predicted - 5
        ci_upper = predicted + 5
        fig = plot_forecast(actual, predicted, ci_lower=ci_lower, ci_upper=ci_upper)
        assert isinstance(fig, mfigure.Figure)


class TestHeatmaps:
    def test_plot_activity_heatmap(self) -> None:
        from oss_pulse.visualize.heatmaps import plot_activity_heatmap

        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "hour": rng.integers(0, 24, 200),
                "day_of_week": rng.integers(0, 7, 200),
            }
        )
        fig = plot_activity_heatmap(df)
        assert isinstance(fig, mfigure.Figure)

    def test_plot_monthly_heatmap(self) -> None:
        from oss_pulse.visualize.heatmaps import plot_monthly_heatmap

        df = pd.DataFrame(
            {
                "repo_name": ["org/repo-a"] * 6 + ["org/repo-b"] * 6,
                "year": [2024] * 12,
                "month": list(range(1, 7)) * 2,
                "pr_count": np.random.default_rng(42).integers(5, 50, 12),
            }
        )
        fig = plot_monthly_heatmap(df)
        assert isinstance(fig, mfigure.Figure)

    def test_plot_correlation_matrix(self) -> None:
        from oss_pulse.visualize.heatmaps import plot_correlation_matrix

        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "metric_a": rng.normal(0, 1, 50),
                "metric_b": rng.normal(0, 1, 50),
                "metric_c": rng.normal(0, 1, 50),
            }
        )
        fig = plot_correlation_matrix(df, metrics=["metric_a", "metric_b", "metric_c"])
        assert isinstance(fig, mfigure.Figure)


class TestComparison:
    def test_plot_boxplot_comparison(self) -> None:
        from oss_pulse.visualize.comparison import plot_boxplot_comparison

        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "merge_time": rng.lognormal(3, 1, 60),
                "org_type": ["company"] * 20 + ["foundation"] * 20 + ["community"] * 20,
            }
        )
        fig = plot_boxplot_comparison(df, metric="merge_time", group="org_type")
        assert isinstance(fig, mfigure.Figure)

    def test_plot_violin_comparison(self) -> None:
        from oss_pulse.visualize.comparison import plot_violin_comparison

        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "merge_time": rng.lognormal(3, 1, 60),
                "org_type": ["company"] * 20 + ["foundation"] * 20 + ["community"] * 20,
            }
        )
        fig = plot_violin_comparison(df, metric="merge_time", group="org_type")
        assert isinstance(fig, mfigure.Figure)

    def test_plot_funnel(self) -> None:
        from oss_pulse.visualize.comparison import plot_funnel

        df = pd.DataFrame(
            {
                "stage": ["1st_pr", "2nd_pr", "5th_pr", "10th_pr", "regular"],
                "count": [1000, 600, 300, 100, 30],
            }
        )
        fig = plot_funnel(df)
        assert isinstance(fig, mfigure.Figure)


class TestTimeseriesExtra:
    def test_plot_acf_pacf(self) -> None:
        from oss_pulse.visualize.timeseries import plot_acf_pacf

        acf_vals = np.linspace(1, 0, 20)
        pacf_vals = np.linspace(1, 0, 20)
        fig = plot_acf_pacf(acf_vals, pacf_vals)
        assert isinstance(fig, mfigure.Figure)


class TestComparisonExtra:
    def test_plot_survival_curves(self) -> None:
        from lifelines import KaplanMeierFitter

        from oss_pulse.visualize.comparison import plot_survival_curves

        rng = np.random.default_rng(42)
        kmf1 = KaplanMeierFitter()
        kmf1.fit(
            rng.exponential(10, 50),
            event_observed=rng.binomial(1, 0.8, 50),
            label="Group A",
        )
        kmf2 = KaplanMeierFitter()
        kmf2.fit(
            rng.exponential(15, 50),
            event_observed=rng.binomial(1, 0.7, 50),
            label="Group B",
        )

        fig = plot_survival_curves([(kmf1, "Group A"), (kmf2, "Group B")])
        assert isinstance(fig, mfigure.Figure)

    def test_plot_roc_curve(self) -> None:
        from oss_pulse.visualize.comparison import plot_roc_curve

        rng = np.random.default_rng(42)
        y_true = rng.binomial(1, 0.5, 100)
        y_scores = {
            "Model A": rng.uniform(0, 1, 100),
            "Model B": rng.uniform(0, 1, 100),
        }
        fig = plot_roc_curve(y_true, y_scores)
        assert isinstance(fig, mfigure.Figure)


class TestDashboard:
    def test_create_overview_dashboard(self) -> None:
        from oss_pulse.visualize.dashboard import create_overview_dashboard

        rng = np.random.default_rng(42)
        dates = pd.date_range("2024-01-01", periods=52, freq="W")
        data = {
            "weekly_trend": pd.DataFrame(
                {"date": dates, "pr_count": rng.integers(50, 200, 52)}
            ),
            "health_ranking": pd.DataFrame(
                {
                    "repo_name": [f"org/repo-{i}" for i in range(15)],
                    "health_index": rng.uniform(30, 90, 15),
                }
            ),
            "merge_times": pd.Series(rng.lognormal(3, 1, 200)),
            "activity_df": pd.DataFrame(
                {
                    "hour": rng.integers(0, 24, 500),
                    "day_of_week": rng.integers(0, 7, 500),
                }
            ),
        }
        fig = create_overview_dashboard(data)
        assert isinstance(fig, mfigure.Figure)
