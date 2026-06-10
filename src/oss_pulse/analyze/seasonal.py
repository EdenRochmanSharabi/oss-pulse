"""Seasonal decomposition and stationarity testing for PR time series."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf


def stl_decompose(series: pd.Series, period: int = 52) -> Any:
    """Decompose a time series via STL into trend, seasonal, and residual.

    The series index should be a DatetimeIndex or PeriodIndex.
    """
    stl = STL(series, period=period, robust=True)
    return stl.fit()


def test_stationarity(series: pd.Series) -> dict[str, Any]:
    """Run ADF and KPSS tests and return a combined stationarity verdict.

    is_stationary is True when ADF rejects the unit-root null (p < 0.05)
    AND KPSS fails to reject the level-stationarity null (p > 0.05).
    """
    adf_result = adfuller(series.dropna(), autolag="AIC")
    adf_stat: float = float(adf_result[0])
    adf_pvalue: float = float(adf_result[1])

    kpss_result = kpss(series.dropna(), regression="c", nlags="auto")
    kpss_stat: float = float(kpss_result[0])
    kpss_pvalue: float = float(kpss_result[1])

    return {
        "adf_stat": adf_stat,
        "adf_pvalue": adf_pvalue,
        "kpss_stat": kpss_stat,
        "kpss_pvalue": kpss_pvalue,
        "is_stationary": adf_pvalue < 0.05 and kpss_pvalue > 0.05,
    }


def compute_acf_pacf(
    series: pd.Series, nlags: int = 52
) -> tuple[np.ndarray, np.ndarray]:
    """Compute autocorrelation and partial autocorrelation values."""
    clean = series.dropna()
    acf_values: np.ndarray = acf(clean, nlags=nlags)
    pacf_values: np.ndarray = pacf(clean, nlags=nlags)
    return acf_values, pacf_values


if __name__ == "__main__":
    data_path = Path("data/processed/repo_weekly.parquet")
    df = pd.read_parquet(data_path)

    first_repo: str = df["repo_name"].iloc[0]
    repo_df = df[df["repo_name"] == first_repo].sort_values("year_week")
    series = repo_df.set_index("year_week")["pr_count"]

    print(f"Repo: {first_repo}")
    print(f"Series length: {len(series)}")

    result = stl_decompose(series)
    print(
        f"\nSTL decomposition complete. Trend range: "
        f"[{result.trend.min():.2f}, {result.trend.max():.2f}]"
    )

    stationarity = test_stationarity(series)
    print("\nStationarity results:")
    for key, value in stationarity.items():
        print(f"  {key}: {value}")
