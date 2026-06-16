"""Time-series forecasting models with benchmarking for PR activity data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX


def _compute_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    """Compute MAE, RMSE, and MAPE between actual and predicted values."""
    residuals = actual - predicted
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals**2)))

    nonzero_mask = actual != 0
    if nonzero_mask.any():
        mape = float(
            np.mean(np.abs(residuals[nonzero_mask] / actual[nonzero_mask])) * 100
        )
    else:
        mape = float("inf")

    return {"mae": mae, "rmse": rmse, "mape": mape}


def _temporal_split(
    series: pd.Series, train_ratio: float = 0.8
) -> tuple[pd.Series, pd.Series]:
    """Split a time series into train/test by temporal order."""
    split_idx = int(len(series) * train_ratio)
    return series.iloc[:split_idx], series.iloc[split_idx:]


def fit_arima(series: pd.Series) -> dict[str, Any]:
    """Fit a SARIMAX model with automatic order selection via AIC grid search.

    Uses a temporal 80/20 train/test split for evaluation.
    """
    train, test = _temporal_split(series)

    best_aic = float("inf")
    best_order: tuple[int, int, int] = (1, 1, 0)

    for p in range(3):
        for d in range(2):
            for q in range(3):
                try:
                    model = SARIMAX(
                        train,
                        order=(p, d, q),
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    fitted = model.fit(disp=False)
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = (p, d, q)
                except Exception:
                    continue

    final_model = SARIMAX(
        train,
        order=best_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted_model = final_model.fit(disp=False)

    predictions = fitted_model.forecast(steps=len(test))
    predictions.index = test.index
    metrics = _compute_metrics(test, predictions)

    return {
        "model": fitted_model,
        "predictions": predictions,
        **metrics,
    }


def fit_prophet(df: pd.DataFrame) -> dict[str, Any]:
    """Fit a Prophet model on a DataFrame with 'ds' and 'y' columns.

    Uses a temporal 80/20 train/test split for evaluation.
    """
    from prophet import Prophet

    sorted_df = df.sort_values("ds").reset_index(drop=True)
    split_idx = int(len(sorted_df) * 0.8)
    train_df = sorted_df.iloc[:split_idx]
    test_df = sorted_df.iloc[split_idx:]

    model = Prophet(yearly_seasonality=True, weekly_seasonality=False)
    model.fit(train_df)

    future = model.make_future_dataframe(periods=len(test_df), freq="W")
    forecast = model.predict(future)

    pred_df = forecast.iloc[split_idx:][["ds", "yhat"]].reset_index(drop=True)
    test_reset = test_df.reset_index(drop=True)

    predictions = pd.Series(pred_df["yhat"].values, index=test_reset.index)
    actual = pd.Series(test_reset["y"].values, index=test_reset.index)
    metrics = _compute_metrics(actual, predictions)

    return {
        "model": model,
        "predictions": predictions,
        **metrics,
    }


def fit_ets(series: pd.Series) -> dict[str, Any]:
    """Fit an Exponential Smoothing (ETS) model.

    Uses a temporal 80/20 train/test split for evaluation.
    """
    train, test = _temporal_split(series)

    model = ExponentialSmoothing(
        train,
        trend="add",
        seasonal=None,
        initialization_method="estimated",
    )
    fitted_model = model.fit(optimized=True)

    predictions = fitted_model.forecast(steps=len(test))
    predictions.index = test.index
    metrics = _compute_metrics(test, predictions)

    return {
        "model": fitted_model,
        "predictions": predictions,
        **metrics,
    }


def fit_xgboost_ts(df: pd.DataFrame, lags: int = 12) -> dict[str, Any]:
    """Fit an XGBRegressor using lag features derived from the 'y' column.

    Uses a temporal 80/20 train/test split for evaluation.
    """
    from xgboost import XGBRegressor

    sorted_df = df.sort_values("ds").reset_index(drop=True)
    y = sorted_df["y"]

    features = pd.DataFrame(index=sorted_df.index)
    for lag in range(1, lags + 1):
        features[f"lag_{lag}"] = y.shift(lag)

    features = features.dropna()
    y_aligned = y.loc[features.index]

    split_idx = int(len(features) * 0.8)
    x_train = features.iloc[:split_idx]
    x_test = features.iloc[split_idx:]
    y_train = y_aligned.iloc[:split_idx]
    y_test = y_aligned.iloc[split_idx:]

    model = XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
    )
    model.fit(x_train, y_train, verbose=False)

    raw_predictions = model.predict(x_test)
    predictions = pd.Series(raw_predictions, index=y_test.index)
    metrics = _compute_metrics(y_test, predictions)

    return {
        "model": model,
        "predictions": predictions,
        **metrics,
    }


def benchmark_models(series: pd.Series) -> pd.DataFrame:
    """Run all four forecasting models and return a comparison DataFrame.

    Models that fail are skipped gracefully.
    """
    prophet_df = pd.DataFrame({"ds": series.index, "y": series.values})

    model_runners: list[tuple[str, Any]] = [
        ("ARIMA", lambda: fit_arima(series)),
        ("Prophet", lambda: fit_prophet(prophet_df)),
        ("ETS", lambda: fit_ets(series)),
        ("XGBoost", lambda: fit_xgboost_ts(prophet_df)),
    ]

    results: list[dict[str, Any]] = []
    for name, runner in model_runners:
        try:
            outcome = runner()
            results.append(
                {
                    "model": name,
                    "mae": outcome["mae"],
                    "rmse": outcome["rmse"],
                    "mape": outcome["mape"],
                }
            )
        except Exception as exc:
            print(f"[WARN] {name} failed: {exc}")

    return pd.DataFrame(results, columns=["model", "mae", "rmse", "mape"])


if __name__ == "__main__":
    data_path = Path("data/processed/repo_weekly.parquet")
    df = pd.read_parquet(data_path)

    # Aggregate across all repos: sum pr_count per week
    agg = df.groupby("year_week")["pr_count"].sum().sort_index()
    agg.index.name = "year_week"

    print(f"Benchmarking models on AGGREGATE weekly PR volume ({len(agg)} weeks, "
          f"{df['repo_name'].nunique()} repos)\n")

    comparison = benchmark_models(agg)
    print(comparison.to_string(index=False))

    # Update stats.json
    stats_path = Path("data/processed/stats.json")
    stats_data: dict[str, Any] = {}
    if stats_path.exists():
        with open(stats_path) as f:
            stats_data = json.load(f)

    models_list = [
        {
            "model": row["model"],
            "mae": round(float(row["mae"]), 4),
            "rmse": round(float(row["rmse"]), 4),
            "mape": round(float(row["mape"]), 4),
        }
        for _, row in comparison.iterrows()
    ]
    best_row = comparison.loc[comparison["mae"].idxmin()] if not comparison.empty else None
    best_model = str(best_row["model"]) if best_row is not None else None

    stats_data["forecast_benchmark"] = {
        "models": models_list,
        "best_model": best_model,
    }

    with open(stats_path, "w") as f:
        json.dump(stats_data, f, indent=2)
    print(f"\nUpdated {stats_path}")
