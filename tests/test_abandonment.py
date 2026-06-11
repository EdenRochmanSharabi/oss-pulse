"""Tests for survival analysis and abandonment prediction."""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture()
def mock_pr_data() -> pd.DataFrame:
    """Minimal PR data for survival analysis tests."""
    rows = []
    for i in range(200):
        created = pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=i % 60)
        if i % 3 == 0:
            state, merged = "closed", "true"
            merged_at = created + pd.Timedelta(hours=i * 2 + 1)
            closed_at = merged_at
        elif i % 3 == 1:
            state, merged = "closed", "false"
            merged_at = pd.NaT
            closed_at = created + pd.Timedelta(days=i % 30 + 1)
        else:
            state, merged = "open", "false"
            merged_at = pd.NaT
            closed_at = pd.NaT

        rows.append(
            {
                "repo_name": f"org/repo-{i % 5}",
                "pr_number": i,
                "state": state,
                "merged": merged,
                "pr_created_at": created,
                "pr_merged_at": merged_at,
                "pr_closed_at": closed_at,
                "pr_updated_at": closed_at
                if pd.notna(closed_at)
                else created + pd.Timedelta(days=1),
                "pr_outcome": (
                    "merged" if i % 3 == 0 else ("closed" if i % 3 == 1 else "open")
                ),
                "author_class": "regular" if i % 4 else "first-timer",
                "author_type": "regular" if i % 4 else "first-timer",
                "org_type": "company" if i % 2 else "community",
                "pr_size_bucket": "small" if i % 3 == 0 else "medium",
                "time_to_merge_hours": float(i * 2 + 1) if i % 3 == 0 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def test_survival_data_format(mock_pr_data: pd.DataFrame) -> None:
    from oss_pulse.analyze.abandonment import build_survival_data

    surv = build_survival_data(mock_pr_data)
    assert "duration_days" in surv.columns
    assert "event" in surv.columns
    assert (surv["duration_days"] > 0).all()
    assert surv["event"].isin([0, 1]).all()


def test_kaplan_meier_monotonic(mock_pr_data: pd.DataFrame) -> None:
    from oss_pulse.analyze.abandonment import build_survival_data, fit_kaplan_meier

    surv = build_survival_data(mock_pr_data)
    kmf = fit_kaplan_meier(surv)
    sf = kmf.survival_function_
    assert sf.iloc[0, 0] >= sf.iloc[-1, 0]


def test_cox_ph_fits(mock_pr_data: pd.DataFrame) -> None:
    from oss_pulse.analyze.abandonment import build_survival_data, fit_cox_ph

    surv = build_survival_data(mock_pr_data)
    cph = fit_cox_ph(surv, covariates=["author_type", "org_type"])
    assert cph.summary is not None
    assert len(cph.summary) > 0


def test_abandonment_classifier(mock_pr_data: pd.DataFrame) -> None:
    import numpy as np

    from oss_pulse.analyze.abandonment import fit_abandonment_classifier

    rng = np.random.default_rng(42)
    n = len(mock_pr_data)
    features_df = pd.DataFrame(
        {
            "feat_a": rng.normal(0, 1, n),
            "feat_b": rng.normal(0, 1, n),
            "feat_c": rng.normal(0, 1, n),
            "is_abandoned": rng.binomial(1, 0.3, n),
        }
    )
    result = fit_abandonment_classifier(features_df)
    assert "rf_auc" in result
    assert "xgb_auc" in result
    assert "best_model" in result
    assert result["best_model"] in ("random_forest", "xgboost")
    assert 0 <= result["rf_auc"] <= 1
    assert 0 <= result["xgb_auc"] <= 1
    assert "feature_importance" in result
    assert len(result["feature_importance"]) == 3
