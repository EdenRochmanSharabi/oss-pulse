"""PR abandonment analysis via survival models and classifiers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def build_survival_data(pr_df: pd.DataFrame) -> pd.DataFrame:
    """Build one row per PR with duration, event indicator, and covariates.

    duration_days: time from creation to merge/close or to reference_date
    if still open. event: 1 if the PR reached a terminal state (merged or
    closed), 0 if still open or abandoned without closure.
    """
    reference_date = pr_df["pr_created_at"].max()

    end_date = (
        pr_df["pr_merged_at"].fillna(pr_df["pr_closed_at"]).fillna(reference_date)
    )
    duration = (end_date - pr_df["pr_created_at"]).dt.total_seconds() / 86400.0

    event = pr_df["pr_outcome"].isin(["merged", "closed"]).astype(int)

    survival = pd.DataFrame(
        {
            "repo_name": pr_df["repo_name"],
            "pr_number": pr_df["pr_number"],
            "duration_days": duration,
            "event": event,
            "author_type": pr_df["author_class"],
            "org_type": pr_df.get("org_type", pd.Series("unknown", index=pr_df.index)),
            "pr_size_bucket": pr_df["pr_size_bucket"],
        }
    )

    survival = survival.drop_duplicates(subset=["repo_name", "pr_number"], keep="last")
    survival["duration_days"] = survival["duration_days"].clip(lower=0.01)

    return survival.reset_index(drop=True)


def fit_kaplan_meier(surv_df: pd.DataFrame) -> KaplanMeierFitter:
    """Fit a Kaplan-Meier estimator to the survival data."""
    kmf = KaplanMeierFitter()
    kmf.fit(
        durations=surv_df["duration_days"],
        event_observed=surv_df["event"],
        label="PR survival",
    )
    return kmf


def fit_cox_ph(surv_df: pd.DataFrame, covariates: list[str]) -> CoxPHFitter:
    """Fit a Cox proportional hazards model with the specified covariates.

    Categorical covariates are one-hot encoded before fitting.
    """
    fit_df = surv_df[["duration_days", "event"] + covariates].copy()
    fit_df = pd.get_dummies(fit_df, columns=covariates, drop_first=True)

    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(fit_df, duration_col="duration_days", event_col="event")
    return cph


def fit_abandonment_classifier(
    features_df: pd.DataFrame,
    target: str = "is_abandoned",
) -> dict[str, Any]:
    """Train RF and XGBoost classifiers to predict PR abandonment.

    features_df should contain numeric feature columns plus a binary target
    column. Returns AUC scores, the best model name, and feature importances.
    """
    feature_cols = [c for c in features_df.columns if c != target]
    X = features_df[feature_cols]
    y = features_df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    rf_auc = float(roc_auc_score(y_test, rf_proba))

    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
    )
    xgb.fit(X_train, y_train)
    xgb_proba = xgb.predict_proba(X_test)[:, 1]
    xgb_auc = float(roc_auc_score(y_test, xgb_proba))

    best_model = "xgboost" if xgb_auc >= rf_auc else "random_forest"

    best_importances = (
        xgb.feature_importances_ if best_model == "xgboost" else rf.feature_importances_
    )
    importance_df = pd.DataFrame(
        {"feature": feature_cols, "importance": best_importances}
    ).sort_values("importance", ascending=False)

    return {
        "rf_auc": rf_auc,
        "xgb_auc": xgb_auc,
        "best_model": best_model,
        "feature_importance": importance_df,
        "rf_model": rf,
        "xgb_model": xgb,
    }


if __name__ == "__main__":
    data_path = Path("data/processed/pr_events_featured.parquet")
    df = pd.read_parquet(data_path)

    print(f"Loaded {len(df)} PR events")

    surv = build_survival_data(df)
    print(f"Survival data: {len(surv)} PRs, {surv['event'].sum()} observed events")

    kmf = fit_kaplan_meier(surv)
    median_survival = kmf.median_survival_time_
    print(f"Median PR survival time: {median_survival:.1f} days")

    print("\nSurvival probabilities at key timepoints:")
    for days in [7, 30, 90, 180, 365]:
        prob = kmf.predict(days)
        print(f"  {days:>3d} days: {prob:.3f}")
