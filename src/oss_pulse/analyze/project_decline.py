"""Project-level decline prediction using repo health and activity features."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from oss_pulse.analyze.health_index import compute_health_components
from oss_pulse.visualize.style import save_fig, setup_style


def _gini(values: np.ndarray) -> float:
    """Compute the Gini coefficient of a 1-D array of non-negative values."""
    arr = np.sort(values.astype(float))
    n = len(arr)
    if n == 0 or arr.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float(
        (2.0 * np.sum(index * arr) - (n + 1) * np.sum(arr)) / (n * np.sum(arr))
    )


def label_declining_repos(repo_monthly: pd.DataFrame) -> pd.DataFrame:
    """Label repos as declining (1) or stable (0).

    Compares average monthly PR count in H2-2024 (Jul-Dec) vs H1-2025
    (Jan-May).  A repo is 'declining' when its H1-2025 average is less
    than 50% of the H2-2024 average.

    Returns a DataFrame with columns [repo_name, avg_h2_2024,
    avg_h1_2025, decline_ratio, is_declining].
    """
    h2_2024 = repo_monthly[
        (repo_monthly["year"] == 2024)
        & (repo_monthly["month"] >= 7)
        & (repo_monthly["month"] <= 12)
    ]
    h1_2025 = repo_monthly[
        (repo_monthly["year"] == 2025)
        & (repo_monthly["month"] >= 1)
        & (repo_monthly["month"] <= 5)
    ]

    avg_h2 = h2_2024.groupby("repo_name")["pr_count"].mean().rename("avg_h2_2024")
    avg_h1 = h1_2025.groupby("repo_name")["pr_count"].mean().rename("avg_h1_2025")

    labels = pd.merge(avg_h2, avg_h1, left_index=True, right_index=True, how="inner")
    labels["decline_ratio"] = labels["avg_h1_2025"] / labels["avg_h2_2024"].replace(
        0, np.nan
    )
    labels["is_declining"] = (labels["decline_ratio"] < 0.5).astype(int)
    labels = labels.reset_index()

    return labels


def build_features(
    repo_monthly: pd.DataFrame,
    pr_events: pd.DataFrame,
    top_repos: pd.DataFrame,
) -> pd.DataFrame:
    """Build per-repo feature matrix from historical data (before 2025).

    Features include health-index components, stars, total PR count,
    contributor concentration (Gini), first-timer ratio, bot ratio,
    median merge time, and one-hot language columns.
    """
    # Use only pre-2025 data for features
    hist_monthly = repo_monthly[repo_monthly["year"] < 2025].copy()
    hist_prs = pr_events[pr_events["pr_created_at"] < "2025-01-01"].copy()

    # --- Health components (from historical monthly data) ---
    health = compute_health_components(hist_monthly)

    # --- Stars ---
    stars = top_repos[["repo_name", "stars"]].drop_duplicates("repo_name")

    # --- Total historical PRs ---
    total_prs = (
        hist_monthly.groupby("repo_name")["pr_count"]
        .sum()
        .rename("total_prs")
        .reset_index()
    )

    # --- Contributor concentration (Gini of author PR counts) ---
    author_counts = (
        hist_prs.groupby(["repo_name", "author"]).size().reset_index(name="n_prs")
    )
    gini_per_repo = (
        author_counts.groupby("repo_name")["n_prs"]
        .apply(lambda x: _gini(x.values))
        .rename("contributor_concentration")
        .reset_index()
    )

    # --- First-timer ratio ---
    ft = (
        hist_prs.groupby("repo_name")["author_class"]
        .apply(lambda x: float((x == "first-timer").sum()) / max(len(x), 1))
        .rename("first_timer_ratio")
        .reset_index()
    )

    # --- Bot ratio (from monthly data, averaged) ---
    bot = (
        hist_monthly.groupby("repo_name")["bot_ratio"]
        .mean()
        .rename("bot_ratio_avg")
        .reset_index()
    )

    # --- Median merge time (from monthly data, median of medians) ---
    mmt = (
        hist_monthly.groupby("repo_name")["median_merge_time_hours"]
        .median()
        .rename("median_merge_time")
        .reset_index()
    )

    # --- Language one-hot (top 8) ---
    lang = top_repos[["repo_name", "language"]].drop_duplicates("repo_name").copy()
    top_langs = lang["language"].value_counts().head(8).index.tolist()
    lang["language_clean"] = lang["language"].where(
        lang["language"].isin(top_langs), "Other"
    )
    lang_dummies = pd.get_dummies(lang["language_clean"], prefix="lang")
    lang_features = pd.concat([lang[["repo_name"]], lang_dummies], axis=1)

    # --- Merge all features ---
    features = health.copy()
    for right_df in [stars, total_prs, gini_per_repo, ft, bot, mmt, lang_features]:
        features = features.merge(right_df, on="repo_name", how="left")

    features = features.fillna(0)
    return features


def train_decline_classifiers(
    features_df: pd.DataFrame,
    target: str = "is_declining",
) -> dict[str, Any]:
    """Train RF and XGBoost classifiers to predict project decline.

    Returns AUC scores, best model info, and feature importances.
    """
    feature_cols = [c for c in features_df.columns if c not in (target, "repo_name")]
    X = features_df[feature_cols].copy()
    y = features_df[target].copy()

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
        "best_auc": max(rf_auc, xgb_auc),
        "feature_importance": importance_df,
        "rf_model": rf,
        "xgb_model": xgb,
    }


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 15,
) -> plt.Figure:
    """Create horizontal bar chart of feature importances."""
    setup_style()

    top = importance_df.head(top_n).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top["feature"], top["importance"], color="#92b1f5", edgecolor="none")
    ax.set_xlabel("Feature Importance")
    ax.set_title("Project Decline Predictor: Feature Importance")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return fig


def update_stats(
    stats_path: Path,
    n_declining: int,
    n_stable: int,
    best_model: str,
    best_auc: float,
    top_3_features: list[str],
) -> None:
    """Add project_decline section to stats.json."""
    with open(stats_path) as f:
        stats = json.load(f)

    stats["project_decline"] = {
        "n_declining": n_declining,
        "n_stable": n_stable,
        "n_total": n_declining + n_stable,
        "best_model": best_model,
        "best_auc": round(best_auc, 3),
        "top_3_features": top_3_features,
    }

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    data_dir = Path("data")

    # Load data
    repo_monthly = pd.read_parquet(data_dir / "processed" / "repo_monthly.parquet")
    pr_events = pd.read_parquet(data_dir / "processed" / "pr_events_featured.parquet")
    top_repos = pd.read_parquet(data_dir / "raw" / "top_repos.parquet")

    print(
        f"Loaded {len(repo_monthly)} monthly records for "
        f"{repo_monthly['repo_name'].nunique()} repos"
    )
    print(f"Loaded {len(pr_events)} PR events")

    # Step 1: Label repos
    labels = label_declining_repos(repo_monthly)
    n_declining = int(labels["is_declining"].sum())
    n_stable = int((labels["is_declining"] == 0).sum())
    print(f"\nDeclining repos: {n_declining}")
    print(f"Stable repos: {n_stable}")

    # Step 5: Print declining repos
    declining = labels[labels["is_declining"] == 1].sort_values("decline_ratio")
    print("\nDeclining repos (sorted by severity):")
    for _, row in declining.iterrows():
        ratio_pct = row["decline_ratio"] * 100
        print(
            f"  {row['repo_name']:50s}  "
            f"H2-2024 avg: {row['avg_h2_2024']:6.1f}  "
            f"H1-2025 avg: {row['avg_h1_2025']:6.1f}  "
            f"ratio: {ratio_pct:5.1f}%"
        )

    # Step 2: Build features
    features = build_features(repo_monthly, pr_events, top_repos)
    dataset = features.merge(
        labels[["repo_name", "is_declining"]], on="repo_name", how="inner"
    )
    print(
        f"\nFeature matrix: {dataset.shape[0]} repos, {dataset.shape[1] - 2} features"
    )

    # Step 3: Train classifiers
    results = train_decline_classifiers(dataset)
    print(f"\nRandom Forest AUC: {results['rf_auc']:.3f}")
    print(f"XGBoost AUC:       {results['xgb_auc']:.3f}")
    print(f"Best model:        {results['best_model']}")

    # Step 4: Feature importance figure
    importance = results["feature_importance"]
    print("\nTop 10 features:")
    for _, row in importance.head(10).iterrows():
        print(f"  {row['feature']:35s}  {row['importance']:.4f}")

    fig = plot_feature_importance(importance)
    path = save_fig(fig, "project_decline_features")
    plt.close(fig)
    print(f"\nSaved feature importance figure to {path}")

    # Step 6: Update stats.json
    top_3 = [
        {"feature": row["feature"], "importance": round(float(row["importance"]), 3)}
        for _, row in importance.head(3).iterrows()
    ]
    stats_path = data_dir / "processed" / "stats.json"
    update_stats(
        stats_path,
        n_declining=n_declining,
        n_stable=n_stable,
        best_model=results["best_model"],
        best_auc=results["best_auc"],
        top_3_features=top_3,
    )
    print(f"Updated {stats_path}")
