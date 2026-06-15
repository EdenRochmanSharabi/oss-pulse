"""PR abandonment analysis via survival models and classifiers.

Predicts which PRs will be abandoned using features of the PR itself
(size, timing, repo context), not the author's identity.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

matplotlib.use("Agg")

# ── Feature labels for human-readable output ───────────────────────────
FEATURE_LABELS: dict[str, str] = {
    "additions": "Lines added",
    "deletions": "Lines deleted",
    "changed_files": "Files changed",
    "hour": "Hour of day (UTC)",
    "day_of_week": "Day of week",
    "is_weekend": "Submitted on weekend",
    "pr_size_ordinal": "PR size (S/M/L)",
    "repo_recent_prs": "Repo PRs in prior 30 days",
    "repo_merge_rate": "Repo historical merge rate",
    "is_first_pr_to_repo": "First PR to this repo",
    "total_lines": "Total lines changed",
}

# ── Survival analysis helpers (unchanged) ──────────────────────────────


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


# ── PR-centric abandonment features ───────────────────────────────────


def build_pr_abandonment_features(pr_df: pd.DataFrame) -> pd.DataFrame:
    """Build a feature matrix for PR abandonment prediction.

    Uses only features of the PR itself (size, timing, repo context),
    not the author's identity.  Only PRs with additions > 0 are included
    (Search API PRs lack size data).
    """
    # Deduplicate to one row per PR
    df = pr_df.drop_duplicates(subset=["repo_name", "pr_number"], keep="last").copy()

    # Filter to PRs with size data
    df = df[df["additions"] > 0].copy()

    # Target
    df["is_abandoned"] = (df["pr_outcome"] == "abandoned").astype(int)

    # Sort by creation time for rolling computations
    df = df.sort_values("pr_created_at").reset_index(drop=True)

    # ── PR size features ──
    df["total_lines"] = df["additions"] + df["deletions"]

    # PR size bucket as ordinal
    size_map = {"small": 0, "medium": 1, "large": 2}
    df["pr_size_ordinal"] = df["pr_size_bucket"].map(size_map).fillna(1).astype(int)

    # ── Timing features (already in data) ──
    df["is_weekend"] = df["is_weekend"].astype(int)

    # ── Repo context (vectorized) ──
    # 1. repo_recent_prs: count of PRs in the same repo in the 30 days before
    #    Approximated via a rolling count on a date index per repo.
    df["_created_date"] = df["pr_created_at"].dt.date

    # Group-level cumulative count with 30-day window
    # We use a merge-based approach: for each repo, count PRs per day,
    # then do a rolling 30-day sum.
    repo_day_counts = (
        df.groupby(["repo_name", "_created_date"]).size().reset_index(name="_day_count")
    )
    repo_day_counts["_created_date"] = pd.to_datetime(repo_day_counts["_created_date"])

    # For each repo, build a daily series and compute rolling 30-day sum
    recent_prs_map: dict[tuple[str, str], int] = {}
    for repo, rdf in repo_day_counts.groupby("repo_name"):
        rdf = rdf.set_index("_created_date").sort_index()
        # Reindex to fill gaps, then rolling sum of prior 30 days (exclude today)
        idx = pd.date_range(rdf.index.min(), rdf.index.max(), freq="D")
        daily = rdf["_day_count"].reindex(idx, fill_value=0)
        # Shift by 1 so today is excluded, then rolling 30
        shifted = daily.shift(1, fill_value=0)
        rolling_30 = shifted.rolling(30, min_periods=1).sum()
        for dt, val in rolling_30.items():
            recent_prs_map[(repo, str(dt.date()))] = int(val)

    df["repo_recent_prs"] = df.apply(
        lambda r: recent_prs_map.get((r["repo_name"], str(r["_created_date"])), 0),
        axis=1,
    )

    # 2. repo_merge_rate: overall merge rate of the repo (static per repo).
    #    Using the full historical rate is a good proxy and avoids O(N^2).
    repo_stats = df.groupby("repo_name").agg(
        _n_merged=("pr_outcome", lambda s: (s == "merged").sum()),
        _n_terminal=("pr_outcome", lambda s: s.isin(["merged", "closed"]).sum()),
    )
    repo_stats["repo_merge_rate"] = repo_stats["_n_merged"] / repo_stats[
        "_n_terminal"
    ].clip(lower=1)
    df = df.merge(
        repo_stats[["repo_merge_rate"]],
        left_on="repo_name",
        right_index=True,
        how="left",
    )
    df["repo_merge_rate"] = df["repo_merge_rate"].fillna(0.5)

    # 3. is_first_pr_to_repo: 1 if the author has no prior PRs to this repo.
    has_author = "author" in df.columns
    if has_author:
        df["_author_repo_cumcount"] = df.groupby(["repo_name", "author"]).cumcount()
        df["is_first_pr_to_repo"] = (df["_author_repo_cumcount"] == 0).astype(int)
    else:
        df["is_first_pr_to_repo"] = 1

    # ── Select final features ──
    feature_cols = [
        "additions",
        "deletions",
        "changed_files",
        "hour",
        "day_of_week",
        "is_weekend",
        "pr_size_ordinal",
        "repo_recent_prs",
        "repo_merge_rate",
        "is_first_pr_to_repo",
        "total_lines",
        "is_abandoned",
    ]

    result = df[feature_cols].copy()
    result = result.dropna()

    return result.reset_index(drop=True)


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

    # Compute abandonment rates by group for actionable insights
    insights: dict[str, Any] = {}

    # Size-based abandonment
    if "total_lines" in features_df.columns:
        small = features_df[features_df["total_lines"] < 50]
        large = features_df[features_df["total_lines"] > 500]
        if len(small) > 0 and len(large) > 0:
            insights["abandon_rate_under_50_lines"] = float(small[target].mean())
            insights["abandon_rate_over_500_lines"] = float(large[target].mean())

    # Weekend vs weekday
    if "is_weekend" in features_df.columns:
        wkend = features_df[features_df["is_weekend"] == 1]
        wkday = features_df[features_df["is_weekend"] == 0]
        if len(wkend) > 0 and len(wkday) > 0:
            insights["abandon_rate_weekend"] = float(wkend[target].mean())
            insights["abandon_rate_weekday"] = float(wkday[target].mean())

    # First PR to repo
    if "is_first_pr_to_repo" in features_df.columns:
        first = features_df[features_df["is_first_pr_to_repo"] == 1]
        repeat = features_df[features_df["is_first_pr_to_repo"] == 0]
        if len(first) > 0 and len(repeat) > 0:
            insights["abandon_rate_first_pr"] = float(first[target].mean())
            insights["abandon_rate_repeat_author"] = float(repeat[target].mean())

    # Low vs high merge-rate repos
    if "repo_merge_rate" in features_df.columns:
        features_df["repo_merge_rate"].median()
        low_mr = features_df[features_df["repo_merge_rate"] < 0.5]
        high_mr = features_df[features_df["repo_merge_rate"] >= 0.7]
        if len(low_mr) > 0 and len(high_mr) > 0:
            insights["abandon_rate_low_merge_repo"] = float(low_mr[target].mean())
            insights["abandon_rate_high_merge_repo"] = float(high_mr[target].mean())

    return {
        "rf_auc": rf_auc,
        "xgb_auc": xgb_auc,
        "best_model": best_model,
        "feature_importance": importance_df,
        "rf_model": rf,
        "xgb_model": xgb,
        "insights": insights,
    }


def plot_feature_importance(
    importance_df: pd.DataFrame,
    output_path: Path,
    bar_color: str = "#92b1f5",
) -> None:
    """Save a horizontal bar chart of feature importances."""
    df = importance_df.copy()
    # Map feature names to human-readable labels
    df["label"] = df["feature"].map(FEATURE_LABELS).fillna(df["feature"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        df["label"][::-1],
        df["importance"][::-1],
        color=bar_color,
        edgecolor="none",
    )
    ax.set_xlabel("Feature Importance", fontsize=11)
    ax.set_title("What Makes a PR Get Ignored?", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(output_path, format="svg", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved figure: {output_path}")


# ── Main runner ────────────────────────────────────────────────────────


def run_abandonment_analysis(
    data_path: Path | None = None,
    output_dir: Path | None = None,
    stats_path: Path | None = None,
) -> dict[str, Any]:
    """Run the full PR-centric abandonment analysis.

    1. Load data and build PR-level features (no author identity).
    2. Train RF + XGBoost (80/20 stratified split).
    3. Generate feature importance figure.
    4. Print top features and actionable insights.
    5. Update stats.json with results.
    """
    if data_path is None:
        data_path = Path("data/processed/pr_events_featured.parquet")
    if output_dir is None:
        output_dir = Path("output/figures")
    if stats_path is None:
        stats_path = Path("data/processed/stats.json")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = pd.read_parquet(data_path)
    print(f"  Loaded {len(df):,} PR events")

    print("Building PR-centric features (no author identity)...")
    features = build_pr_abandonment_features(df)
    n_abandoned = features["is_abandoned"].sum()
    n_total = len(features)
    print(f"  {n_total:,} PRs, {n_abandoned:,} abandoned")

    print("Training classifiers (RF + XGBoost, 80/20 stratified)...")
    result = fit_abandonment_classifier(features)

    rf_auc = result["rf_auc"]
    xgb_auc = result["xgb_auc"]
    best = result["best_model"]
    best_auc = max(rf_auc, xgb_auc)
    importance_df = result["feature_importance"]
    insights = result["insights"]

    print(f"\n  Random Forest AUC:  {rf_auc:.3f}")
    print(f"  XGBoost AUC:        {xgb_auc:.3f}")
    print(f"  Best model:         {best} (AUC={best_auc:.3f})")

    print("\nTop features:")
    for _, row in importance_df.head(5).iterrows():
        label = FEATURE_LABELS.get(row["feature"], row["feature"])
        print(f"  {label:30s}  {row['importance']:.3f}")

    # Plot
    fig_path = output_dir / "abandonment_feature_importance.svg"
    plot_feature_importance(importance_df, fig_path)

    # Actionable insights
    print("\nActionable insights:")
    if (
        "abandon_rate_under_50_lines" in insights
        and "abandon_rate_over_500_lines" in insights
    ):
        r_small = insights["abandon_rate_under_50_lines"]
        r_large = insights["abandon_rate_over_500_lines"]
        ratio = r_large / r_small if r_small > 0 else float("inf")
        lg = 100 * r_large
        sm = 100 * r_small
        print(f"  >500 lines: {lg:.1f}% vs <50: {sm:.1f}% ({ratio:.1f}x)")

    if "abandon_rate_weekend" in insights and "abandon_rate_weekday" in insights:
        r_we = insights["abandon_rate_weekend"]
        r_wd = insights["abandon_rate_weekday"]
        pct_more = 100 * (r_we - r_wd) / r_wd if r_wd > 0 else 0
        we = 100 * r_we
        wd = 100 * r_wd
        print(f"  Weekend: {we:.1f}% vs weekday: {wd:.1f}% ({pct_more:+.0f}%)")

    if "abandon_rate_first_pr" in insights and "abandon_rate_repeat_author" in insights:
        r_first = insights["abandon_rate_first_pr"]
        r_repeat = insights["abandon_rate_repeat_author"]
        ratio = r_first / r_repeat if r_repeat > 0 else float("inf")
        fp = 100 * r_first
        rp = 100 * r_repeat
        print(f"  First PR: {fp:.1f}% vs repeat: {rp:.1f}% ({ratio:.1f}x)")

    if (
        "abandon_rate_low_merge_repo" in insights
        and "abandon_rate_high_merge_repo" in insights
    ):
        r_low = insights["abandon_rate_low_merge_repo"]
        r_high = insights["abandon_rate_high_merge_repo"]
        ratio = r_low / r_high if r_high > 0 else float("inf")
        lo = 100 * r_low
        hi = 100 * r_high
        print(f"  Low MR: {lo:.1f}% vs high: {hi:.1f}% ({ratio:.1f}x)")

    # Update stats.json
    stats_data: dict[str, Any] = {}
    if stats_path.exists():
        with open(stats_path) as f:
            stats_data = json.load(f)

    stats_data["pr_abandonment"] = {
        "n_prs_analyzed": int(n_total),
        "n_abandoned": int(n_abandoned),
        "abandon_rate_pct": round(100 * n_abandoned / n_total, 1),
        "rf_auc": round(rf_auc, 3),
        "xgb_auc": round(xgb_auc, 3),
        "best_model": best,
        "best_auc": round(best_auc, 3),
        "top_features": [
            {
                "feature": row["feature"],
                "label": FEATURE_LABELS.get(row["feature"], row["feature"]),
                "importance": round(float(row["importance"]), 3),
            }
            for _, row in importance_df.head(5).iterrows()
        ],
        "insights": {k: round(v, 4) for k, v in insights.items()},
    }

    with open(stats_path, "w") as f:
        json.dump(stats_data, f, indent=2)
    print(f"\nUpdated {stats_path}")

    return result


if __name__ == "__main__":
    run_abandonment_analysis()
