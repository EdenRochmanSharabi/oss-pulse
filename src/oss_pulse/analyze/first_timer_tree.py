"""First-timer merge prediction via per-language decision trees.

Trains a shallow DecisionTreeClassifier for each major language to predict
whether a first-timer's PR will be merged.  Also identifies the most
welcoming repos per language (high first-timer merge rate with enough volume).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

matplotlib.use("Agg")

# Languages to analyze (must have >= 100 first-timer PRs).
TARGET_LANGUAGES = [
    "Python",
    "TypeScript",
    "Go",
    "Rust",
    "C++",
    "Java",
    "C",
    "Ruby",
    "C#",
    "JavaScript",
]

FEATURE_COLS = [
    "pr_size",
    "is_weekend",
    "repo_merge_rate",
    "repo_total_prs",
    "additions",
    "deletions",
    "changed_files",
    "hour",
    "day_of_week",
]

MIN_FIRST_TIMER_PRS = 100
MIN_REPO_FIRST_TIMER_PRS = 20


# ── Helpers ───────────────────────────────────────────────────────────


def _extract_decision_rules(tree: DecisionTreeClassifier,
                            feature_names: list[str]) -> str:
    """Extract a concise human-readable summary of the tree's main merge path.

    Walks the tree from root, following the branch at each node that leads
    to the highest merged-class probability, and summarises the conditions.
    """
    tree_ = tree.tree_
    merged_class_idx = list(tree.classes_).index(1)
    rules: list[str] = []

    node = 0
    while tree_.feature[node] >= 0:  # not a leaf
        feat = feature_names[tree_.feature[node]]
        threshold = tree_.threshold[node]
        left = tree_.children_left[node]
        right = tree_.children_right[node]

        # Pick the child whose leaf descendants have higher merge probability
        left_merged = tree_.value[left][0][merged_class_idx]
        left_total = tree_.value[left][0].sum()
        right_merged = tree_.value[right][0][merged_class_idx]
        right_total = tree_.value[right][0].sum()
        left_rate = left_merged / left_total if left_total > 0 else 0
        right_rate = right_merged / right_total if right_total > 0 else 0

        if right_rate >= left_rate:
            rules.append(f"{feat} > {threshold:.2f}")
            node = right
        else:
            rules.append(f"{feat} <= {threshold:.2f}")
            node = left

    # Leaf info
    leaf_vals = tree_.value[node][0]
    merge_prob = leaf_vals[merged_class_idx] / leaf_vals.sum()

    condition = " AND ".join(rules)
    outcome = "likely merged" if merge_prob > 0.5 else "likely not merged"
    return f"{condition} -> {outcome} ({merge_prob:.0%})"


def _build_first_timer_df(
    pr_df: pd.DataFrame,
    repos_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build a feature matrix of first-timer PRs joined with repo metadata."""
    # Deduplicate to one row per PR, keep human first-timers only
    df = pr_df.drop_duplicates(subset=["repo_name", "pr_number"], keep="last").copy()
    df = df[~df["is_bot"] & (df["author_class"] == "first-timer")].copy()

    # Drop PRs without size data
    df = df[df["additions"] > 0].copy()

    # Target: merged or not
    df["is_merged"] = (df["pr_outcome"] == "merged").astype(int)

    # PR size feature
    df["pr_size"] = df["additions"] + df["deletions"]
    df["is_weekend"] = df["is_weekend"].astype(int)

    # Repo merge rate: computed from ALL human PRs per repo (not just first-timers)
    all_human = pr_df[~pr_df["is_bot"]].drop_duplicates(
        subset=["repo_name", "pr_number"], keep="last"
    )
    repo_stats = all_human.groupby("repo_name").agg(
        _n_merged=("pr_outcome", lambda s: (s == "merged").sum()),
        _n_terminal=("pr_outcome", lambda s: s.isin(["merged", "closed"]).sum()),
        repo_total_prs=("pr_number", "count"),
    )
    repo_stats["repo_merge_rate"] = (
        repo_stats["_n_merged"] / repo_stats["_n_terminal"].clip(lower=1)
    )
    df = df.merge(
        repo_stats[["repo_merge_rate", "repo_total_prs"]],
        left_on="repo_name",
        right_index=True,
        how="left",
    )
    df["repo_merge_rate"] = df["repo_merge_rate"].fillna(0.5)
    df["repo_total_prs"] = df["repo_total_prs"].fillna(0).astype(int)

    # Join language and stars from top_repos
    df = df.merge(
        repos_df[["repo_name", "language", "stars"]],
        on="repo_name",
        how="left",
    )

    cols_needed = FEATURE_COLS + ["is_merged", "repo_name", "language", "stars"]
    df = df[cols_needed].dropna().reset_index(drop=True)
    return df


def _top_repos_for_language(
    lang_df: pd.DataFrame,
    n: int = 5,
) -> list[dict[str, Any]]:
    """Find the top-n repos for first-timers in a language group.

    Requires >= MIN_REPO_FIRST_TIMER_PRS first-timer PRs per repo.
    Sorted by first-timer merge rate descending.
    """
    repo_agg = lang_df.groupby("repo_name").agg(
        ft_prs=("is_merged", "count"),
        ft_merged=("is_merged", "sum"),
        stars=("stars", "first"),
        repo_merge_rate=("repo_merge_rate", "first"),
    )
    repo_agg = repo_agg[repo_agg["ft_prs"] >= MIN_REPO_FIRST_TIMER_PRS]
    repo_agg["ft_merge_rate"] = repo_agg["ft_merged"] / repo_agg["ft_prs"]
    repo_agg = repo_agg.sort_values("ft_merge_rate", ascending=False).head(n)

    results = []
    for repo, row in repo_agg.iterrows():
        results.append({
            "repo": repo,
            "ft_merge_rate": round(float(row["ft_merge_rate"] * 100), 1),
            "overall_merge_rate": round(float(row["repo_merge_rate"] * 100), 1),
            "ft_prs": int(row["ft_prs"]),
            "stars": int(row["stars"]),
        })
    return results


# ── Visualization ─────────────────────────────────────────────────────


def plot_tree_for_language(
    tree: DecisionTreeClassifier,
    feature_names: list[str],
    language: str,
    output_dir: Path,
) -> Path:
    """Render a decision tree as an SVG figure."""
    fig, ax = plt.subplots(figsize=(24, 10))
    plot_tree(
        tree,
        feature_names=feature_names,
        class_names=["not merged", "merged"],
        filled=True,
        rounded=True,
        fontsize=8,
        ax=ax,
        impurity=False,
        proportion=True,
    )
    ax.set_title(
        f"First-Timer Merge Prediction: {language}",
        fontsize=14,
        fontweight="bold",
    )
    fig_path = output_dir / f"first_timer_tree_{language.lower().replace('+', 'p').replace('#', 'sharp')}.svg"
    fig.savefig(fig_path, format="svg", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved tree figure: {fig_path}")
    return fig_path


# ── Main analysis ─────────────────────────────────────────────────────


def run_first_timer_analysis(
    data_path: Path | None = None,
    repos_path: Path | None = None,
    output_dir: Path | None = None,
    stats_path: Path | None = None,
) -> dict[str, Any]:
    """Run per-language decision-tree analysis for first-timer merge prediction.

    1. Load PR events and repo metadata.
    2. For each language with enough data, train a DecisionTreeClassifier.
    3. Extract feature importances and human-readable decision rules.
    4. Identify the most welcoming repos per language.
    5. Generate tree visualizations for Python, Rust, and Go.
    6. Update stats.json.
    """
    if data_path is None:
        data_path = Path("data/processed/pr_events_featured.parquet")
    if repos_path is None:
        repos_path = Path("data/raw/top_repos.parquet")
    if output_dir is None:
        output_dir = Path("output/figures")
    if stats_path is None:
        stats_path = Path("data/processed/stats.json")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    pr_df = pd.read_parquet(data_path)
    repos_df = pd.read_parquet(repos_path)
    print(f"  PR events: {len(pr_df):,}")
    print(f"  Repos:     {len(repos_df):,}")

    print("Building first-timer feature matrix...")
    ft_df = _build_first_timer_df(pr_df, repos_df)
    print(f"  First-timer PRs with features: {len(ft_df):,}")

    results: dict[str, Any] = {}
    tree_langs_to_plot = ["Python", "Rust", "Go"]

    for lang in TARGET_LANGUAGES:
        lang_df = ft_df[ft_df["language"] == lang]
        if len(lang_df) < MIN_FIRST_TIMER_PRS:
            print(f"  {lang}: skipped ({len(lang_df)} < {MIN_FIRST_TIMER_PRS} PRs)")
            continue

        X = lang_df[FEATURE_COLS]
        y = lang_df["is_merged"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y,
        )

        clf = DecisionTreeClassifier(
            max_depth=4,
            min_samples_leaf=50,
            random_state=42,
        )
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)

        acc = float(accuracy_score(y_test, y_pred))

        # AUC: handle single-class edge cases
        if len(y_proba.shape) == 2 and y_proba.shape[1] == 2:
            merged_idx = list(clf.classes_).index(1)
            auc_val = float(roc_auc_score(y_test, y_proba[:, merged_idx]))
        else:
            auc_val = 0.0

        # Feature importances
        importances = sorted(
            zip(FEATURE_COLS, clf.feature_importances_),
            key=lambda x: x[1],
            reverse=True,
        )

        # Decision rules
        rules = _extract_decision_rules(clf, FEATURE_COLS)

        # Recommended repos
        recommended = _top_repos_for_language(lang_df)

        results[lang] = {
            "accuracy": round(acc, 3),
            "auc": round(auc_val, 3),
            "n_samples": int(len(lang_df)),
            "feature_importances": [
                {"feature": f, "importance": round(float(imp), 3)}
                for f, imp in importances
            ],
            "decision_rules": rules,
            "recommended_repos": recommended,
        }

        top_feat = importances[0]
        print(
            f"  {lang:12s}: n={len(lang_df):>6,}, "
            f"acc={acc:.3f}, auc={auc_val:.3f}, "
            f"top={top_feat[0]} ({top_feat[1]:.3f})"
        )

        # Tree visualizations for select languages
        if lang in tree_langs_to_plot:
            plot_tree_for_language(clf, FEATURE_COLS, lang, output_dir)

    # ── Recommendations figure ────────────────────────────────────────
    _plot_recommendations(results, output_dir)

    # ── Update stats.json ─────────────────────────────────────────────
    stats_data: dict[str, Any] = {}
    if stats_path.exists():
        with open(stats_path) as f:
            stats_data = json.load(f)

    stats_data["first_timer_trees"] = results

    with open(stats_path, "w") as f:
        json.dump(stats_data, f, indent=2)
    print(f"\nUpdated {stats_path}")

    return results


def _plot_recommendations(
    results: dict[str, Any],
    output_dir: Path,
) -> None:
    """Create a multi-panel figure of recommended repos per language."""
    from oss_pulse.visualize.style import PALETTE, setup_style

    setup_style()

    # Filter to languages that have recommended repos
    langs_with_recs = [
        lang for lang in TARGET_LANGUAGES
        if lang in results and results[lang]["recommended_repos"]
    ]
    if not langs_with_recs:
        print("  No recommended repos to plot.")
        return

    n_langs = len(langs_with_recs)
    fig, axes = plt.subplots(
        n_langs, 1,
        figsize=(14, max(6, n_langs * 2.5)),
        squeeze=False,
    )

    cmap = plt.cm.RdYlGn  # type: ignore[attr-defined]

    for idx, lang in enumerate(langs_with_recs):
        ax = axes[idx, 0]
        recs = results[lang]["recommended_repos"]
        if not recs:
            ax.set_visible(False)
            continue

        repos = [r["repo"] for r in recs]
        rates = [r["ft_merge_rate"] for r in recs]
        stars = [r["stars"] for r in recs]

        # Normalize rates to 0-1 for colormap (rates are percentages 0-100)
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
    fig.tight_layout()

    fig_path = output_dir / "first_timer_recommendations.svg"
    fig.savefig(fig_path, format="svg", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved recommendations figure: {fig_path}")


if __name__ == "__main__":
    run_first_timer_analysis()
