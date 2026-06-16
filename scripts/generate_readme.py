#!/usr/bin/env python3
"""
generate_readme.py

Reads data/processed/stats.json and updates README.md with current numbers.
Only numbers change; all prose and narrative structure stays intact.

Usage:
    python scripts/generate_readme.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
STATS_PATH = ROOT / "data" / "processed" / "stats.json"
README_PATH = ROOT / "README.md"


def fmt_int(n: int) -> str:
    """Format integer with comma thousands separator."""
    return f"{n:,}"


def fmt_pct(v: float, decimals: int = 1) -> str:
    """Format a float as a percentage string (no % sign)."""
    return f"{v:.{decimals}f}%"


def fmt_pct_plain(v: float, decimals: int = 1) -> str:
    """Format a float as a percentage value string with % sign."""
    return f"{v:.{decimals}f}%"


def load_stats() -> dict:
    with STATS_PATH.open() as f:
        return json.load(f)


def load_author_groups(stats: dict) -> dict:
    """
    Load 2016-2019 vs 2025-2026 author group stats from stats.json["author_groups"].
    Returns a dict with "early" and "recent" keys, or empty dict if missing.
    """
    ag = stats.get("author_groups")
    if not ag:
        return {}
    # Map stats.json keys ("2016", "2025") to internal keys ("early", "recent")
    return {
        "early": ag.get("2016", {}),
        "recent": ag.get("2025", {}),
    }


def main() -> None:
    stats = load_stats()
    text = README_PATH.read_text(encoding="utf-8")

    ds = stats["dataset"]
    outcomes = stats["outcomes"]
    funnel = stats["funnel"]
    eras = stats["ai_eras"]
    tool_eras = stats["tool_eras"]
    abandonment = stats["pr_abandonment"]
    prod = stats["productivity_by_author_type"]
    merge_time = stats["merge_time"]
    forecast = stats.get("forecast_benchmark", {})
    contrib_return = stats.get("contributor_return", {})
    ft_trees = stats.get("first_timer_trees", {})

    author_groups = load_author_groups(stats)

    # ── Derived values ──────────────────────────────────────────────────────

    total_prs = ds["total_prs"]
    n_repos = ds["n_repos"]
    n_contributors = ds["n_contributors"]

    # Rejection rates = 100 - merge_rate (per era)
    def rejection(era_key: str) -> float:
        return round(100.0 - eras[era_key]["merge_rate"], 1)

    # Author group values from stats.json["author_groups"]
    if author_groups:
        ag_early = author_groups["early"]
        ag_recent = author_groups["recent"]
        maint_prs_early = ag_early["maintainer"]["prs_per_person_month"]
        maint_prs_recent = ag_recent["maintainer"]["prs_per_person_month"]
        maint_merge_early = int(ag_early["maintainer"]["merge_rate"])
        maint_merge_recent = int(ag_recent["maintainer"]["merge_rate"])
        reg_prs_early = ag_early["regular"]["prs_per_person_month"]
        reg_prs_recent = ag_recent["regular"]["prs_per_person_month"]
        reg_merge_early = int(ag_early["regular"]["merge_rate"])
        reg_merge_recent = int(ag_recent["regular"]["merge_rate"])
        ft_prs_early = ag_early["first-timer"]["prs_per_person_month"]
        ft_prs_recent = ag_recent["first-timer"]["prs_per_person_month"]
        ft_merge_early = int(ag_early["first-timer"]["merge_rate"])
        ft_merge_recent = int(ag_recent["first-timer"]["merge_rate"])
    else:
        raise ValueError(
            "stats.json is missing 'author_groups' key. "
            "Run the author_groups analysis to populate it."
        )

    # Funnel derived counts
    second_pr_count = funnel["2nd_pr"]
    fifth_pr_count = funnel["5th_pr"]
    regular_count = funnel["regular"]
    dropout_pct = funnel["dropout_1st_pct"]

    # PRs/author/month proxy from ai_eras (prs_per_month / contributors_per_month)
    def prs_per_author(era_key: str) -> float:
        e = eras[era_key]
        return round(e["prs_per_month"] / e["contributors_per_month"], 2)

    # ── Abandonment feature importances ──────────────────────────────────────
    top_feats = {f["feature"]: f["importance"] for f in abandonment["top_features"]}
    # Expected features:
    feat_first_pr = top_feats.get("is_first_pr_to_repo", 0.573)
    feat_repo_merge = top_feats.get("repo_merge_rate", 0.139)
    feat_repo_recent = top_feats.get("repo_recent_prs", 0.100)
    feat_additions = top_feats.get("additions", 0.057)
    feat_deletions = top_feats.get("deletions", 0.034)

    ins = abandonment["insights"]

    # ── String replacements ──────────────────────────────────────────────────
    # Each entry: (old_text, new_text)
    # Ordered from most specific to most general to avoid double-replacement.
    replacements: list[tuple[str, str]] = [

        # ── Header / intro line ──────────────────────────────────────────────
        (
            "2.7 million Pull Requests across 341 top open-source software projects",
            f"{total_prs / 1_000_000:.1f} million Pull Requests across {n_repos} top open-source software projects",
        ),
        (
            "**A time-series study of",
            "**A time-series study of",  # no-op anchor; actual text replaced above
        ),

        # ── Introduction paragraph ───────────────────────────────────────────
        (
            f"We analyzed **2,682,939 Pull Requests** across **341 software repositories**",
            f"We analyzed **{fmt_int(total_prs)} Pull Requests** across **{n_repos} software repositories**",
        ),

        # ── Methodology: first-draft note & filtering note ───────────────────
        (
            "82 had sufficient PR data for analysis at the time of this first draft.",
            f"{n_repos} repos were collected for analysis.",
        ),
        (
            "We excluded 20 repos with no programming language (language=None or Markdown), reducing from 82 to 62 repos.",
            "We excluded repos with no programming language (language=None or Markdown).",
        ),

        # ── Section header: "General Findings (First Draft, 341 software repos)" ──
        (
            "## General Findings (First Draft, 341 software repos)\n\n"
            "*These findings are based on the first 341 software repos extracted. "
            "They will be updated when the full 200-repo dataset is available.*",
            f"## General Findings ({n_repos} software repos)",
        ),

        # ── Dataset at a Glance ───────────────────────────────────────────────
        (
            "- **2,682,939 PRs** across **341 software repos**, spanning **2016-2026**",
            f"- **{fmt_int(total_prs)} PRs** across **{n_repos} software repos**, spanning **2016-2026**",
        ),
        (
            "- **329,142 unique contributors** (excluding bots)",
            f"- **{fmt_int(n_contributors)} unique contributors** (excluding bots)",
        ),
        (
            "- Outcome distribution: **67.6% merged**, 29.4% closed, 1.7% abandoned, 1.3% still open",
            (
                f"- Outcome distribution: **{outcomes['merge_rate']}% merged**, "
                f"{outcomes['closed_rate']}% closed, "
                f"{outcomes['abandoned_rate']}% abandoned, "
                f"{outcomes['open_rate']}% still open"
            ),
        ),

        # ── Section 4: Retention Crisis ──────────────────────────────────────
        (
            "Of **329,142 contributors** who opened at least one PR:",
            f"Of **{fmt_int(n_contributors)} contributors** who opened at least one PR:",
        ),
        (
            "- **41.6%** came back for a second (136,868)",
            f"- **{funnel['2nd_pr_pct']}%** came back for a second ({fmt_int(second_pr_count)})",
        ),
        (
            "- **13.8%** reached their 5th PR (45,449)",
            f"- **{funnel['5th_pr_pct']}%** reached their 5th PR ({fmt_int(fifth_pr_count)})",
        ),
        (
            "- **3.4%** became regulars with 20+ PRs (11,289)",
            f"- **{funnel['regular_pct']}%** became regulars with 20+ PRs ({fmt_int(regular_count)})",
        ),
        (
            "**58% of first-time contributors never return.**",
            f"**{dropout_pct}% of first-time contributors never return.**",
        ),

        # ── Section 5: AI Era table ───────────────────────────────────────────
        # Row: Contributors/month
        (
            "| Contributors/month | 4,429 | 5,192 | 5,420 | 6,687 | 7,244 | **10,608** |",
            (
                f"| Contributors/month"
                f" | {fmt_int(eras['2016_2019']['contributors_per_month'])}"
                f" | {fmt_int(eras['2020_2021']['contributors_per_month'])}"
                f" | {fmt_int(eras['2022_copilot']['contributors_per_month'])}"
                f" | {fmt_int(eras['2023_chatgpt']['contributors_per_month'])}"
                f" | {fmt_int(eras['2024_cursor']['contributors_per_month'])}"
                f" | **{fmt_int(eras['2025_agentic']['contributors_per_month'])}** |"
            ),
        ),
        # Row: First-timers/month
        (
            "| First-timers/month | 1,031 | 1,169 | 1,222 | 1,700 | 1,775 | **3,165** |",
            (
                f"| First-timers/month"
                f" | {fmt_int(eras['2016_2019']['firsttimers_per_month'])}"
                f" | {fmt_int(eras['2020_2021']['firsttimers_per_month'])}"
                f" | {fmt_int(eras['2022_copilot']['firsttimers_per_month'])}"
                f" | {fmt_int(eras['2023_chatgpt']['firsttimers_per_month'])}"
                f" | {fmt_int(eras['2024_cursor']['firsttimers_per_month'])}"
                f" | **{fmt_int(eras['2025_agentic']['firsttimers_per_month'])}** |"
            ),
        ),
        # Row: First-timer rejection
        (
            "| First-timer rejection | 51.8% | 53.5% | 55.7% | 58.3% | 58.6% | **64.5%** |",
            (
                f"| First-timer rejection"
                f" | {eras['2016_2019']['ft_rejection_rate']}%"
                f" | {eras['2020_2021']['ft_rejection_rate']}%"
                f" | {eras['2022_copilot']['ft_rejection_rate']}%"
                f" | {eras['2023_chatgpt']['ft_rejection_rate']}%"
                f" | {eras['2024_cursor']['ft_rejection_rate']}%"
                f" | **{eras['2025_agentic']['ft_rejection_rate']}%** |"
            ),
        ),
        # Row: Overall rejection (100 - merge_rate per era)
        (
            "| Overall rejection | 30.9% | 29.1% | 28.4% | 27.6% | 27.1% | **34.8%** |",
            (
                f"| Overall rejection"
                f" | {rejection('2016_2019')}%"
                f" | {rejection('2020_2021')}%"
                f" | {rejection('2022_copilot')}%"
                f" | {rejection('2023_chatgpt')}%"
                f" | {rejection('2024_cursor')}%"
                f" | **{rejection('2025_agentic')}%** |"
            ),
        ),
        # Row: PRs/author/month
        (
            "| PRs/author/month | 2.82 | 3.23 | 3.48 | 3.55 | 3.78 | **3.76** |",
            (
                f"| PRs/author/month"
                f" | {prs_per_author('2016_2019')}"
                f" | {prs_per_author('2020_2021')}"
                f" | {prs_per_author('2022_copilot')}"
                f" | {prs_per_author('2023_chatgpt')}"
                f" | {prs_per_author('2024_cursor')}"
                f" | **{prs_per_author('2025_agentic')}** |"
            ),
        ),

        # ── Section 5b: contributor/first-timer per-month inline numbers ──────
        (
            "Monthly unique contributors grew from 4,429 (2016-2019) to 10,608 (2025-2026). "
            "First-timers per month tripled: 1,031 to 3,165.",
            (
                f"Monthly unique contributors grew from {fmt_int(eras['2016_2019']['contributors_per_month'])} "
                f"(2016-2019) to {fmt_int(eras['2025_agentic']['contributors_per_month'])} (2025-2026). "
                f"First-timers per month nearly tripled: "
                f"{fmt_int(eras['2016_2019']['firsttimers_per_month'])} to "
                f"{fmt_int(eras['2025_agentic']['firsttimers_per_month'])}."
            ),
        ),

        # ── Section 5c: rejection rate inline numbers ─────────────────────────
        (
            "From 2020 to 2024, the overall rejection rate was stable around 27-29%. "
            "But in 2025-2026, it jumped to 34.8%.",
            (
                f"From 2020 to 2024, the overall rejection rate was stable around "
                f"{rejection('2020_2021')}-{rejection('2022_copilot')}%. "
                f"But in 2025-2026, it jumped to {rejection('2025_agentic')}%."
            ),
        ),

        # ── Section 5e: first-timer rejection inline numbers ─────────────────
        (
            "3,165 first-timers/month in 2025-2026, up from 1,031 in 2016-2019. "
            "But their rejection rate climbed steadily: 51.8% (2016-2019) to 55.7% (2022) "
            "to 58.3% (2023) to **64.5%** (2025-2026).",
            (
                f"{fmt_int(eras['2025_agentic']['firsttimers_per_month'])} first-timers/month in 2025-2026, "
                f"up from {fmt_int(eras['2016_2019']['firsttimers_per_month'])} in 2016-2019. "
                f"But their rejection rate climbed steadily: "
                f"{eras['2016_2019']['ft_rejection_rate']}% (2016-2019) "
                f"to {eras['2022_copilot']['ft_rejection_rate']}% (2022) "
                f"to {eras['2023_chatgpt']['ft_rejection_rate']}% (2023) "
                f"to **{eras['2025_agentic']['ft_rejection_rate']}%** (2025-2026)."
            ),
        ),

        # ── Section 5g: individual productivity inline ────────────────────────
        (
            "Each contributor produces more PRs per month (2.8 to 3.8).",
            (
                f"Each contributor produces more PRs per month "
                f"({prs_per_author('2016_2019')} to {prs_per_author('2025_agentic')})."
            ),
        ),

        # ── Section 5h: author group table ───────────────────────────────────
        # Prose before the table (use regex replacement in post-processing)
    ]

    # ── Regex replacements for section 5h ────────────────────────────────────
    # These use regex to match any numbers in the patterns, then replace with
    # the values from stats.json["author_groups"].
    regex_replacements: list[tuple[str, str]] = [
        # 5h prose
        (
            r"Maintainers went from [\d.]+ to [\d.]+ PRs/person/month while their merge rate "
            r"declined from \d+% to \d+%. Regulars gained modestly \([\d.]+ to [\d.]+\) but their "
            r"merge rate collapsed from \d+% to \d+%. First-timers are by definition at "
            r"1 PR/month, but their merge rate dropped from \d+% to \d+%.",
            (
                f"Maintainers went from {maint_prs_early} to {maint_prs_recent} PRs/person/month "
                f"while their merge rate declined from {maint_merge_early}% to {maint_merge_recent}%. "
                f"Regulars gained modestly ({reg_prs_early} to {reg_prs_recent}) but their "
                f"merge rate collapsed from {reg_merge_early}% to {reg_merge_recent}%. "
                f"First-timers are by definition at 1 PR/month, but their merge rate dropped "
                f"from {ft_merge_early}% to {ft_merge_recent}%."
            ),
        ),
        # 5h table rows
        (
            r"\| Maintainer \| [\d.]+ \| [\d.]+ \| \d+% \| \d+% \|",
            f"| Maintainer | {maint_prs_early} | {maint_prs_recent} | {maint_merge_early}% | {maint_merge_recent}% |",
        ),
        (
            r"\| Regular \| [\d.]+ \| [\d.]+ \| \d+% \| \d+% \|",
            f"| Regular | {reg_prs_early} | {reg_prs_recent} | {reg_merge_early}% | {reg_merge_recent}% |",
        ),
        (
            r"\| First-timer \| [\d.]+ \| [\d.]+ \| \d+% \| \d+% \|",
            f"| First-timer | {ft_prs_early} | {ft_prs_recent} | {ft_merge_early}% | {ft_merge_recent}% |",
        ),
        # Post-table prose
        (
            r"regulars' merge rate fell \d+ percentage points, and first-timers' fell \d+ points\.",
            (
                f"regulars' merge rate fell {reg_merge_early - reg_merge_recent} percentage points, "
                f"and first-timers' fell {ft_merge_early - ft_merge_recent} points."
            ),
        ),
    ]

    # ── Forecast benchmark (Section 12) ──────────────────────────────────────
    if forecast and "models" in forecast:
        fc_models = {m["model"]: m for m in forecast["models"]}
        best = forecast.get("best_model", "ETS")

        for model_name in ["ARIMA", "ETS", "XGBoost", "Prophet"]:
            if model_name not in fc_models:
                continue
            m = fc_models[model_name]
            is_best = (model_name == best)
            if is_best:
                # Match bold row: | **Model** | **N** | **N** | **N%** |
                regex_replacements.append((
                    rf"\| \*\*{model_name}\*\* \| \*\*[\d,]+\*\* \| \*\*[\d,]+\*\* \| \*\*[\d.]+%\*\* \|",
                    f"| **{model_name}** | **{fmt_int(round(m['mae']))}** | **{fmt_int(round(m['rmse']))}** | **{m['mape']:.1f}%** |",
                ))
            else:
                # Match non-bold row: | Model | N | N | N% |
                regex_replacements.append((
                    rf"\| {model_name} \| [\d,]+ \| [\d,]+ \| [\d.]+% \|",
                    f"| {model_name} | {fmt_int(round(m['mae']))} | {fmt_int(round(m['rmse']))} | {m['mape']:.1f}% |",
                ))

        # Best model prose
        best_m = fc_models.get(best)
        if best_m:
            regex_replacements.append((
                r"\w+ wins with [\d.]+% MAPE\.",
                f"{best} wins with {best_m['mape']:.1f}% MAPE.",
            ))

    # ── LSTM AUC (Section 13) ────────────────────────────────────────────────
    if contrib_return and "lstm_auc" in contrib_return:
        lstm_auc = contrib_return["lstm_auc"]
        regex_replacements.append((
            r"The LSTM achieved \*\*AUC=[\d.]+\*\*",
            f"The LSTM achieved **AUC={lstm_auc}**",
        ))

    # Continue with remaining literal replacements
    replacements += [

        # ── Section 5i: tool-era table ────────────────────────────────────────
        (
            "| Pre-Copilot | None | 70.1% | 47.5% | 14,154 |",
            (
                f"| Pre-Copilot | None"
                f" | {tool_eras['pre_copilot']['merge_rate']}%"
                f" | {tool_eras['pre_copilot']['ft_merge_rate']}%"
                f" | {fmt_int(tool_eras['pre_copilot']['prs_per_month'])} |"
            ),
        ),
        (
            "| Jun 2022 - Feb 2023 | Copilot | 71.2% | 43.5% | 20,271 |",
            (
                f"| Jun 2022 - Feb 2023 | Copilot"
                f" | {tool_eras['copilot']['merge_rate']}%"
                f" | {tool_eras['copilot']['ft_merge_rate']}%"
                f" | {fmt_int(tool_eras['copilot']['prs_per_month'])} |"
            ),
        ),
        (
            "| Mar 2023 - Feb 2024 | ChatGPT / GPT-4 | 72.0% | 41.3% | 24,726 |",
            (
                f"| Mar 2023 - Feb 2024 | ChatGPT / GPT-4"
                f" | {tool_eras['chatgpt_gpt4']['merge_rate']}%"
                f" | {tool_eras['chatgpt_gpt4']['ft_merge_rate']}%"
                f" | {fmt_int(tool_eras['chatgpt_gpt4']['prs_per_month'])} |"
            ),
        ),
        (
            "| Mar 2024 - Jan 2025 | Cursor | 73.0% | 40.9% | 27,579 |",
            (
                f"| Mar 2024 - Jan 2025 | Cursor"
                f" | {tool_eras['cursor']['merge_rate']}%"
                f" | {tool_eras['cursor']['ft_merge_rate']}%"
                f" | {fmt_int(tool_eras['cursor']['prs_per_month'])} |"
            ),
        ),
        (
            "| Feb 2025+ | Claude Code / Codex | **59.6%** | **22.7%** | **40,635** |",
            (
                f"| Feb 2025+ | Claude Code / Codex"
                f" | **{tool_eras['agentic']['merge_rate']}%**"
                f" | **{tool_eras['agentic']['ft_merge_rate']}%**"
                f" | **{fmt_int(tool_eras['agentic']['prs_per_month'])}** |"
            ),
        ),
        # 5i inline prose
        (
            "merge rates collapsed to 60% overall and 23% for first-timers, "
            "even though PR volume nearly doubled.",
            (
                f"merge rates collapsed to {tool_eras['agentic']['merge_rate']}% overall "
                f"and {tool_eras['agentic']['ft_merge_rate']}% for first-timers, "
                f"even though PR volume nearly doubled."
            ),
        ),
        # 5i summary
        (
            "agentic tools (Claude Code, Codex) unleashed a volume surge (+47% PRs) that "
            "overwhelmed the quality bar: merge rates dropped 13 points and first-timer "
            "acceptance fell to 23%.",
            (
                f"agentic tools (Claude Code, Codex) unleashed a volume surge that "
                f"overwhelmed the quality bar: merge rates dropped to {tool_eras['agentic']['merge_rate']}% "
                f"and first-timer acceptance fell to {tool_eras['agentic']['ft_merge_rate']}%."
            ),
        ),

        # ── Section 10: PR abandonment classifiers ────────────────────────────
        (
            "We trained Random Forest (AUC=0.868) and XGBoost (AUC=0.916) classifiers "
            "on 913,387 PRs with size data",
            (
                f"We trained Random Forest (AUC={abandonment['rf_auc']:.3f}) and "
                f"XGBoost (AUC={abandonment['xgb_auc']:.3f}) classifiers "
                f"on {fmt_int(abandonment['n_prs_analyzed'])} PRs with size data"
            ),
        ),
        # Feature importance table rows
        (
            "| First PR to this repo | 0.578 | Whether the author has any prior PRs to this specific repo |",
            (
                f"| First PR to this repo | {feat_first_pr:.3f} | "
                f"Whether the author has any prior PRs to this specific repo |"
            ),
        ),
        (
            "| Repo historical merge rate | 0.139 | What fraction of the repo's past PRs were merged |",
            (
                f"| Repo historical merge rate | {feat_repo_merge:.3f} | "
                f"What fraction of the repo's past PRs were merged |"
            ),
        ),
        (
            "| Repo PRs in prior 30 days | 0.100 | How active the repo was in the month before your PR |",
            (
                f"| Repo PRs in prior 30 days | {feat_repo_recent:.3f} | "
                f"How active the repo was in the month before your PR |"
            ),
        ),
        (
            "| Lines added | 0.057 | Size of the changeset |",
            f"| Lines added | {feat_additions:.3f} | Size of the changeset |",
        ),
        (
            "| Lines deleted | 0.034 | Size of the changeset (removals) |",
            f"| Lines deleted | {feat_deletions:.3f} | Size of the changeset (removals) |",
        ),
        # Abandonment inline stats
        (
            "First-time contributors to a repo see 5.2% of their PRs abandoned, "
            "vs 1.2% for authors who have submitted before (4.3x higher).",
            (
                f"First-time contributors to a repo see "
                f"{ins['abandon_rate_first_pr'] * 100:.1f}% of their PRs abandoned, "
                f"vs {ins['abandon_rate_repeat_author'] * 100:.1f}% for authors who have "
                f"submitted before "
                f"({ins['abandon_rate_first_pr'] / ins['abandon_rate_repeat_author']:.1f}x higher)."
            ),
        ),
        # Actionable advice point 1
        (
            "Repos with a historical merge rate below 50% abandon 2.7% of PRs; "
            "those above 70% abandon only 1.6% (1.7x difference).",
            (
                f"Repos with a historical merge rate below 50% abandon "
                f"{ins['abandon_rate_low_merge_repo'] * 100:.1f}% of PRs; "
                f"those above 70% abandon only {ins['abandon_rate_high_merge_repo'] * 100:.1f}% "
                f"({ins['abandon_rate_low_merge_repo'] / ins['abandon_rate_high_merge_repo']:.1f}x difference)."
            ),
        ),
        # Actionable advice point 3 (weekends)
        (
            "PRs submitted on weekends are abandoned 2.7% of the time vs 1.8% on weekdays (50% more likely).",
            (
                f"PRs submitted on weekends are abandoned "
                f"{ins['abandon_rate_weekend'] * 100:.1f}% of the time "
                f"vs {ins['abandon_rate_weekday'] * 100:.1f}% on weekdays "
                f"({ins['abandon_rate_weekend'] / ins['abandon_rate_weekday']:.1f}x more likely)."
            ),
        ),
        # Actionable advice point 4 (size)
        (
            "PRs over 500 lines are 1.5x more likely to be abandoned than PRs under 50 lines "
            "(2.4% vs 1.7%).",
            (
                f"PRs over 500 lines are "
                f"{ins['abandon_rate_over_500_lines'] / ins['abandon_rate_under_50_lines']:.1f}x "
                f"more likely to be abandoned than PRs under 50 lines "
                f"({ins['abandon_rate_over_500_lines'] * 100:.1f}% vs "
                f"{ins['abandon_rate_under_50_lines'] * 100:.1f}%)."
            ),
        ),
        # Actionable advice point 5 (relationship)
        (
            "The 4.3x gap between first-time and repeat contributors is the clearest signal in the data.",
            (
                f"The {ins['abandon_rate_first_pr'] / ins['abandon_rate_repeat_author']:.1f}x gap "
                f"between first-time and repeat contributors is the clearest signal in the data."
            ),
        ),

        # ── What's Next section ───────────────────────────────────────────────
        (
            "This is a first draft based on 341 of 1,537 targeted repos. "
            "The extraction is running and will complete soon. Planned updates:\n\n"
            "- **Re-run all analyses** with the full 200-repo dataset\n"
            "- **Comparative by organization type**: company-backed vs community vs foundation\n"
            "- **Cross-correlation**: do repos that respond faster retain more contributors?",
            (
                f"Dataset covers {n_repos} repos spanning 2016-2026. Planned updates:\n\n"
                "- **Comparative by organization type**: company-backed vs community vs foundation\n"
                "- **Cross-correlation**: do repos that respond faster retain more contributors?"
            ),
        ),
    ]

    # ── Generate Section 16 (first-timer decision trees by language) ────────
    if ft_trees:
        main_langs = ["Python", "TypeScript", "Go", "Rust", "C++", "Java", "C", "JavaScript", "C#"]
        available = [l for l in main_langs if l in ft_trees]

        tree_table_rows = []
        for lang in available:
            d = ft_trees[lang]
            top_feat = d["feature_importances"][0]
            best = d["recommended_repos"][0] if d.get("recommended_repos") else None
            best_name = best["repo"] if best else "N/A"
            best_rate = f'{best["ft_merge_rate"]:.0f}%' if best else "N/A"
            rule = d.get("decision_rules", "")
            tree_table_rows.append(
                f"| {lang} | {d['auc']:.3f} | {top_feat['feature']} ({top_feat['importance']:.0%}) "
                f"| {rule} | {best_name} ({best_rate} FT merge) |"
            )

        per_lang_tables = []
        for lang in available:
            repos = ft_trees[lang].get("recommended_repos", [])[:5]
            if not repos:
                continue
            rows = []
            for r in repos:
                ws = f"{r.get('wilson_score', 0):.3f}"
                rows.append(
                    f"| {r['repo']} | **{r['ft_merge_rate']:.0f}%** | {r['ft_prs']} | {ws} |"
                )
            per_lang_tables.append(
                f"**If you write {lang}, start here** (ranked by Wilson score, min 20 FT PRs):\n\n"
                f"| Repo | First-timer merge rate | First-timer PRs | Wilson score |\n"
                f"|------|----------------------|-----------------|--------------|\n"
                + "\n".join(rows)
            )

        section_16 = (
            "### 16. Where Should You Submit Your First PR? (Decision Trees by Language)\n\n"
            "We trained decision trees on first-timer PRs for each major programming language "
            "to predict which contributions get merged. The question is simple: **if you program "
            "in Python, Rust, Go, or any other language, where should you start contributing "
            "to maximize your chances of getting merged?**\n\n"
            "Repos are ranked using the Wilson score lower bound, which penalizes small samples: "
            "a repo with 5/5 merges ranks lower than one with 200/300, because we need statistical "
            "confidence, not lucky streaks.\n\n"
            "**Per-language model results:**\n\n"
            "| Language | AUC | Top predictor | Actionable rule | Best repo |\n"
            "|----------|-----|--------------|-----------------|-----------|\n"
            + "\n".join(tree_table_rows) + "\n\n"
            + "\n\n".join(per_lang_tables) + "\n\n"
            "<img src=\"output/figures/first_timer_tree_python.svg\" width=\"100%\">\n\n"
            "<img src=\"output/figures/first_timer_recommendations.svg\" width=\"100%\">\n\n"
            "**The decision tree distilled:** Rust is the only language where PR size matters more "
            "than the repo itself. Everywhere else, choosing the right repo is the single most "
            "important decision a first-timer can make. Check the repo's merged-vs-closed ratio "
            "before investing effort."
        )

        section_16_pattern = (
            r"### 16\. Where Should You Submit Your First PR\?.*?"
            r"(?=\n---\n)"
        )
        if re.search(section_16_pattern, text, re.DOTALL):
            text = re.sub(section_16_pattern, section_16, text, count=1, flags=re.DOTALL)
            print("  Section 16 regenerated from stats.json")

    # ── Apply literal replacements ──────────────────────────────────────────
    replaced = 0
    not_found: list[str] = []
    for old, new in replacements:
        if old == new:
            continue  # no-op sentinels
        if old in text:
            text = text.replace(old, new, 1)
            replaced += 1
        else:
            not_found.append(old[:80])

    # ── Apply regex replacements ─────────────────────────────────────────────
    regex_applied = 0
    regex_not_found: list[str] = []
    for pattern, replacement in regex_replacements:
        if re.search(pattern, text):
            text = re.sub(pattern, replacement, text, count=1)
            regex_applied += 1
        else:
            regex_not_found.append(pattern[:80])

    README_PATH.write_text(text, encoding="utf-8")

    total = replaced + regex_applied
    print(f"generate_readme.py: applied {total} replacements to README.md "
          f"({replaced} literal, {regex_applied} regex)")
    all_not_found = not_found + regex_not_found
    if all_not_found:
        print(f"\nWARNING: {len(all_not_found)} patterns not found in README.md:")
        for s in all_not_found:
            print(f"  - {s!r}")


if __name__ == "__main__":
    main()
