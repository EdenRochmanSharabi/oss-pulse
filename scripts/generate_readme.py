#!/usr/bin/env python3
"""
generate_readme.py

Reads data/processed/stats.json and updates README.md with current numbers.
Only numbers change; all prose and narrative structure stays intact.

All replacements use regex patterns that match ANY number in each position,
so the script is idempotent regardless of what numbers are currently in the
README.

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


# ── Formatting helpers ──────────────────────────────────────────────────────

def fmt_int(n: int | float) -> str:
    """Format integer with comma thousands separator."""
    return f"{int(n):,}"


def fmt_pct(v: float, decimals: int = 1) -> str:
    """Format a float as a percentage string with % sign."""
    return f"{v:.{decimals}f}%"


# ── Number-matching regex fragments ────────────────────────────────────────

NUM = r"[\d,]+"           # matches "329,142" or "4048297"
FLOAT = r"[\d,]+\.?\d*"   # matches "2.82" or "3" or "329,142"
PCT = r"[\d.]+%"          # matches "67.6%" or "34.8%"


def load_stats() -> dict:
    with STATS_PATH.open() as f:
        return json.load(f)


def regex_replace(text: str, pattern: str, replacement: str,
                  label: str = "") -> tuple[str, bool]:
    """Apply a single regex replacement. Returns (new_text, matched)."""
    if re.search(pattern, text, re.DOTALL):
        text = re.sub(pattern, replacement, text, count=1, flags=re.DOTALL)
        return text, True
    return text, False


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

    # ── Derived values ──────────────────────────────────────────────────────

    total_prs = ds["total_prs"]
    n_repos = ds["n_repos"]
    n_contributors = ds["n_contributors"]

    def rejection(era_key: str) -> float:
        return round(100.0 - eras[era_key]["merge_rate"], 1)

    def prs_per_author(era_key: str) -> float:
        e = eras[era_key]
        return round(e["prs_per_month"] / e["contributors_per_month"], 2)

    # ── Author group values from productivity_by_author_type ────────────────
    maint_pre = prod["maintainer"]["pre_2023"]
    maint_post = prod["maintainer"]["post_2023"]
    reg_pre = prod["regular"]["pre_2023"]
    reg_post = prod["regular"]["post_2023"]
    ft_pre = prod["first-timer"]["pre_2023"]
    ft_post = prod["first-timer"]["post_2023"]

    maint_prs_early = maint_pre["prs_per_person_month"]
    maint_prs_recent = maint_post["prs_per_person_month"]
    maint_merge_early = int(maint_pre["merge_rate_pct"])
    maint_merge_recent = int(maint_post["merge_rate_pct"])

    reg_prs_early = reg_pre["prs_per_person_month"]
    reg_prs_recent = reg_post["prs_per_person_month"]
    reg_merge_early = int(reg_pre["merge_rate_pct"])
    reg_merge_recent = int(reg_post["merge_rate_pct"])

    ft_prs_early = ft_pre["prs_per_person_month"]
    ft_prs_recent = ft_post["prs_per_person_month"]
    ft_merge_early = int(ft_pre["merge_rate_pct"])
    ft_merge_recent = int(ft_post["merge_rate_pct"])

    # Funnel derived
    second_pr_count = funnel["2nd_pr"]
    fifth_pr_count = funnel["5th_pr"]
    regular_count = funnel["regular"]
    dropout_pct = funnel["dropout_1st_pct"]

    # ── Abandonment feature importances ──────────────────────────────────────
    top_feats = {f["feature"]: f["importance"] for f in abandonment["top_features"]}
    feat_first_pr = top_feats.get("is_first_pr_to_repo", 0.573)
    feat_repo_merge = top_feats.get("repo_merge_rate", 0.139)
    feat_repo_recent = top_feats.get("repo_recent_prs", 0.100)
    feat_additions = top_feats.get("additions", 0.057)
    feat_deletions = top_feats.get("deletions", 0.034)
    ins = abandonment["insights"]

    # Compute pct-point drops for author groups
    reg_merge_drop = reg_merge_early - reg_merge_recent
    ft_merge_drop = ft_merge_early - ft_merge_recent

    # Compute productivity gain pct
    prs_first = prs_per_author('2016_2019')
    prs_last = prs_per_author('2025_agentic')
    productivity_gain_pct = round((prs_last / prs_first - 1) * 100)

    # ── Build replacement list: (pattern, replacement, label) ───────────────
    # Each pattern uses regex to match ANY number in each position.
    replacements: list[tuple[str, str, str]] = []

    # ── A. Header / intro ───────────────────────────────────────────────────

    # Title line: "X.X million Pull Requests across NNN top open-source..."
    replacements.append((
        r"\*\*A time-series study of [\d.]+ million Pull Requests across " + NUM + r" top open-source software projects",
        f"**A time-series study of {total_prs / 1_000_000:.1f} million Pull Requests across {n_repos} top open-source software projects",
        "title line",
    ))

    # Introduction paragraph: "We analyzed **N PRs** across **N software repos**"
    replacements.append((
        r"We analyzed \*\*" + NUM + r" Pull Requests\*\* across \*\*" + NUM + r" software repositories\*\*",
        f"We analyzed **{fmt_int(total_prs)} Pull Requests** across **{n_repos} software repositories**",
        "intro paragraph",
    ))

    # Methodology: "N software repos were collected for analysis"
    # Matches "N repos were collected" or "N had sufficient PR data"
    replacements.append((
        r"Of these, " + NUM + r" software repos were collected for analysis",
        f"Of these, {n_repos} software repos were collected for analysis",
        "methodology repo count",
    ))

    # ── Section header: "General Findings (N software repos)" ────────────────
    replacements.append((
        r"## General Findings \(" + NUM + r" software repos\)",
        f"## General Findings ({n_repos} software repos)",
        "section header",
    ))

    # ── Dataset at a Glance ───────────────────────────────────────────────

    # "- **N PRs** across **N software repos**, spanning **2016-2026**"
    replacements.append((
        r"- \*\*" + NUM + r" PRs\*\* across \*\*" + NUM + r" software repos\*\*, spanning \*\*2016-2026\*\*",
        f"- **{fmt_int(total_prs)} PRs** across **{n_repos} software repos**, spanning **2016-2026**",
        "dataset glance PRs",
    ))

    # "- **N unique contributors** (excluding bots)"
    replacements.append((
        r"- \*\*" + NUM + r" unique contributors\*\* \(excluding bots\)",
        f"- **{fmt_int(n_contributors)} unique contributors** (excluding bots)",
        "dataset glance contributors",
    ))

    # "- Outcome distribution: **N% merged**, N% closed, N% abandoned, N% still open"
    replacements.append((
        r"- Outcome distribution: \*\*" + PCT + r" merged\*\*, " + PCT + r" closed, " + PCT + r" abandoned, " + PCT + r" still open",
        (
            f"- Outcome distribution: **{outcomes['merge_rate']}% merged**, "
            f"{outcomes['closed_rate']}% closed, "
            f"{outcomes['abandoned_rate']}% abandoned, "
            f"{outcomes['open_rate']}% still open"
        ),
        "outcome distribution",
    ))

    # ── Section 4: Retention Crisis ──────────────────────────────────────

    # "Of **N contributors** who opened at least one PR:"
    replacements.append((
        r"Of \*\*" + NUM + r" contributors\*\* who opened at least one PR:",
        f"Of **{fmt_int(n_contributors)} contributors** who opened at least one PR:",
        "funnel header",
    ))

    # "- **N%** came back for a second (N)"
    replacements.append((
        r"- \*\*" + PCT + r"\*\* came back for a second \(" + NUM + r"\)",
        f"- **{funnel['2nd_pr_pct']}%** came back for a second ({fmt_int(second_pr_count)})",
        "funnel 2nd pr",
    ))

    # "- **N%** reached their 5th PR (N)"
    replacements.append((
        r"- \*\*" + PCT + r"\*\* reached their 5th PR \(" + NUM + r"\)",
        f"- **{funnel['5th_pr_pct']}%** reached their 5th PR ({fmt_int(fifth_pr_count)})",
        "funnel 5th pr",
    ))

    # "- **N%** became regulars with 20+ PRs (N)"
    replacements.append((
        r"- \*\*" + PCT + r"\*\* became regulars with 20\+ PRs \(" + NUM + r"\)",
        f"- **{funnel['regular_pct']}%** became regulars with 20+ PRs ({fmt_int(regular_count)})",
        "funnel regulars",
    ))

    # "**N% of first-time contributors never return.**"
    replacements.append((
        r"\*\*" + r"[\d.]+" + r"% of first-time contributors never return\.\*\*",
        f"**{dropout_pct}% of first-time contributors never return.**",
        "dropout pct",
    ))

    # ── B. AI Era table (Section 5) ──────────────────────────────────────
    # Match each row by its label, then 6 number cells (last one bold)

    era_keys = ['2016_2019', '2020_2021', '2022_copilot', '2023_chatgpt', '2024_cursor', '2025_agentic']

    # Contributors/month row
    replacements.append((
        r"\| Contributors/month \|" + (r" " + NUM + r" \|") * 5 + r" \*\*" + NUM + r"\*\* \|",
        (
            "| Contributors/month"
            + "".join(f" | {fmt_int(eras[k]['contributors_per_month'])}" for k in era_keys[:-1])
            + f" | **{fmt_int(eras[era_keys[-1]]['contributors_per_month'])}** |"
        ),
        "era table contributors/month",
    ))

    # First-timers/month row
    replacements.append((
        r"\| First-timers/month \|" + (r" " + NUM + r" \|") * 5 + r" \*\*" + NUM + r"\*\* \|",
        (
            "| First-timers/month"
            + "".join(f" | {fmt_int(eras[k]['firsttimers_per_month'])}" for k in era_keys[:-1])
            + f" | **{fmt_int(eras[era_keys[-1]]['firsttimers_per_month'])}** |"
        ),
        "era table first-timers/month",
    ))

    # First-timer rejection row
    replacements.append((
        r"\| First-timer rejection \|" + (r" " + PCT + r" \|") * 5 + r" \*\*" + PCT + r"\*\* \|",
        (
            "| First-timer rejection"
            + "".join(f" | {eras[k]['ft_rejection_rate']}%" for k in era_keys[:-1])
            + f" | **{eras[era_keys[-1]]['ft_rejection_rate']}%** |"
        ),
        "era table ft rejection",
    ))

    # Overall rejection row
    replacements.append((
        r"\| Overall rejection \|" + (r" " + PCT + r" \|") * 5 + r" \*\*" + PCT + r"\*\* \|",
        (
            "| Overall rejection"
            + "".join(f" | {rejection(k)}%" for k in era_keys[:-1])
            + f" | **{rejection(era_keys[-1])}%** |"
        ),
        "era table overall rejection",
    ))

    # PRs/author/month row
    replacements.append((
        r"\| PRs/author/month \|" + (r" " + FLOAT + r" \|") * 5 + r" \*\*" + FLOAT + r"\*\* \|",
        (
            "| PRs/author/month"
            + "".join(f" | {prs_per_author(k)}" for k in era_keys[:-1])
            + f" | **{prs_per_author(era_keys[-1])}** |"
        ),
        "era table prs/author/month",
    ))

    # ── Section 5b: contributor/first-timer inline ───────────────────────

    # "Monthly unique contributors grew from N (2016-2019) to N (2025-2026). First-timers per month nearly tripled: N to N."
    replacements.append((
        r"Monthly unique contributors grew from " + NUM + r" \(2016-2019\) to " + NUM + r" \(2025-2026\)\."
        r" First-timers per month (?:nearly )?tripled: " + NUM + r" to " + NUM + r"\.",
        (
            f"Monthly unique contributors grew from {fmt_int(eras['2016_2019']['contributors_per_month'])} "
            f"(2016-2019) to {fmt_int(eras['2025_agentic']['contributors_per_month'])} (2025-2026). "
            f"First-timers per month nearly tripled: "
            f"{fmt_int(eras['2016_2019']['firsttimers_per_month'])} to "
            f"{fmt_int(eras['2025_agentic']['firsttimers_per_month'])}."
        ),
        "5b inline",
    ))

    # ── Section 5c: rejection rate inline ────────────────────────────────

    # "From 2020 to 2024, the overall rejection rate was stable around N-N%. But in 2025-2026, it jumped to N%."
    replacements.append((
        r"overall rejection rate was stable around " + FLOAT + r"-" + PCT
        + r"\. But in 2025-2026, it jumped to " + PCT + r"\.",
        (
            f"overall rejection rate was stable around "
            f"{rejection('2020_2021')}-{rejection('2022_copilot')}%. "
            f"But in 2025-2026, it jumped to {rejection('2025_agentic')}%."
        ),
        "5c inline",
    ))

    # ── Section 5e: first-timer rejection inline ─────────────────────────

    # "N first-timers/month in 2025-2026, up from N in 2016-2019. But their rejection rate climbed steadily: N% (2016-2019) to N% (2022) to N% (2023) to **N%** (2025-2026)."
    replacements.append((
        NUM + r" first-timers/month in 2025-2026, up from " + NUM + r" in 2016-2019\."
        r" But their rejection rate climbed steadily: " + PCT + r" \(2016-2019\)"
        r" to " + PCT + r" \(2022\)"
        r" to " + PCT + r" \(2023\)"
        r" to \*\*" + PCT + r"\*\* \(2025-2026\)\.",
        (
            f"{fmt_int(eras['2025_agentic']['firsttimers_per_month'])} first-timers/month in 2025-2026, "
            f"up from {fmt_int(eras['2016_2019']['firsttimers_per_month'])} in 2016-2019. "
            f"But their rejection rate climbed steadily: "
            f"{eras['2016_2019']['ft_rejection_rate']}% (2016-2019) "
            f"to {eras['2022_copilot']['ft_rejection_rate']}% (2022) "
            f"to {eras['2023_chatgpt']['ft_rejection_rate']}% (2023) "
            f"to **{eras['2025_agentic']['ft_rejection_rate']}%** (2025-2026)."
        ),
        "5e inline",
    ))

    # ── Section 5g: individual productivity inline ───────────────────────

    # "Individual productivity is up N%"
    replacements.append((
        r"Individual productivity is up \d+%",
        f"Individual productivity is up {productivity_gain_pct}%",
        "5g headline pct",
    ))

    # "Each contributor produces more PRs per month (N to N)."
    replacements.append((
        r"Each contributor produces more PRs per month \(" + FLOAT + r" to " + FLOAT + r"\)\.",
        f"Each contributor produces more PRs per month ({prs_first} to {prs_last}).",
        "5g inline",
    ))

    # ── Section 5h: author group prose ───────────────────────────────────

    # "The aggregate +N% masks a stark inequality."
    replacements.append((
        r"The aggregate \+\d+% masks a stark inequality\.",
        f"The aggregate +{productivity_gain_pct}% masks a stark inequality.",
        "5h aggregate pct",
    ))

    # "Maintainers went from N to N PRs/person/month while their merge rate declined from N% to N%. Regulars gained modestly (N to N) but their merge rate collapsed from N% to N%. First-timers are by definition at 1 PR/month, but their merge rate dropped from N% to N%."
    replacements.append((
        r"Maintainers went from " + FLOAT + r" to " + FLOAT + r" PRs/person/month while their merge rate "
        r"declined from \d+% to \d+%\. Regulars gained modestly \(" + FLOAT + r" to " + FLOAT + r"\) but their "
        r"merge rate collapsed from \d+% to \d+%\. First-timers are by definition at "
        r"1 PR/month, but their merge rate dropped from \d+% to \d+%\.",
        (
            f"Maintainers went from {maint_prs_early} to {maint_prs_recent} PRs/person/month "
            f"while their merge rate declined from {maint_merge_early}% to {maint_merge_recent}%. "
            f"Regulars gained modestly ({reg_prs_early} to {reg_prs_recent}) but their "
            f"merge rate collapsed from {reg_merge_early}% to {reg_merge_recent}%. "
            f"First-timers are by definition at 1 PR/month, but their merge rate dropped "
            f"from {ft_merge_early}% to {ft_merge_recent}%."
        ),
        "5h prose",
    ))

    # ── C. Author group table (Section 5h) ──────────────────────────────

    # Maintainer row
    replacements.append((
        r"\| Maintainer \| " + FLOAT + r" \| " + FLOAT + r" \| \d+% \| \d+% \|",
        f"| Maintainer | {maint_prs_early} | {maint_prs_recent} | {maint_merge_early}% | {maint_merge_recent}% |",
        "5h table maintainer",
    ))

    # Regular row
    replacements.append((
        r"\| Regular \| " + FLOAT + r" \| " + FLOAT + r" \| \d+% \| \d+% \|",
        f"| Regular | {reg_prs_early} | {reg_prs_recent} | {reg_merge_early}% | {reg_merge_recent}% |",
        "5h table regular",
    ))

    # First-timer row
    replacements.append((
        r"\| First-timer \| " + FLOAT + r" \| " + FLOAT + r" \| \d+% \| \d+% \|",
        f"| First-timer | {ft_prs_early} | {ft_prs_recent} | {ft_merge_early}% | {ft_merge_recent}% |",
        "5h table first-timer",
    ))

    # Post-table prose: "regulars' merge rate fell N percentage points, and first-timers' fell N points."
    replacements.append((
        r"regulars' merge rate fell \d+ percentage points, and first-timers' fell \d+ points\.",
        f"regulars' merge rate fell {reg_merge_drop} percentage points, and first-timers' fell {ft_merge_drop} points.",
        "5h post-table prose",
    ))

    # ── D. Tool-era table (Section 5i) ──────────────────────────────────

    # Pre-Copilot row
    replacements.append((
        r"\| Pre-Copilot \| None \| " + PCT + r" \| " + PCT + r" \| " + NUM + r" \|",
        (
            f"| Pre-Copilot | None"
            f" | {tool_eras['pre_copilot']['merge_rate']}%"
            f" | {tool_eras['pre_copilot']['ft_merge_rate']}%"
            f" | {fmt_int(tool_eras['pre_copilot']['prs_per_month'])} |"
        ),
        "tool-era pre-copilot",
    ))

    # Jun 2022 - Feb 2023 | Copilot row
    replacements.append((
        r"\| Jun 2022 - Feb 2023 \| Copilot \| " + PCT + r" \| " + PCT + r" \| " + NUM + r" \|",
        (
            f"| Jun 2022 - Feb 2023 | Copilot"
            f" | {tool_eras['copilot']['merge_rate']}%"
            f" | {tool_eras['copilot']['ft_merge_rate']}%"
            f" | {fmt_int(tool_eras['copilot']['prs_per_month'])} |"
        ),
        "tool-era copilot",
    ))

    # Mar 2023 - Feb 2024 | ChatGPT / GPT-4 row
    replacements.append((
        r"\| Mar 2023 - Feb 2024 \| ChatGPT / GPT-4 \| " + PCT + r" \| " + PCT + r" \| " + NUM + r" \|",
        (
            f"| Mar 2023 - Feb 2024 | ChatGPT / GPT-4"
            f" | {tool_eras['chatgpt_gpt4']['merge_rate']}%"
            f" | {tool_eras['chatgpt_gpt4']['ft_merge_rate']}%"
            f" | {fmt_int(tool_eras['chatgpt_gpt4']['prs_per_month'])} |"
        ),
        "tool-era chatgpt",
    ))

    # Mar 2024 - Jan 2025 | Cursor row
    replacements.append((
        r"\| Mar 2024 - Jan 2025 \| Cursor \| " + PCT + r" \| " + PCT + r" \| " + NUM + r" \|",
        (
            f"| Mar 2024 - Jan 2025 | Cursor"
            f" | {tool_eras['cursor']['merge_rate']}%"
            f" | {tool_eras['cursor']['ft_merge_rate']}%"
            f" | {fmt_int(tool_eras['cursor']['prs_per_month'])} |"
        ),
        "tool-era cursor",
    ))

    # Feb 2025+ | Claude Code / Codex row (bold values)
    replacements.append((
        r"\| Feb 2025\+ \| Claude Code / Codex \| \*\*" + PCT + r"\*\* \| \*\*" + PCT + r"\*\* \| \*\*" + NUM + r"\*\* \|",
        (
            f"| Feb 2025+ | Claude Code / Codex"
            f" | **{tool_eras['agentic']['merge_rate']}%**"
            f" | **{tool_eras['agentic']['ft_merge_rate']}%**"
            f" | **{fmt_int(tool_eras['agentic']['prs_per_month'])}** |"
        ),
        "tool-era agentic",
    ))

    # 5i inline prose: "merge rates collapsed to N% overall and N% for first-timers"
    replacements.append((
        r"merge rates collapsed to " + PCT + r" overall and " + PCT + r" for first-timers",
        (
            f"merge rates collapsed to {tool_eras['agentic']['merge_rate']}% overall "
            f"and {tool_eras['agentic']['ft_merge_rate']}% for first-timers"
        ),
        "5i inline merge collapse",
    ))

    # 5i summary: "merge rates dropped to N% and first-timer acceptance fell to N%."
    replacements.append((
        r"merge rates dropped to " + PCT + r" and first-timer acceptance fell to " + PCT + r"\.",
        (
            f"merge rates dropped to {tool_eras['agentic']['merge_rate']}% "
            f"and first-timer acceptance fell to {tool_eras['agentic']['ft_merge_rate']}%."
        ),
        "5i summary",
    ))

    # ── E. PR Abandonment (Section 10) ──────────────────────────────────

    # "We trained Random Forest (AUC=N) and XGBoost (AUC=N) classifiers on N PRs with size data"
    replacements.append((
        r"We trained Random Forest \(AUC=" + FLOAT + r"\) and XGBoost \(AUC=" + FLOAT + r"\) classifiers"
        r" on " + NUM + r" PRs with size data",
        (
            f"We trained Random Forest (AUC={abandonment['rf_auc']:.3f}) and "
            f"XGBoost (AUC={abandonment['xgb_auc']:.3f}) classifiers "
            f"on {fmt_int(abandonment['n_prs_analyzed'])} PRs with size data"
        ),
        "abandonment classifiers",
    ))

    # Feature importance table rows
    replacements.append((
        r"\| First PR to this repo \| " + FLOAT + r" \| Whether the author has any prior PRs to this specific repo \|",
        f"| First PR to this repo | {feat_first_pr:.3f} | Whether the author has any prior PRs to this specific repo |",
        "feat first_pr",
    ))
    replacements.append((
        r"\| Repo historical merge rate \| " + FLOAT + r" \| What fraction of the repo's past PRs were merged \|",
        f"| Repo historical merge rate | {feat_repo_merge:.3f} | What fraction of the repo's past PRs were merged |",
        "feat repo_merge",
    ))
    replacements.append((
        r"\| Repo PRs in prior 30 days \| " + FLOAT + r" \| How active the repo was in the month before your PR \|",
        f"| Repo PRs in prior 30 days | {feat_repo_recent:.3f} | How active the repo was in the month before your PR |",
        "feat repo_recent",
    ))
    replacements.append((
        r"\| Lines added \| " + FLOAT + r" \| Size of the changeset \|",
        f"| Lines added | {feat_additions:.3f} | Size of the changeset |",
        "feat additions",
    ))
    replacements.append((
        r"\| Lines deleted \| " + FLOAT + r" \| Size of the changeset \(removals\) \|",
        f"| Lines deleted | {feat_deletions:.3f} | Size of the changeset (removals) |",
        "feat deletions",
    ))

    # Abandonment inline stats: first-time vs repeat
    replacements.append((
        r"First-time contributors to a repo see " + PCT + r" of their PRs abandoned,"
        r" vs " + PCT + r" for authors who have submitted before \(" + FLOAT + r"x higher\)\.",
        (
            f"First-time contributors to a repo see "
            f"{ins['abandon_rate_first_pr'] * 100:.1f}% of their PRs abandoned, "
            f"vs {ins['abandon_rate_repeat_author'] * 100:.1f}% for authors who have "
            f"submitted before "
            f"({ins['abandon_rate_first_pr'] / ins['abandon_rate_repeat_author']:.1f}x higher)."
        ),
        "abandon first vs repeat",
    ))

    # Advice point 1: low vs high merge repo
    replacements.append((
        r"Repos with a historical merge rate below 50% abandon " + PCT + r" of PRs;"
        r" those above 70% abandon only " + PCT + r" \(" + FLOAT + r"x difference\)\.",
        (
            f"Repos with a historical merge rate below 50% abandon "
            f"{ins['abandon_rate_low_merge_repo'] * 100:.1f}% of PRs; "
            f"those above 70% abandon only {ins['abandon_rate_high_merge_repo'] * 100:.1f}% "
            f"({ins['abandon_rate_low_merge_repo'] / ins['abandon_rate_high_merge_repo']:.1f}x difference)."
        ),
        "abandon low vs high merge",
    ))

    # Advice point 3: weekends
    replacements.append((
        r"PRs submitted on weekends are abandoned " + PCT + r" of the time"
        r" vs " + PCT + r" on weekdays \(" + FLOAT + r"x more likely\)\.",
        (
            f"PRs submitted on weekends are abandoned "
            f"{ins['abandon_rate_weekend'] * 100:.1f}% of the time "
            f"vs {ins['abandon_rate_weekday'] * 100:.1f}% on weekdays "
            f"({ins['abandon_rate_weekend'] / ins['abandon_rate_weekday']:.1f}x more likely)."
        ),
        "abandon weekends",
    ))

    # Advice point 4: size
    replacements.append((
        r"PRs over 500 lines are " + FLOAT + r"x more likely to be abandoned than PRs under 50 lines"
        r" \(" + PCT + r" vs " + PCT + r"\)\.",
        (
            f"PRs over 500 lines are "
            f"{ins['abandon_rate_over_500_lines'] / ins['abandon_rate_under_50_lines']:.1f}x "
            f"more likely to be abandoned than PRs under 50 lines "
            f"({ins['abandon_rate_over_500_lines'] * 100:.1f}% vs "
            f"{ins['abandon_rate_under_50_lines'] * 100:.1f}%)."
        ),
        "abandon size",
    ))

    # Advice point 5: relationship gap
    replacements.append((
        r"The " + FLOAT + r"x gap between first-time and repeat contributors is the clearest signal in the data\.",
        (
            f"The {ins['abandon_rate_first_pr'] / ins['abandon_rate_repeat_author']:.1f}x gap "
            f"between first-time and repeat contributors is the clearest signal in the data."
        ),
        "abandon gap",
    ))

    # ── F. Forecast benchmark table (Section 12) ────────────────────────

    if forecast and "models" in forecast:
        fc_models = {m["model"]: m for m in forecast["models"]}
        best = forecast.get("best_model", "ETS")

        for model_name in ["ARIMA", "ETS", "XGBoost", "Prophet"]:
            if model_name not in fc_models:
                continue
            m = fc_models[model_name]
            is_best = (model_name == best)
            if is_best:
                replacements.append((
                    rf"\| \*\*{model_name}\*\* \| \*\*" + NUM + r"\*\* \| \*\*" + NUM + r"\*\* \| \*\*" + PCT + r"\*\* \|",
                    f"| **{model_name}** | **{fmt_int(round(m['mae']))}** | **{fmt_int(round(m['rmse']))}** | **{m['mape']:.1f}%** |",
                    f"forecast {model_name} (best)",
                ))
            else:
                replacements.append((
                    rf"\| {model_name} \| " + NUM + r" \| " + NUM + r" \| " + PCT + r" \|",
                    f"| {model_name} | {fmt_int(round(m['mae']))} | {fmt_int(round(m['rmse']))} | {m['mape']:.1f}% |",
                    f"forecast {model_name}",
                ))

        # Best model prose
        best_m = fc_models.get(best)
        if best_m:
            replacements.append((
                r"\w+ wins with " + PCT + r" MAPE\.",
                f"{best} wins with {best_m['mape']:.1f}% MAPE.",
                "forecast best prose",
            ))

    # ── I. LSTM AUC (Section 13) ────────────────────────────────────────

    if contrib_return and "lstm_auc" in contrib_return:
        lstm_auc = contrib_return["lstm_auc"]
        replacements.append((
            r"The LSTM achieved \*\*AUC=" + FLOAT + r"\*\*",
            f"The LSTM achieved **AUC={lstm_auc}**",
            "LSTM AUC",
        ))

    # Also update the cross-reference "single PR will be abandoned (AUC=N)"
    replacements.append((
        r"whether a \*single PR\* will be abandoned \(AUC=" + FLOAT + r"\)",
        f"whether a *single PR* will be abandoned (AUC={abandonment['xgb_auc']:.3f})",
        "section 13 cross-ref AUC",
    ))

    # ── G. What's Next section ──────────────────────────────────────────

    replacements.append((
        r"Dataset covers " + NUM + r" repos spanning 2016-2026\.",
        f"Dataset covers {n_repos} repos spanning 2016-2026.",
        "whats next",
    ))

    # ── H. Generate Section 16 (first-timer decision trees) ─────────────
    if ft_trees:
        main_langs = ["Python", "TypeScript", "Go", "Rust", "C++", "Java", "C", "JavaScript", "C#"]
        available = [lang for lang in main_langs if lang in ft_trees]

        tree_table_rows = []
        for lang in available:
            d = ft_trees[lang]
            top_feat = d["feature_importances"][0]
            best_repo = d["recommended_repos"][0] if d.get("recommended_repos") else None
            best_name = best_repo["repo"] if best_repo else "N/A"
            best_rate = f'{best_repo["ft_merge_rate"]:.0f}%' if best_repo else "N/A"
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

    # ── Apply all regex replacements ────────────────────────────────────

    applied = 0
    not_found: list[str] = []
    for pattern, replacement, label in replacements:
        text, matched = regex_replace(text, pattern, replacement, label)
        if matched:
            applied += 1
        else:
            not_found.append(label or pattern[:80])

    README_PATH.write_text(text, encoding="utf-8")

    print(f"generate_readme.py: applied {applied} regex replacements to README.md")
    if not_found:
        print(f"\nWARNING: {len(not_found)} patterns not found in README.md:")
        for s in not_found:
            print(f"  - {s!r}")


if __name__ == "__main__":
    main()
