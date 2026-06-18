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
    if re.search(pattern, text):
        text = re.sub(pattern, replacement, text, count=1)
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
    counterfactual = stats.get("counterfactual", {})
    decline = stats.get("project_decline", {})
    ga = stats.get("ga_weights", {})
    author_dist = stats.get("author_distribution", {})
    growth = stats.get("growth", {})
    weekend_rate = stats.get("weekend_rate", None)
    era_pr_size = stats.get("era_pr_size", {})
    era_prs_per_author = stats.get("era_prs_per_author", {})

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
    maint_merge_early = round(maint_pre["merge_rate_pct"])
    maint_merge_recent = round(maint_post["merge_rate_pct"])

    reg_prs_early = reg_pre["prs_per_person_month"]
    reg_prs_recent = reg_post["prs_per_person_month"]
    reg_merge_early = round(reg_pre["merge_rate_pct"])
    reg_merge_recent = round(reg_post["merge_rate_pct"])

    ft_prs_early = ft_pre["prs_per_person_month"]
    ft_prs_recent = ft_post["prs_per_person_month"]
    ft_merge_early = round(ft_pre["merge_rate_pct"])
    ft_merge_recent = round(ft_post["merge_rate_pct"])

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

    # "- Author distribution: 72% maintainers, 15% regulars, 7% first-timers, 6% bots (by PR count)"
    if author_dist:
        replacements.append((
            r"- Author distribution: " + PCT + r" maintainers, " + PCT + r" regulars, " + PCT + r" first-timers, " + PCT + r" bots \(by PR count\)",
            (
                f"- Author distribution: {author_dist['maintainer_pct']}% maintainers, "
                f"{author_dist['regular_pct']}% regulars, "
                f"{author_dist['firsttimer_pct']}% first-timers, "
                f"{author_dist['bot_pct']}% bots (by PR count)"
            ),
            "author distribution",
        ))

    # ── Era table: Median PR size row ────────────────────────────────────
    if era_pr_size:
        era_keys_list = ['2016_2019', '2020_2021', '2022_copilot', '2023_chatgpt', '2024_cursor', '2025_agentic']
        replacements.append((
            r"\| Median PR size \(lines\)\*? \|" + (r" " + NUM + r" \|") * 5 + r" \*\*" + NUM + r"\*\* \|",
            (
                "| Median PR size (lines)*"
                + "".join(f" | {era_pr_size.get(k, 0)}" for k in era_keys_list[:-1])
                + f" | **{era_pr_size.get(era_keys_list[-1], 0)}** |"
            ),
            "era table median PR size",
        ))

    # ── Section 5f: PR size inline prose ─────────────────────────────────
    if era_pr_size:
        first_size = era_pr_size.get("2016_2019", 10)
        last_size = era_pr_size.get("2025_agentic", 37)
        cursor_size = era_pr_size.get("2024_cursor", 19)
        size_ratio = f"~{round(last_size / first_size)}x" if first_size > 0 else "~4x"
        replacements.append((
            r"\*\*5f\. PRs grew ~\d+x\.\*\* Among PRs with reported size data, the median PR went from "
            + NUM + r" lines \(2016-2019\) to " + NUM + r" lines \(2025-2026\)\. The growth was gradual until 2024 \("
            + NUM + r" lines\)",
            (
                f"**5f. PRs grew {size_ratio}.** Among PRs with reported size data, the median PR went from "
                f"{first_size} lines (2016-2019) to {last_size} lines (2025-2026). The growth was gradual until 2024 ("
                f"{cursor_size} lines)"
            ),
            "5f PR size inline",
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

    # ── J. Hacktoberfest (Section 8) ─────────────────────────────────────

    hacktoberfest = stats.get("hacktoberfest", {})
    if hacktoberfest:
        hf_spike = hacktoberfest["peak_spike_pct"]
        hf_year = hacktoberfest["peak_year"]

        # "+153%" and "(2018)" in the Hacktoberfest paragraph
        replacements.append((
            r"October PR volume spiked up to \*\*\+" + FLOAT + r"%\*\* above the monthly average \(" + NUM + r"\)\.",
            (
                f"October PR volume spiked up to **+{hf_spike}%** above the monthly "
                f"average ({hf_year})."
            ),
            "hacktoberfest spike",
        ))

        # "The spike peaked in 2018"
        replacements.append((
            r"The spike peaked in " + NUM + r" and has moderated since",
            f"The spike peaked in {hf_year} and has moderated since",
            "hacktoberfest peak year",
        ))

    # ── K. Language Ecosystems (Section 9) ──────────────────────────────

    lang_comp = stats.get("language_comparison", {})
    if lang_comp:
        langs = lang_comp["languages"]
        kh = lang_comp["kruskal_h"]

        # Build a map for quick lookup
        lang_map = {l["lang"]: l["merge_rate"] for l in langs}

        # "Kruskal-Wallis H=72,707"
        replacements.append((
            r"Kruskal-Wallis H=" + NUM,
            f"Kruskal-Wallis H={fmt_int(round(kh))}",
            "language kruskal H",
        ))

        # "**C#** (79.9%) and **Rust** (77.7%) projects have the highest merge rates"
        # Match the two highest languages with their rates
        top2 = langs[:2]
        replacements.append((
            r"\*\*\w[^*]*\*\* \(" + PCT + r"\) and \*\*\w[^*]*\*\* \(" + PCT + r"\) projects have the highest merge rates",
            (
                f"**{top2[0]['lang']}** ({top2[0]['merge_rate']}%) and "
                f"**{top2[1]['lang']}** ({top2[1]['merge_rate']}%) projects have the highest merge rates"
            ),
            "language top merge rates",
        ))

        # "while **Python** (55.7%) and **Shell** (44.6%) are at the bottom"
        # Get the bottom two (excluding languages with 0% or very niche ones)
        # Use the last two with > 0 merge rate from the list
        bottom_langs = [l for l in langs if l["merge_rate"] > 0]
        bot2 = bottom_langs[-2:]
        replacements.append((
            r"while \*\*\w[^*]*\*\* \(" + PCT + r"\) and \*\*\w[^*]*\*\* \(" + PCT + r"\) are at the bottom",
            (
                f"while **{bot2[0]['lang']}** ({bot2[0]['merge_rate']}%) and "
                f"**{bot2[1]['lang']}** ({bot2[1]['merge_rate']}%) are at the bottom"
            ),
            "language bottom merge rates",
        ))

    # ── L. Project Decline (Section 11) ─────────────────────────────────

    proj_decline = stats.get("project_decline", {})
    if proj_decline:
        n_dec = proj_decline["n_declining"]
        n_stab = proj_decline["n_stable"]
        total_decline_repos = n_dec + n_stab
        best_decline_model = proj_decline["best_model"]
        best_decline_auc = proj_decline["best_auc"]
        top3 = proj_decline["top_3_features"]

        # "Of 287 repos with data in both periods, **28 are declining** and **259 are stable**."
        replacements.append((
            r"Of " + NUM + r" repos with data in both periods, \*\*" + NUM + r" are declining\*\* and \*\*" + NUM + r" are stable\*\*",
            (
                f"Of {total_decline_repos} repos with data in both periods, "
                f"**{n_dec} are declining** and **{n_stab} are stable**"
            ),
            "decline counts",
        ))

        # "An XGBoost classifier achieved **AUC=0.766**, outperforming Random Forest (AUC=0.673)."
        # The models could be in either order depending on which won, so match generically
        replacements.append((
            r"An? [\w ]+classifier achieved \*\*AUC=" + FLOAT + r"\*\*, outperforming [\w ]+ \(AUC=" + FLOAT + r"\)\.",
            (
                f"An XGBoost classifier achieved **AUC={best_decline_auc:.3f}**, "
                f"outperforming Random Forest (AUC={best_decline_auc:.3f})."
                if best_decline_model == "xgboost" else
                f"A Random Forest classifier achieved **AUC={best_decline_auc:.3f}**, "
                f"outperforming XGBoost (AUC={best_decline_auc:.3f})."
            ),
            "decline AUC",
        ))

    # ── M. GA Weights (Section 15) ──────────────────────────────────────

    ga = stats.get("ga_weights", stats.get("ga_optimization", {}))
    if ga:
        ga_w = ga["weights"]
        orig_spearman = ga["original_spearman"]
        opt_spearman = ga["optimized_spearman"]

        # Table rows: "| Response time | 25% | **48.6%** |"
        # Each component row in the table
        component_labels = {
            "response_time": "Response time",
            "bus_factor": "Bus factor",
            "diversity": "Diversity",
            "merge_rate": "Merge rate",
            "trend": "Trend",
        }

        for key, label in component_labels.items():
            if key not in ga_w:
                continue
            orig_pct = ga_w[key]["original_pct"]
            opt_pct = ga_w[key]["optimized_pct"]
            # Match the row, allowing for optional bold markers on the optimized weight
            replacements.append((
                rf"\| {label} \| " + PCT + r" \| (?:\*\*)?" + PCT + r"(?:\*\*)? \|",
                f"| {label} | {orig_pct}% | {opt_pct}% |",
                f"GA table {label}",
            ))

        # "The optimized weights improved Spearman correlation from 0.14 to 0.21."
        SIGNED_FLOAT = r"-?" + FLOAT
        replacements.append((
            r"improved Spearman correlation from " + SIGNED_FLOAT + r" to " + SIGNED_FLOAT + r"\.",
            f"improved Spearman correlation from {orig_spearman} to {opt_spearman}.",
            "GA spearman improvement",
        ))

        # ── Dynamic GA prose: find biggest winner and biggest loser ────
        # Sort components by change in weight to describe accurately
        ga_changes = []
        for key, info in ga_w.items():
            ga_changes.append({
                "key": key,
                "label": component_labels.get(key, key),
                "orig": info["original_pct"],
                "opt": info["optimized_pct"],
                "delta": info["optimized_pct"] - info["original_pct"],
            })
        ga_changes.sort(key=lambda x: x["delta"], reverse=True)

        # Biggest increase = first element, biggest decrease = last element
        biggest_increase = ga_changes[0]
        biggest_decrease = ga_changes[-1]

        # Build the two-change prose line: "The GA massively increased trend (15.0% to 72.8%) and eliminated response time (25.0% to 0%)."
        inc_label = biggest_increase["label"].lower()
        dec_label = biggest_decrease["label"].lower()

        inc_desc = f"massively increased {inc_label} ({biggest_increase['orig']}% to {biggest_increase['opt']}%)"
        if biggest_decrease["opt"] == 0.0:
            dec_desc = f"eliminated {dec_label} ({biggest_decrease['orig']}% to {biggest_decrease['opt']}%)"
        else:
            dec_desc = f"reduced {dec_label} ({biggest_decrease['orig']}% to {biggest_decrease['opt']}%)"

        # Replace the old two-change prose sentence (matches both halves)
        replacements.append((
            r"The GA [\w ]+\(" + PCT + r" to " + FLOAT + r"%\) and [\w ]+\(" + PCT + r" to " + PCT + r"\)\.",
            f"The GA {inc_desc} and {dec_desc}.",
            "GA prose two-change",
        ))

        # ── Dynamic insight paragraph ─────────────────────────────────
        # The old insight says "how fast it responds" (response_time).
        # Generate correct insight based on the component with highest optimized weight.
        top_component = max(ga_w.items(), key=lambda x: x[1]["optimized_pct"])
        top_key = top_component[0]
        top_label = component_labels.get(top_key, top_key).lower()
        insight_map = {
            "trend": "its recent momentum (activity trend)",
            "response_time": "how fast it responds to contributions",
            "bus_factor": "its bus factor (contribution concentration)",
            "diversity": "its contributor diversity",
            "merge_rate": "its merge rate",
        }
        top_insight = insight_map.get(top_key, top_label)
        replacements.append((
            r"\*\*The insight\*\*: The single best predictor of whether a project will grow is \*\*[^*]+\*\*\."
            r" Not [^.]+\.",
            f"**The insight**: The single best predictor of whether a project will grow is **{top_insight}**."
            f" Not its merge rate, not its response time.",
            "GA insight paragraph",
        ))
        # Also replace the follow-up sentence about alignment
        replacements.append((
            r"Speed of response\. This aligns with.*?fast feedback\.",
            f"A project with strong upward momentum attracts more contributors regardless of other factors.",
            "GA insight follow-up",
        ))

    # ── H. Counterfactual table (Section 5j) ────────────────────────────
    if counterfactual:
        cf_map = {
            "pr_volume": "PR volume/month",
            "unique_contributors": "Unique contributors/month",
            "first_timers": "First-timers/month",
            "rejection_rate": "Rejection rate",
            "ft_rejection_rate": "First-timer rejection rate",
            "median_pr_size": "Median PR size (lines)",
        }
        for key, label in cf_map.items():
            if key not in counterfactual:
                continue
            cf = counterfactual[key]
            pred = cf["predicted"]
            actual = cf["actual"]
            excess = cf["excess_pct"]
            pred_str = fmt_int(int(pred)) if pred > 10 else f"{pred:.0f}%"
            act_str = fmt_int(int(actual)) if actual > 10 else f"{actual:.0f}%"
            if key == "rejection_rate":
                pred_str = f"{pred * 100:.0f}%"
                act_str = f"{actual * 100:.0f}%"
            elif key == "ft_rejection_rate":
                pred_str = f"{pred * 100:.0f}%"
                act_str = f"{actual * 100:.0f}%"
            elif key == "median_pr_size":
                pred_str = str(round(pred))
                act_str = str(round(actual))

            escaped_label = re.escape(label)
            # Match table cells that may contain plain numbers or percentages
            CF_CELL = r"[\d,.]+%?"
            replacements.append((
                r"\| " + escaped_label + r" \| " + CF_CELL + r" \| " + CF_CELL + r" \| \*\*\+" + NUM + r"%\*\* \|",
                f"| {label} | {pred_str} | {act_str} | **+{excess}%** |",
                f"counterfactual {key}",
            ))

        # Narrative prose: "N% above what the pre-AI trend predicted"
        if "pr_volume" in counterfactual:
            pv = counterfactual["pr_volume"]["excess_pct"]
            replacements.append((
                r"PR volume is " + NUM + r"% above",
                f"PR volume is {pv}% above",
                "counterfactual PR volume prose",
            ))
        if "unique_contributors" in counterfactual:
            uc = counterfactual["unique_contributors"]["excess_pct"]
            replacements.append((
                NUM + r"% more unique contributors than expected",
                f"{uc}% more unique contributors than expected",
                "counterfactual contributors prose",
            ))
        if "first_timers" in counterfactual:
            ft_ex = counterfactual["first_timers"]["excess_pct"]
            replacements.append((
                NUM + r"% more first-timers than the model predicted",
                f"{ft_ex}% more first-timers than the model predicted",
                "counterfactual first-timers prose",
            ))
        if "rejection_rate" in counterfactual:
            rr = counterfactual["rejection_rate"]["excess_pct"]
            replacements.append((
                r"rejection rate diverged.*?" + NUM + r"% higher than expected",
                f"rejection rate diverged the most from the prediction: {rr}% higher than expected",
                "counterfactual rejection prose",
            ))
        if "ft_rejection_rate" in counterfactual:
            ftr_cf = counterfactual["ft_rejection_rate"]
            ftr = ftr_cf["excess_pct"]
            ftr_pred_pct = ftr_cf["predicted"] * 100
            ftr_act_pct = ftr_cf["actual"] * 100
            replacements.append((
                r"First-timer rejection is " + NUM + r"% above the counterfactual",
                f"First-timer rejection is {ftr}% above the counterfactual",
                "counterfactual FT rejection prose",
            ))
            # "The model predicted rejection would stabilize around 52%; instead it climbed to 58-63%."
            replacements.append((
                r"The model predicted rejection would stabilize around " + NUM + r"%; instead it climbed to " + NUM + r"-\d+%",
                f"The model predicted rejection would stabilize around {ftr_pred_pct:.0f}%; instead it climbed to {ftr_act_pct:.0f}-{eras['2025_agentic']['ft_rejection_rate']:.0f}%",
                "counterfactual FT rejection narrative",
            ))
        if "median_pr_size" in counterfactual:
            ms = counterfactual["median_pr_size"]["excess_pct"]
            replacements.append((
                r"PRs are " + NUM + r"% larger than expected",
                f"PRs are {ms}% larger than expected",
                "counterfactual PR size prose",
            ))

    # ── I. Project decline (Section 11) ───────────────────────────────
    if decline and isinstance(decline.get("top_3_features"), list) and decline["top_3_features"]:
        n_dec = decline["n_declining"]
        n_sta = decline["n_stable"]
        n_tot = decline.get("n_total", n_dec + n_sta)
        best_auc_d = decline["best_auc"]

        replacements.append((
            r"Of " + NUM + r" repos with data in both periods, \*\*" + NUM + r" are declining\*\* and \*\*" + NUM + r" are stable\*\*",
            f"Of {n_tot} repos with data in both periods, **{n_dec} are declining** and **{n_sta} are stable**",
            "decline counts",
        ))

        top3 = decline["top_3_features"]
        if isinstance(top3[0], dict):
            f1 = top3[0]
            f2 = top3[1] if len(top3) > 1 else {"feature": "?", "importance": 0}
            f3 = top3[2] if len(top3) > 2 else {"feature": "?", "importance": 0}

            replacements.append((
                r"1\. \*\*[A-Za-z_ ]+\*\* \([^)]+, [\d.]+%\)",
                f"1. **{f1['feature']}** ({f1['feature']}, {f1['importance']*100:.1f}%)",
                "decline feature 1",
            ))
            replacements.append((
                r"2\. \*\*[A-Za-z_ ]+\*\* \([\d.]+%\)",
                f"2. **{f2['feature']}** ({f2['importance']*100:.1f}%)",
                "decline feature 2",
            ))
            replacements.append((
                r"3\. \*\*[A-Za-z_ ]+\*\* \([\d.]+%\)",
                f"3. **{f3['feature']}** ({f3['importance']*100:.1f}%)",
                "decline feature 3",
            ))

    # ── J. Generate Section 16 (first-timer decision trees) ─────────────
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
