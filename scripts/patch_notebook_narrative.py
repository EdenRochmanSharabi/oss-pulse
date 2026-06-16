"""
Patch 00_narrative_analysis.ipynb:
- Add stats.json loading to Cell 1
- Convert every markdown cell that contains hardcoded numbers into a code cell
  that renders the text dynamically from stats.json / computed values.
"""

import json
from pathlib import Path

NB_PATH = Path(__file__).parent.parent / "notebooks" / "00_narrative_analysis.ipynb"


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source,
    }


with open(NB_PATH) as f:
    nb = json.load(f)

cells = nb["cells"]

# ──────────────────────────────────────────────────────────────────────────────
# Cell 0: title + dataset hero stats
# ──────────────────────────────────────────────────────────────────────────────
cells[0] = code_cell(
    "from IPython.display import display, Markdown\n"
    "\n"
    "display(Markdown(f\"\"\"\n"
    "# OSS Pulse: A Data-Driven Story of Open Source PR Activity\n"
    "\n"
    "**{stats['dataset']['total_prs']:,} pull requests. "
    "{stats['dataset']['n_repos']} repositories. "
    "{stats['dataset']['n_contributors']:,}+ contributors. 10 years of data.**\n"
    "\n"
    "This notebook tells a narrative about how open source software is built, maintained, and evolved,\n"
    "following the data from broad patterns to specific discoveries. Each finding motivates the next question.\n"
    "\"\"\"))"
)

# ──────────────────────────────────────────────────────────────────────────────
# Cell 1: add stats.json loading after existing imports
# ──────────────────────────────────────────────────────────────────────────────
existing_cell1 = "".join(cells[1]["source"])
stats_load = (
    "\n\nimport json\n"
    "with open(DATA / \"processed/stats.json\") as f:\n"
    "    stats = json.load(f)"
)
cells[1]["source"] = existing_cell1 + stats_load

# ──────────────────────────────────────────────────────────────────────────────
# Cell 2: section header — "156,000 Pull Requests Walk Into a Dataset"
# The number in the heading is part of the section title (flavor text, not a
# stat), so update to match the actual total.
# ──────────────────────────────────────────────────────────────────────────────
cells[2] = code_cell(
    "display(Markdown(f\"\"\"\n"
    "---\n"
    "## 1. \\\"{stats['dataset']['total_prs']:,} Pull Requests Walk Into a Dataset\\\"\n"
    "\n"
    "Before we can tell any story, we need to meet the cast. Let's load the data and get a sense of its shape:\n"
    "how many repos, how many PRs, what time period, and what languages are represented.\n"
    "\"\"\"))"
)

# ──────────────────────────────────────────────────────────────────────────────
# Cell 6: merge rate key takeaway
# ──────────────────────────────────────────────────────────────────────────────
cells[6] = code_cell(
    "display(Markdown(f\"\"\"\n"
    "**Key takeaway:** About {stats['outcomes']['merge_rate']:.0f}% of all PRs eventually get merged, "
    "but the variance across repositories is enormous.\n"
    "Some projects merge nearly everything; others reject (or ignore) the majority. This raises a natural question:\n"
    "is the overall volume of open source contributions actually growing, or are we just seeing the same projects churn?\n"
    "\"\"\"))"
)

# ──────────────────────────────────────────────────────────────────────────────
# Cell 7: section header — "The Growth Curve No One Talks About"
# ──────────────────────────────────────────────────────────────────────────────
cells[7] = code_cell(
    "display(Markdown(f\"\"\"\n"
    "---\n"
    "## 2. \\\"The Growth Curve No One Talks About\\\"\n"
    "\n"
    "Open source is often described as \\\"exploding\\\" in popularity, but is it? "
    "Let's look at monthly PR volume\n"
    "across all {stats['dataset']['n_repos']} repos and decompose the signal into "
    "trend, seasonality, and residuals.\n"
    "\"\"\"))"
)

# ──────────────────────────────────────────────────────────────────────────────
# Cell 12: growth discovery
# Growth factor is computed inline from tool_eras in stats.json
#   agentic era prs_per_month / pre_copilot prs_per_month
# ──────────────────────────────────────────────────────────────────────────────
cells[12] = code_cell(
    "_growth_factor = (\n"
    "    stats['tool_eras']['agentic']['prs_per_month']\n"
    "    / stats['tool_eras']['pre_copilot']['prs_per_month']\n"
    ")\n"
    "display(Markdown(f\"\"\"\n"
    "**Discovery:** OSS contributions have grown roughly {_growth_factor:.1f}x "
    "from the pre-Copilot era to today, but the growth is not uniform.\n"
    "The aggregate trend is dominated by a handful of extremely active repos, "
    "while many others stay relatively flat.\n"
    "The STL decomposition reveals a clear seasonal dip every December/January "
    "and the ADF test confirms\n"
    "the trend is genuine (the series is non-stationary).\n"
    "\n"
    "But if contributions are clustered in time, are they also clustered in "
    "the *day and hour* they happen?\n"
    "\"\"\"))"
)

# ──────────────────────────────────────────────────────────────────────────────
# Cell 17: weekend rate discovery
# weekend_rate is computed inline from pr_df (not yet in stats.json)
# ──────────────────────────────────────────────────────────────────────────────
cells[17] = code_cell(
    "# Weekend rate: computed from unique PRs (not events)\n"
    "_pr_unique = pr_df.drop_duplicates(subset=['repo_name', 'pr_number'])\n"
    "_weekend_rate_pct = _pr_unique['is_weekend'].mean() * 100\n"
    "\n"
    "display(Markdown(f\"\"\"\n"
    "**Discovery:** The workweek pattern is unmistakable. The vast majority of PRs are "
    "created Monday through Friday,\n"
    "peaking during afternoon UTC hours (which corresponds to morning in the US and "
    "afternoon in Europe).\n"
    "Only about {_weekend_rate_pct:.0f}% of contributions happen on weekends.\n"
    "\n"
    "This strongly suggests that most open source work happens during business hours, "
    "probably as part of\n"
    "professional roles rather than purely as a hobby. The language-level patterns are "
    "broadly similar, though\n"
    "the exact peak hours shift slightly by ecosystem.\n"
    "\n"
    "Now that we know *when* code is contributed, the next question is: "
    "how quickly does it get reviewed and merged?\n"
    "\"\"\"))"
)

# ──────────────────────────────────────────────────────────────────────────────
# Cell 18: section header — "The X-Hour PR: How Fast Is Open Source?"
# ──────────────────────────────────────────────────────────────────────────────
cells[18] = code_cell(
    "display(Markdown(f\"\"\"\n"
    "---\n"
    "## 4. \\\"The {stats['merge_time']['median_hours']:.1f}-Hour PR: "
    "How Fast Is Open Source?\\\"\n"
    "\n"
    "Speed matters. A PR that sits for weeks discourages future contributions. "
    "Let's look at the distribution\n"
    "of merge times and see what factors accelerate or slow down the process.\n"
    "\"\"\"))"
)

# ──────────────────────────────────────────────────────────────────────────────
# Cell 23: merge time discovery
# ──────────────────────────────────────────────────────────────────────────────
cells[23] = code_cell(
    "display(Markdown(f\"\"\"\n"
    "**Discovery:** The median PR takes about {stats['merge_time']['median_hours']:.1f} hours "
    "to merge, but first-timers wait significantly longer\n"
    "than maintainers. Large PRs also take longer, which is expected given the review burden.\n"
    "\n"
    "Is the longer wait for first-timers gatekeeping, or is it healthy quality control?\n"
    "One way to investigate is to look at what happens *after* the first PR.\n"
    "If contributors feel welcomed, they come back. If they don't, "
    "the project has a retention problem.\n"
    "\"\"\"))"
)

# ──────────────────────────────────────────────────────────────────────────────
# Cell 24: section header — "X% Never Come Back"
# dropout = 100 - 2nd_pr_pct
# ──────────────────────────────────────────────────────────────────────────────
cells[24] = code_cell(
    "_dropout_pct = 100 - stats['funnel']['2nd_pr_pct']\n"
    "display(Markdown(f\"\"\"\n"
    "---\n"
    "## 5. \\\"{_dropout_pct:.0f}% Never Come Back\\\"\n"
    "\n"
    "Every open source project depends on new contributors. "
    "But how many people who open their first PR\n"
    "ever come back for a second? Let's trace the contributor funnel.\n"
    "\"\"\"))"
)

# ──────────────────────────────────────────────────────────────────────────────
# Cell 26: funnel chart — update title with live numbers
# ──────────────────────────────────────────────────────────────────────────────
# Cell 26 is a code cell already; we just update the title string in the source
cell26_src = "".join(cells[26]["source"])
cells[26]["source"] = cell26_src.replace(
    'plot_funnel(funnel, title="Contributor Engagement Funnel (46k+ authors)")',
    'plot_funnel(funnel, title=f"Contributor Engagement Funnel ({stats[\'dataset\'][\'n_contributors\']:,}+ authors)")',
)

# ──────────────────────────────────────────────────────────────────────────────
# Cell 29: attrition discovery
# ──────────────────────────────────────────────────────────────────────────────
cells[29] = code_cell(
    "_dropout_pct = 100 - stats['funnel']['2nd_pr_pct']\n"
    "display(Markdown(f\"\"\"\n"
    "**Discovery:** Open source has a massive attrition problem. "
    "Around {_dropout_pct:.0f}% of contributors never return after\n"
    "their first PR. The funnel narrows dramatically at every stage, "
    "from {stats['funnel']['total']:,} first-time contributors down to just\n"
    "{stats['funnel']['regular']:,} who reach \\\"regular\\\" status "
    "({stats['funnel']['regular_pct']:.0f}% of starters, defined as 20+ PRs).\n"
    "\n"
    "The first-to-second PR transition is where the bulk of contributors are lost, "
    "and this pattern holds\n"
    "across all language ecosystems. The question of *why* they leave is harder "
    "to answer with PR data alone,\n"
    "but one contributing factor might be how the broader ecosystem is evolving.\n"
    "\n"
    "Speaking of ecosystem-level shifts, has the rise of AI coding tools like "
    "GitHub Copilot and ChatGPT\n"
    "left any detectable fingerprint on PR activity?\n"
    "\"\"\"))"
)

# ──────────────────────────────────────────────────────────────────────────────
# Cell 33: changepoint discovery — replace "82 repos" with live n_repos
# ──────────────────────────────────────────────────────────────────────────────
cells[33] = code_cell(
    "display(Markdown(f\"\"\"\n"
    "**Discovery:** The changepoint detector finds structural breaks in the PR volume series, "
    "and we can compare\n"
    "their timing against major AI tool launches. Whether any detected shift *caused by* "
    "AI tools versus other\n"
    "factors (organic growth, new repos entering the dataset, etc.) remains an open question.\n"
    "The data reveals correlation at best, not causation.\n"
    "\n"
    "Rather than speculate further on macro trends, let's turn inward and ask: "
    "across all {stats['dataset']['n_repos']} repos,\n"
    "which ones are actually *healthy* projects?\n"
    "\"\"\"))"
)

# ──────────────────────────────────────────────────────────────────────────────
# Cell 45: section header — "Who's Actually More Productive?"
# The 40% figure was the old hardcoded headline; replace with the live maintainer gain.
# ──────────────────────────────────────────────────────────────────────────────
cells[45] = code_cell(
    "_maint_gain = stats['productivity_by_author_type']['maintainer']['productivity_change_pct']\n"
    "display(Markdown(f\"\"\"\n"
    "---\n"
    "## 8b. \\\"Who's Actually More Productive?\\\"\n"
    "\n"
    "The overall productivity increase (driven by a {_maint_gain:.0f}% surge among maintainers) "
    "masks very different dynamics across contributor types.\n"
    "Let's break it down: are first-timers, regulars, and maintainers all producing more? "
    "And crucially,\n"
    "are they getting their PRs merged at the same rate?\n"
    "\"\"\"))"
)

# ──────────────────────────────────────────────────────────────────────────────
# Cell 48: productivity discovery
# ──────────────────────────────────────────────────────────────────────────────
cells[48] = code_cell(
    "_p = stats['productivity_by_author_type']\n"
    "_m_pre  = _p['maintainer']['pre_2023']['prs_per_person_month']\n"
    "_m_post = _p['maintainer']['post_2023']['prs_per_person_month']\n"
    "_m_gain = _p['maintainer']['productivity_change_pct']\n"
    "_r_gain = _p['regular']['productivity_change_pct']\n"
    "_ft_mr_pre  = _p['first-timer']['pre_2023']['merge_rate_pct']\n"
    "_ft_mr_post = _p['first-timer']['post_2023']['merge_rate_pct']\n"
    "_reg_mr_pre  = _p['regular']['pre_2023']['merge_rate_pct']\n"
    "_reg_mr_post = _p['regular']['post_2023']['merge_rate_pct']\n"
    "_maint_mr_pre  = _p['maintainer']['pre_2023']['merge_rate_pct']\n"
    "_maint_mr_post = _p['maintainer']['post_2023']['merge_rate_pct']\n"
    "\n"
    "display(Markdown(f\"\"\"\n"
    "**Discovery:** The {_m_gain:.0f}% productivity increase among maintainers "
    "({_m_pre} to {_m_post} PRs/month) is the dominant story.\n"
    "First-timers stay at 1.0 by definition, and regulars barely moved (+{_r_gain:.0f}%). "
    "Meanwhile, merge rates\n"
    "dropped sharply for first-timers ({_ft_mr_pre:.0f}% to {_ft_mr_post:.0f}%) "
    "and regulars ({_reg_mr_pre:.0f}% to {_reg_mr_post:.0f}%), "
    "while maintainers held steady\n"
    "at ~{_maint_mr_post:.0f}%. More people are producing more code, "
    "but the gatekeeping has tightened for everyone except\n"
    "the inner circle.\n"
    "\"\"\"))"
)

# ──────────────────────────────────────────────────────────────────────────────
# Cell 50: key findings summary
# ──────────────────────────────────────────────────────────────────────────────
cells[50] = code_cell(
    "_p = stats['productivity_by_author_type']\n"
    "_dropout_pct = 100 - stats['funnel']['2nd_pr_pct']\n"
    "_regular_pct = stats['funnel']['regular_pct']\n"
    "_growth_factor = (\n"
    "    stats['tool_eras']['agentic']['prs_per_month']\n"
    "    / stats['tool_eras']['pre_copilot']['prs_per_month']\n"
    ")\n"
    "_pr_unique = pr_df.drop_duplicates(subset=['repo_name', 'pr_number'])\n"
    "_weekend_rate_pct = _pr_unique['is_weekend'].mean() * 100\n"
    "\n"
    "display(Markdown(f\"\"\"\n"
    "### Key Findings\n"
    "\n"
    "1. **The merge rate is {stats['outcomes']['merge_rate']:.0f}%, but the variance is massive.**\n"
    "   Across {stats['dataset']['n_repos']} repos, some merge nearly every PR while others "
    "reject the majority.\n"
    "   The \\\"average\\\" hides significant project-level differences.\n"
    "\n"
    "2. **PR volume has grown ~{_growth_factor:.1f}x from the pre-Copilot era to today,**\n"
    "   but the growth is concentrated in a few high-activity repos.\n"
    "   Many projects maintain a steady, modest pace.\n"
    "\n"
    "3. **Open source is a workday activity.**\n"
    "   Only {_weekend_rate_pct:.0f}% of contributions happen on weekends.\n"
    "   Peak activity is Monday through Thursday, 14:00-17:00 UTC, suggesting most contributors are\n"
    "   working during business hours in US/European time zones.\n"
    "\n"
    "4. **The median PR merges in {stats['merge_time']['median_hours']:.1f} hours,**\n"
    "   but first-timer PRs wait significantly longer than maintainer PRs, "
    "and large PRs take longer than small ones.\n"
    "\n"
    "5. **{_dropout_pct:.0f}% of contributors never come back after their first PR.**\n"
    "   The first-to-second PR transition is where most contributors are lost.\n"
    "   Only about {_regular_pct:.1f}% of first-time contributors ever reach \\\"regular\\\" status.\n"
    "\n"
    "6. **Changepoint detection finds structural breaks** in PR volume that may or may not "
    "coincide with\n"
    "   AI tool launches. The data shows correlation, not causation.\n"
    "\n"
    "7. **Stars are a poor proxy for project health.** The composite health index "
    "(response time, merge rate,\n"
    "   diversity, trend, bus factor) reveals that some of the most popular repos score poorly,\n"
    "   while lesser-known projects can be very healthy.\n"
    "\n"
    "8. **Survival analysis confirms the two-speed nature of OSS.** Most PRs resolve within days,\n"
    "   but a significant tail lingers for months. Author type and PR size are strong predictors\n"
    "   of resolution speed.\n"
    "\n"
    "### Limitations\n"
    "\n"
    "- **{stats['dataset']['n_repos']} repos.** These are among the most popular repos on GitHub,\n"
    "  not representative of the long tail of smaller projects.\n"
    "- **Selection bias toward starred repos.** Popularity is the sampling criterion.\n"
    "- **PR-level data only.** We lack issue data, commit-level data, and discussion/community\n"
    "  engagement signals that would give a fuller picture.\n"
    "- **UTC timestamps.** Contributor location inference from UTC activity patterns is "
    "approximate at best.\n"
    "\n"
    "### Future Work\n"
    "\n"
    "- Expand coverage and include issue data\n"
    "- Add commit-level analysis to distinguish code contributions from documentation/CI changes\n"
    "- Build a predictive model for contributor retention\n"
    "- Track the health index over time to detect declining projects early\n"
    "\"\"\"))"
)

# ──────────────────────────────────────────────────────────────────────────────
# Write patched notebook
# ──────────────────────────────────────────────────────────────────────────────
with open(NB_PATH, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook patched successfully.")
print(f"Written to: {NB_PATH}")
