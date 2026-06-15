#!/usr/bin/env python
"""Regenerate ALL analysis figures for oss-pulse with the 2.5M PR / 325 repo dataset.

Run from project root:
    .venv/bin/python scripts/regenerate_all_figures.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from oss_pulse.visualize.style import PALETTE, save_fig, setup_style

setup_style()
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*kurtosistest.*")
warnings.filterwarnings("ignore", message=".*divide by zero.*")
warnings.filterwarnings("ignore", message=".*invalid value.*")

import os
os.chdir(PROJECT_ROOT)

DATA = PROJECT_ROOT / "data"
FIGURES = PROJECT_ROOT / "output" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
BAR_BLUE = "#92b1f5"
LINE_BLUE = "#2563eb"
WADA = {"red": "#a4133c", "blue": "#3a86ff", "green": "#06d6a0"}

AI_EVENT_LINES = [
    ("2022-06-21", "Copilot GA"),
    ("2022-11-30", "ChatGPT"),
    ("2023-03-14", "GPT-4"),
    ("2024-03-01", "Cursor AI"),
    ("2025-02-24", "Claude Code"),
]

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading data ...")
pr_df = pd.read_parquet(DATA / "processed/pr_events_featured.parquet")
repo_monthly = pd.read_parquet(DATA / "processed/repo_monthly.parquet")
top_repos = pd.read_parquet(DATA / "raw/top_repos.parquet")

repo_meta = top_repos[["repo_name", "language", "stars"]].copy()
pr_df = pr_df.merge(repo_meta, on="repo_name", how="left")

# Deduplicate to one row per PR for many analyses
pr_unique = pr_df.drop_duplicates(subset=["repo_name", "pr_number"], keep="last").copy()

# Add year_month column
pr_unique["year_month"] = pr_unique["pr_created_at"].dt.to_period("M")

# Build aggregate monthly
monthly_agg = repo_monthly.groupby(["year", "month"]).agg(
    pr_count=("pr_count", "sum"),
    merged_count=("merged_count", "sum"),
    unique_contributors=("unique_contributors", "sum"),
    median_merge_time_hours=("median_merge_time_hours", "median"),
).reset_index()
monthly_agg["date"] = pd.to_datetime(
    monthly_agg["year"].astype(str) + "-"
    + monthly_agg["month"].astype(str).str.zfill(2)
    + "-01"
)
monthly_agg = monthly_agg.sort_values("date")

print(f"  Total PRs:        {len(pr_df):,}")
print(f"  Unique PRs:       {len(pr_unique):,}")
print(f"  Repositories:     {pr_df['repo_name'].nunique()}")
print(f"  Unique authors:   {pr_unique['author'].nunique():,}")


def _add_ai_event_lines(ax, ypos_frac=0.95):
    """Add vertical event lines for AI milestones."""
    ylim = ax.get_ylim()
    ypos = ylim[0] + (ylim[1] - ylim[0]) * ypos_frac
    for date_str, label in AI_EVENT_LINES:
        d = pd.Timestamp(date_str)
        ax.axvline(d, color="#9ca3af", linestyle=":", alpha=0.6, linewidth=0.8)
        ax.text(
            d, ypos, label, rotation=45, fontsize=7,
            ha="right", va="top", color="#6b7280",
        )


# ===================================================================
# 1. AI DEEP-DIVE (ai_01 through ai_06)
# ===================================================================
print("\n--- AI deep-dive figures ---")

# Monthly aggregates for AI series
human = pr_unique[~pr_unique["is_bot"]].copy()
human["ym"] = human["pr_created_at"].dt.to_period("M")

ai_monthly = human.groupby("ym").agg(
    unique_authors=("author", "nunique"),
    total_prs=("pr_number", "count"),
    rejected=("pr_outcome", lambda x: (x == "closed").sum()),
    first_timers=("author_class", lambda x: (x == "first-timer").sum()),
    ft_rejected=("pr_outcome", lambda x: 0),  # placeholder
    median_size=("additions", "median"),
).reset_index()
ai_monthly["date"] = ai_monthly["ym"].dt.to_timestamp()
ai_monthly = ai_monthly.sort_values("date")
ai_monthly["rejection_rate"] = ai_monthly["rejected"] / ai_monthly["total_prs"]
ai_monthly["prs_per_contributor"] = ai_monthly["total_prs"] / ai_monthly["unique_authors"]

# First-timer rejection rate
ft_prs = human[human["author_class"] == "first-timer"].copy()
ft_monthly = ft_prs.groupby(ft_prs["pr_created_at"].dt.to_period("M")).agg(
    ft_count=("pr_number", "count"),
    ft_rejected=("pr_outcome", lambda x: (x == "closed").sum()),
    ft_merged=("pr_outcome", lambda x: (x == "merged").sum()),
).reset_index()
ft_monthly.columns = ["ym", "ft_count", "ft_rejected", "ft_merged"]
ft_monthly["date"] = ft_monthly["ym"].dt.to_timestamp()
ft_monthly = ft_monthly.sort_values("date")
ft_monthly["ft_rejection_rate"] = ft_monthly["ft_rejected"] / ft_monthly["ft_count"]
ft_monthly["ft_merge_rate"] = ft_monthly["ft_merged"] / ft_monthly["ft_count"]

# Rejected contributors monthly
rejected_authors_monthly = (
    human[human["pr_outcome"] == "closed"]
    .groupby(human.loc[human["pr_outcome"] == "closed", "pr_created_at"].dt.to_period("M"))["author"]
    .nunique()
    .reset_index()
)
rejected_authors_monthly.columns = ["ym", "rejected_authors"]
rejected_authors_monthly["date"] = rejected_authors_monthly["ym"].dt.to_timestamp()
rejected_authors_monthly = rejected_authors_monthly.sort_values("date")

# PR size trend
size_monthly = human.groupby(human["pr_created_at"].dt.to_period("M")).agg(
    median_additions=("additions", "median"),
    median_deletions=("deletions", "median"),
).reset_index()
size_monthly.columns = ["ym", "median_additions", "median_deletions"]
size_monthly["date"] = size_monthly["ym"].dt.to_timestamp()
size_monthly["median_total"] = size_monthly["median_additions"] + size_monthly["median_deletions"]
size_monthly = size_monthly.sort_values("date")


# ai_01: Unique contributors
print("  ai_01_unique_contributors")
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(ai_monthly["date"], ai_monthly["unique_authors"], color=LINE_BLUE, linewidth=1.5)
ax.fill_between(ai_monthly["date"], ai_monthly["unique_authors"], alpha=0.08, color=LINE_BLUE)
_add_ai_event_lines(ax)
ax.set_xlabel("Date")
ax.set_ylabel("Unique Contributors per Month")
ax.set_title("Monthly Unique Contributors (non-bot)", fontsize=14, fontweight="bold")
save_fig(fig, "ai_01_unique_contributors")
plt.close(fig)

# ai_02: Rejection rate
print("  ai_02_rejection_rate")
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(ai_monthly["date"], ai_monthly["rejection_rate"] * 100, color=LINE_BLUE, linewidth=1.5)
_add_ai_event_lines(ax)
ax.set_xlabel("Date")
ax.set_ylabel("Rejection Rate (%)")
ax.set_title("Monthly PR Rejection Rate (non-bot)", fontsize=14, fontweight="bold")
save_fig(fig, "ai_02_rejection_rate")
plt.close(fig)

# ai_03: Rejected contributors
print("  ai_03_rejected_contributors")
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(rejected_authors_monthly["date"], rejected_authors_monthly["rejected_authors"],
        color=LINE_BLUE, linewidth=1.5)
_add_ai_event_lines(ax)
ax.set_xlabel("Date")
ax.set_ylabel("Unique Authors with Rejected PRs")
ax.set_title("Monthly Rejected Contributors (non-bot)", fontsize=14, fontweight="bold")
save_fig(fig, "ai_03_rejected_contributors")
plt.close(fig)

# ai_04: First-timer analysis (count + merge rate dual axis)
print("  ai_04_firsttimer_analysis")
fig, ax1 = plt.subplots(figsize=(14, 6))
ax2 = ax1.twinx()
ax1.bar(ft_monthly["date"], ft_monthly["ft_count"], width=25, color=BAR_BLUE, alpha=0.6, label="First-timer PRs")
ax2.plot(ft_monthly["date"], ft_monthly["ft_merge_rate"] * 100, color=LINE_BLUE, linewidth=1.5, label="FT Merge Rate %")
_add_ai_event_lines(ax1)
ax1.set_xlabel("Date")
ax1.set_ylabel("First-timer PR Count")
ax2.set_ylabel("First-timer Merge Rate (%)")
ax1.set_title("First-timer PRs and Merge Rate", fontsize=14, fontweight="bold")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
save_fig(fig, "ai_04_firsttimer_analysis")
plt.close(fig)

# ai_05: PR size trend
print("  ai_05_pr_size_trend")
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(size_monthly["date"], size_monthly["median_total"], color=LINE_BLUE, linewidth=1.5)
_add_ai_event_lines(ax)
ax.set_xlabel("Date")
ax.set_ylabel("Median PR Size (additions + deletions)")
ax.set_title("Monthly Median PR Size Trend (non-bot)", fontsize=14, fontweight="bold")
save_fig(fig, "ai_05_pr_size_trend")
plt.close(fig)

# ai_06: PRs per contributor
print("  ai_06_prs_per_contributor")
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(ai_monthly["date"], ai_monthly["prs_per_contributor"], color=LINE_BLUE, linewidth=1.5)
_add_ai_event_lines(ax)
ax.set_xlabel("Date")
ax.set_ylabel("PRs per Contributor per Month")
ax.set_title("Monthly PRs per Contributor (non-bot)", fontsize=14, fontweight="bold")
save_fig(fig, "ai_06_prs_per_contributor")
plt.close(fig)


# ===================================================================
# 2. ai_07: Productivity by group (maintainer/regular/first-timer)
# ===================================================================
print("  ai_07_productivity_by_group")

wada_colors = [WADA["red"], WADA["blue"], WADA["green"]]
author_types = ["maintainer", "regular", "first-timer"]

group_monthly = (
    human.groupby([human["pr_created_at"].dt.to_period("M"), "author_class"])
    .agg(
        pr_count=("pr_number", "count"),
        unique_authors=("author", "nunique"),
        merged=("pr_outcome", lambda x: (x == "merged").sum()),
    )
    .reset_index()
)
group_monthly.columns = ["ym", "author_class", "pr_count", "unique_authors", "merged"]
group_monthly["date"] = group_monthly["ym"].dt.to_timestamp()
group_monthly["prs_per_person"] = group_monthly["pr_count"] / group_monthly["unique_authors"]
group_monthly["merge_rate"] = group_monthly["merged"] / group_monthly["pr_count"]
group_monthly = group_monthly.sort_values("date")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

for i, atype in enumerate(author_types):
    sub = group_monthly[group_monthly["author_class"] == atype]
    ax1.plot(sub["date"], sub["prs_per_person"], color=wada_colors[i], linewidth=1.3, label=atype)
    ax2.plot(sub["date"], sub["merge_rate"] * 100, color=wada_colors[i], linewidth=1.3, label=atype)

ax1.set_title("PRs/Person/Month by Author Type", fontsize=13, fontweight="bold")
ax1.set_xlabel("Date")
ax1.set_ylabel("PRs per Person")
ax1.legend()

ax2.set_title("Merge Rate by Author Type", fontsize=13, fontweight="bold")
ax2.set_xlabel("Date")
ax2.set_ylabel("Merge Rate (%)")
ax2.legend()

save_fig(fig, "ai_07_productivity_by_group")
plt.close(fig)


# ===================================================================
# 3. COUNTERFACTUAL (ETS forecasts, counterfactual_01 through 06)
# ===================================================================
print("\n--- Counterfactual figures ---")

from statsmodels.tsa.holtwinters import ExponentialSmoothing

COPILOT_DATE = pd.Timestamp("2022-06-01")

# Build monthly series from repo_monthly
cf_agg = repo_monthly.groupby(["year", "month"]).agg(
    pr_count=("pr_count", "sum"),
    unique_authors=("unique_contributors", "sum"),
    median_size=("median_merge_time_hours", "median"),
    merge_rate=("merge_rate", "mean"),
).reset_index()
cf_agg["date"] = pd.to_datetime(
    cf_agg["year"].astype(str) + "-" + cf_agg["month"].astype(str).str.zfill(2) + "-01"
)
cf_agg = cf_agg.sort_values("date").set_index("date")
cf_agg.index = pd.DatetimeIndex(cf_agg.index, freq="MS")

# First-timer counts and rejection from pr_unique
ft_unique = pr_unique[pr_unique["author_class"] == "first-timer"].copy()
ft_unique["ym_date"] = ft_unique["pr_created_at"].dt.to_period("M").dt.to_timestamp()

ft_cf = ft_unique.groupby("ym_date").agg(
    ft_count=("pr_number", "count"),
    ft_rejected=("pr_outcome", lambda x: (x == "closed").sum()),
).reset_index()
ft_cf = ft_cf.set_index("ym_date").sort_index()
ft_cf.index = pd.DatetimeIndex(ft_cf.index, freq="MS")
ft_cf["ft_rejection"] = ft_cf["ft_rejected"] / ft_cf["ft_count"]

# Median PR size from human PRs
human_size = human.copy()
human_size["ym_date"] = human_size["pr_created_at"].dt.to_period("M").dt.to_timestamp()
size_cf = human_size.groupby("ym_date").agg(
    median_size=("additions", "median"),
).reset_index().set_index("ym_date").sort_index()
size_cf.index = pd.DatetimeIndex(size_cf.index, freq="MS")

# Overall rejection rate
rej_cf = human.copy()
rej_cf["ym_date"] = rej_cf["pr_created_at"].dt.to_period("M").dt.to_timestamp()
rej_agg = rej_cf.groupby("ym_date").agg(
    total=("pr_number", "count"),
    rejected=("pr_outcome", lambda x: (x == "closed").sum()),
).reset_index().set_index("ym_date").sort_index()
rej_agg.index = pd.DatetimeIndex(rej_agg.index, freq="MS")
rej_agg["rejection_rate"] = rej_agg["rejected"] / rej_agg["total"]

# Unique authors per month from cf_agg
# We already have cf_agg which has unique_authors

counterfactual_specs = [
    ("counterfactual_01_pr_count", cf_agg["pr_count"], "Aggregate Monthly PR Count"),
    ("counterfactual_02_unique_authors", cf_agg["unique_authors"], "Monthly Unique Contributors"),
    ("counterfactual_03_ft_count", ft_cf["ft_count"], "Monthly First-timer PRs"),
    ("counterfactual_04_rejection_rate", rej_agg["rejection_rate"], "Monthly Rejection Rate"),
    ("counterfactual_05_ft_rejection", ft_cf["ft_rejection"], "Monthly First-timer Rejection Rate"),
    ("counterfactual_06_median_size", size_cf["median_size"], "Monthly Median PR Size (additions)"),
]

for fig_name, series, title in counterfactual_specs:
    print(f"  {fig_name}")
    s = series.dropna()
    if len(s) < 24:
        print(f"    SKIP: only {len(s)} data points")
        continue

    # Ensure frequency
    s = s.asfreq("MS")
    s = s.ffill()

    pre = s[s.index < COPILOT_DATE]
    post = s[s.index >= COPILOT_DATE]

    if len(pre) < 12:
        print(f"    SKIP: only {len(pre)} pre-Copilot points")
        continue

    try:
        model = ExponentialSmoothing(
            pre, trend="add", seasonal=None, initialization_method="estimated"
        )
        fitted = model.fit(optimized=True)
        forecast = fitted.forecast(steps=len(post))
        forecast.index = post.index[:len(forecast)]
    except Exception as e:
        print(f"    ETS failed: {e}")
        continue

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(pre.index, pre.values, color=LINE_BLUE, linewidth=1.3, label="Pre-Copilot (actual)")
    ax.plot(post.index, post.values, color=LINE_BLUE, linewidth=1.3, label="Post-Copilot (actual)")
    ax.plot(forecast.index, forecast.values, color="#ef4444", linewidth=1.3, linestyle="--",
            label="ETS forecast (no AI scenario)")
    ax.axvline(COPILOT_DATE, color="#9ca3af", linestyle=":", alpha=0.7, linewidth=1)
    ax.text(COPILOT_DATE, ax.get_ylim()[1] * 0.95, "Copilot GA", fontsize=8, ha="right",
            va="top", color="#6b7280", rotation=45)
    ax.fill_between(forecast.index, forecast.values, post.values[:len(forecast)],
                    alpha=0.08, color="#ef4444")
    ax.set_title(f"Counterfactual: {title}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel(title.split("Monthly ")[-1] if "Monthly " in title else title)
    ax.legend()
    save_fig(fig, fig_name)
    plt.close(fig)


# ===================================================================
# 4. HOLIDAYS (holidays_01, 02, 03)
# ===================================================================
print("\n--- Holiday figures ---")

# Holiday definitions by country (major holidays as month-day)
COUNTRY_HOLIDAYS = {
    "US": [(1, 1), (7, 4), (11, 28), (12, 25)],  # NY, July4, Thanksgiving ~, Christmas
    "CN": [(2, 1), (10, 1)],  # CNY ~, National Day ~
    "IN": [(1, 26), (8, 15), (10, 24)],  # Republic, Independence, Diwali ~
    "DE": [(12, 25), (12, 26), (1, 1)],  # Christmas, NY
    "JP": [(1, 1), (5, 3), (5, 4), (5, 5)],  # NY, Golden Week
}

# Compute daily PR counts
daily_prs = pr_unique.copy()
daily_prs["date"] = daily_prs["pr_created_at"].dt.date
daily_counts = daily_prs.groupby("date").size().reset_index(name="pr_count")
daily_counts["date"] = pd.to_datetime(daily_counts["date"])
daily_counts = daily_counts.sort_values("date").set_index("date")

# Compute baseline: median for that day of week in the same month across years
daily_counts["dow"] = daily_counts.index.dayofweek
daily_counts["month"] = daily_counts.index.month
baseline = daily_counts.groupby(["dow", "month"])["pr_count"].median()
daily_counts["baseline"] = daily_counts.apply(
    lambda r: baseline.get((r["dow"], r["month"]), r["pr_count"]), axis=1
)
daily_counts["pct_change"] = (daily_counts["pr_count"] - daily_counts["baseline"]) / daily_counts["baseline"] * 100

# holidays_01: Impact by country
print("  holidays_01_by_country")
country_effects = {}
for country, holidays in COUNTRY_HOLIDAYS.items():
    effects = []
    for m, d in holidays:
        # Look at the holiday +/- 1 day window
        mask = False
        for year in range(2016, 2027):
            for delta in range(-1, 2):
                try:
                    hdate = pd.Timestamp(year, m, d) + pd.Timedelta(days=delta)
                    if hdate in daily_counts.index:
                        mask = mask | (daily_counts.index == hdate)
                except ValueError:
                    continue
        if isinstance(mask, pd.Series) or (isinstance(mask, np.ndarray) and mask.any()):
            sub = daily_counts.loc[mask] if not isinstance(mask, bool) else pd.DataFrame()
            if len(sub) > 0:
                effects.append(sub["pct_change"].mean())
    country_effects[country] = np.mean(effects) if effects else 0

ce_df = pd.DataFrame(list(country_effects.items()), columns=["country", "avg_pct_drop"])
ce_df = ce_df.sort_values("avg_pct_drop")

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(ce_df["country"], ce_df["avg_pct_drop"], color=BAR_BLUE, edgecolor="white")
for bar, val in zip(bars, ce_df["avg_pct_drop"]):
    ax.text(bar.get_width() - 1 if val < 0 else bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", fontsize=10)
ax.set_xlabel("Average % Change from Baseline")
ax.set_title("PR Activity Change During National Holidays", fontsize=14, fontweight="bold")
ax.axvline(0, color="gray", linewidth=0.5)
save_fig(fig, "holidays_01_by_country")
plt.close(fig)

# holidays_02: Impact by overlap (multiple countries' holidays on same day)
print("  holidays_02_by_overlap")
# Count how many countries have a holiday on each month-day
from collections import Counter
all_holidays_md = []
for country, holidays in COUNTRY_HOLIDAYS.items():
    for m, d in holidays:
        all_holidays_md.append((m, d))
md_counts = Counter(all_holidays_md)

overlap_effects = {0: [], 1: [], 2: []}  # 0 = no holiday, 1 = one country, 2+ = overlap
for idx, row in daily_counts.iterrows():
    m, d = idx.month, idx.day
    n_countries = md_counts.get((m, d), 0)
    # Also check +/- 1
    for delta in [-1, 1]:
        try:
            od = (idx + pd.Timedelta(days=delta))
            n_countries = max(n_countries, md_counts.get((od.month, od.day), 0))
        except Exception:
            pass
    if n_countries == 0:
        overlap_effects[0].append(row["pct_change"])
    elif n_countries == 1:
        overlap_effects[1].append(row["pct_change"])
    else:
        overlap_effects[2].append(row["pct_change"])

overlap_labels = ["No Holiday", "Single Country", "Multi-Country\nOverlap"]
overlap_means = [np.mean(overlap_effects[k]) if overlap_effects[k] else 0 for k in [0, 1, 2]]

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(overlap_labels, overlap_means, color=BAR_BLUE, edgecolor="white")
for i, v in enumerate(overlap_means):
    ax.text(i, v + 0.2 if v >= 0 else v - 1.5, f"{v:.1f}%", ha="center", fontsize=11, fontweight="bold")
ax.set_ylabel("Average % Change from Baseline")
ax.set_title("PR Activity by Holiday Overlap", fontsize=14, fontweight="bold")
ax.axhline(0, color="gray", linewidth=0.5)
save_fig(fig, "holidays_02_by_overlap")
plt.close(fig)

# holidays_03: Activity by month
print("  holidays_03_by_month")
monthly_avg = daily_counts.groupby("month")["pr_count"].mean()
overall_avg = daily_counts["pr_count"].mean()
monthly_pct = ((monthly_avg - overall_avg) / overall_avg * 100)

fig, ax = plt.subplots(figsize=(12, 5))
month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
colors = [BAR_BLUE if v >= 0 else "#f87171" for v in monthly_pct.values]
ax.bar(month_names, monthly_pct.values, color=colors, edgecolor="white")
for i, v in enumerate(monthly_pct.values):
    ax.text(i, v + 0.3 if v >= 0 else v - 1.0, f"{v:.1f}%", ha="center", fontsize=9)
ax.set_ylabel("% Deviation from Annual Average")
ax.set_title("Monthly PR Activity Seasonality", fontsize=14, fontweight="bold")
ax.axhline(0, color="gray", linewidth=0.5)
save_fig(fig, "holidays_03_by_month")
plt.close(fig)


# ===================================================================
# 5. HACKTOBERFEST
# ===================================================================
print("\n--- Hacktoberfest figure ---")

pr_unique_copy = pr_unique.copy()
pr_unique_copy["year"] = pr_unique_copy["pr_created_at"].dt.year
pr_unique_copy["month_num"] = pr_unique_copy["pr_created_at"].dt.month

# October vs non-October PRs per year
yearly = pr_unique_copy.groupby("year").agg(total=("pr_number", "count")).reset_index()
oct_prs = pr_unique_copy[pr_unique_copy["month_num"] == 10].groupby("year").agg(
    oct_count=("pr_number", "count"),
    oct_merged=("pr_outcome", lambda x: (x == "merged").sum()),
).reset_index()
non_oct = pr_unique_copy[pr_unique_copy["month_num"] != 10].groupby("year").agg(
    non_oct_count=("pr_number", "count"),
    non_oct_merged=("pr_outcome", lambda x: (x == "merged").sum()),
).reset_index()

hack_df = yearly.merge(oct_prs, on="year", how="left").merge(non_oct, on="year", how="left")
hack_df = hack_df[(hack_df["year"] >= 2016) & (hack_df["year"] <= 2025)]
hack_df["oct_pct_of_year"] = hack_df["oct_count"] / hack_df["total"] * 100
hack_df["oct_merge_rate"] = hack_df["oct_merged"] / hack_df["oct_count"] * 100
hack_df["non_oct_merge_rate"] = hack_df["non_oct_merged"] / hack_df["non_oct_count"] * 100

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Left: October spike
ax1.bar(hack_df["year"], hack_df["oct_pct_of_year"], color=BAR_BLUE, edgecolor="white")
for i, row in hack_df.iterrows():
    ax1.text(row["year"], row["oct_pct_of_year"] + 0.2,
             f"{row['oct_pct_of_year']:.1f}%", ha="center", fontsize=8)
ax1.axhline(100 / 12, color="gray", linestyle="--", linewidth=1, label="Expected (8.3%)")
ax1.set_xlabel("Year")
ax1.set_ylabel("October as % of Annual PRs")
ax1.set_title("October PR Spike (Hacktoberfest)", fontsize=13, fontweight="bold")
ax1.legend()

# Right: Merge rate comparison
x = np.arange(len(hack_df))
w = 0.35
ax2.bar(x - w / 2, hack_df["oct_merge_rate"], w, color=BAR_BLUE, label="October", edgecolor="white")
ax2.bar(x + w / 2, hack_df["non_oct_merge_rate"], w, color=WADA["green"], label="Non-October", edgecolor="white")
ax2.set_xticks(x)
ax2.set_xticklabels(hack_df["year"].astype(str), rotation=45)
ax2.set_ylabel("Merge Rate (%)")
ax2.set_title("October vs Non-October Merge Rate", fontsize=13, fontweight="bold")
ax2.legend()

save_fig(fig, "hacktoberfest_effect")
plt.close(fig)


# ===================================================================
# 6. COMPARATIVE (merge rate & merge time by language)
# ===================================================================
print("\n--- Comparative figures ---")

repo_monthly_lang = repo_monthly.merge(
    repo_meta[["repo_name", "language"]], on="repo_name", how="left"
)
lang_stats = repo_monthly_lang.dropna(subset=["language"]).groupby("language").agg(
    mean_merge_rate=("merge_rate", "mean"),
    median_merge_time=("median_merge_time_hours", "median"),
    total_prs=("pr_count", "sum"),
).reset_index()
# Keep only languages with substantial data
lang_stats = lang_stats[lang_stats["total_prs"] >= 1000].sort_values("mean_merge_rate", ascending=True)

# comparative_merge_by_language
print("  comparative_merge_by_language")
fig, ax = plt.subplots(figsize=(12, max(6, len(lang_stats) * 0.4)))
ax.barh(lang_stats["language"], lang_stats["mean_merge_rate"] * 100, color=BAR_BLUE, edgecolor="white")
for i, (_, row) in enumerate(lang_stats.iterrows()):
    ax.text(row["mean_merge_rate"] * 100 + 0.5, i,
            f"{row['mean_merge_rate'] * 100:.1f}%", va="center", fontsize=9)
ax.set_xlabel("Mean Merge Rate (%)")
ax.set_title("PR Merge Rate by Language", fontsize=14, fontweight="bold")
save_fig(fig, "comparative_merge_by_language")
plt.close(fig)

# comparative_mergetime_by_language
print("  comparative_mergetime_by_language")
lang_stats_time = lang_stats.sort_values("median_merge_time", ascending=True)
fig, ax = plt.subplots(figsize=(12, max(6, len(lang_stats_time) * 0.4)))
ax.barh(lang_stats_time["language"], lang_stats_time["median_merge_time"], color=BAR_BLUE, edgecolor="white")
for i, (_, row) in enumerate(lang_stats_time.iterrows()):
    ax.text(row["median_merge_time"] + 0.5, i,
            f"{row['median_merge_time']:.1f}h", va="center", fontsize=9)
ax.set_xlabel("Median Merge Time (hours)")
ax.set_title("Median PR Merge Time by Language", fontsize=14, fontweight="bold")
save_fig(fig, "comparative_mergetime_by_language")
plt.close(fig)


# ===================================================================
# 7. ABANDONMENT (feature importance from RF classifier)
# ===================================================================
print("\n--- Abandonment feature importance ---")

from oss_pulse.analyze.abandonment import fit_abandonment_classifier

# Build features for abandonment classifier
aband_df = pr_unique.copy()
aband_df["is_abandoned"] = (aband_df["pr_outcome"] == "abandoned").astype(int)

# Only include closed/merged/abandoned (exclude open)
aband_df = aband_df[aband_df["pr_outcome"].isin(["merged", "closed", "abandoned"])].copy()

# Encode features
aband_df["is_first_timer"] = (aband_df["author_class"] == "first-timer").astype(int)
aband_df["is_maintainer"] = (aband_df["author_class"] == "maintainer").astype(int)
aband_df["is_regular"] = (aband_df["author_class"] == "regular").astype(int)
aband_df["is_weekend_int"] = aband_df["is_weekend"].astype(int)
aband_df["total_changes"] = aband_df["additions"] + aband_df["deletions"]
aband_df["log_additions"] = np.log1p(aband_df["additions"])
aband_df["log_deletions"] = np.log1p(aband_df["deletions"])
aband_df["log_changed_files"] = np.log1p(aband_df["changed_files"])
aband_df["log_total_changes"] = np.log1p(aband_df["total_changes"])

feature_cols = [
    "log_additions", "log_deletions", "log_changed_files", "log_total_changes",
    "is_first_timer", "is_maintainer", "is_regular",
    "is_weekend_int", "hour", "day_of_week", "month",
]
features_df = aband_df[feature_cols + ["is_abandoned"]].dropna()

# Sample if too large
if len(features_df) > 500_000:
    features_df = features_df.sample(500_000, random_state=42)

print(f"  Training on {len(features_df)} PRs ({features_df['is_abandoned'].sum()} abandoned)")
result = fit_abandonment_classifier(features_df)
print(f"  RF AUC: {result['rf_auc']:.4f}, XGB AUC: {result['xgb_auc']:.4f}")
print(f"  Best model: {result['best_model']}")

imp_df = result["feature_importance"].head(15)
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(imp_df["feature"].values[::-1], imp_df["importance"].values[::-1],
        color=BAR_BLUE, edgecolor="white")
for i, (_, row) in enumerate(imp_df.iloc[::-1].iterrows()):
    ax.text(row["importance"] + 0.002, i, f"{row['importance']:.3f}", va="center", fontsize=9)
ax.set_xlabel("Feature Importance")
ax.set_title(f"Abandonment Prediction: Top Features ({result['best_model']}, AUC={max(result['rf_auc'], result['xgb_auc']):.3f})",
             fontsize=13, fontweight="bold")
save_fig(fig, "abandonment_feature_importance")
plt.close(fig)


# ===================================================================
# 8. FORECAST BENCHMARK
# ===================================================================
print("\n--- Forecast benchmark ---")

from oss_pulse.analyze.forecast import benchmark_models

# Aggregate monthly PR count series
forecast_series = monthly_agg.set_index("date")["pr_count"].copy()
forecast_series.index = pd.DatetimeIndex(forecast_series.index, freq="MS")

print("  Running benchmark (ARIMA, Prophet, ETS, XGBoost) ...")
bench_df = benchmark_models(forecast_series)
print(bench_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(bench_df))
metrics = ["mae", "rmse", "mape"]
wada_list = [WADA["red"], WADA["blue"], WADA["green"]]
w = 0.25

for i, metric in enumerate(metrics):
    vals = bench_df[metric].values
    ax.bar(x + i * w, vals, w, label=metric.upper(), color=wada_list[i], edgecolor="white")

ax.set_xticks(x + w)
ax.set_xticklabels(bench_df["model"].values)
ax.set_ylabel("Error Metric Value")
ax.set_title("Forecast Model Benchmark (Aggregate Monthly PRs)", fontsize=14, fontweight="bold")
ax.legend()
save_fig(fig, "forecast_benchmark")
plt.close(fig)


# ===================================================================
# 9. FIRST-TIMER (success features + best repos)
# ===================================================================
print("\n--- First-timer figures ---")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# First-timer success features
ft_all = pr_unique[pr_unique["author_class"] == "first-timer"].copy()
ft_all = ft_all[ft_all["pr_outcome"].isin(["merged", "closed"])].copy()
ft_all["is_merged"] = (ft_all["pr_outcome"] == "merged").astype(int)
ft_all["log_additions"] = np.log1p(ft_all["additions"])
ft_all["log_deletions"] = np.log1p(ft_all["deletions"])
ft_all["log_changed_files"] = np.log1p(ft_all["changed_files"])
ft_all["total_changes"] = ft_all["additions"] + ft_all["deletions"]
ft_all["log_total_changes"] = np.log1p(ft_all["total_changes"])
ft_all["is_weekend_int"] = ft_all["is_weekend"].astype(int)

ft_feature_cols = [
    "log_additions", "log_deletions", "log_changed_files", "log_total_changes",
    "is_weekend_int", "hour", "day_of_week", "month",
]
ft_features = ft_all[ft_feature_cols + ["is_merged"]].dropna()
if len(ft_features) > 200_000:
    ft_features = ft_features.sample(200_000, random_state=42)

X_ft = ft_features[ft_feature_cols]
y_ft = ft_features["is_merged"]

X_train, X_test, y_train, y_test = train_test_split(X_ft, y_ft, test_size=0.2, random_state=42, stratify=y_ft)
rf_ft = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf_ft.fit(X_train, y_train)
ft_auc = roc_auc_score(y_test, rf_ft.predict_proba(X_test)[:, 1])
print(f"  First-timer RF AUC: {ft_auc:.4f}")

ft_imp = pd.DataFrame({
    "feature": ft_feature_cols,
    "importance": rf_ft.feature_importances_,
}).sort_values("importance", ascending=False)

print("  firsttimer_success_features")
fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(ft_imp["feature"].values[::-1], ft_imp["importance"].values[::-1],
        color=BAR_BLUE, edgecolor="white")
for i, (_, row) in enumerate(ft_imp.iloc[::-1].iterrows()):
    ax.text(row["importance"] + 0.002, i, f"{row['importance']:.3f}", va="center", fontsize=9)
ax.set_xlabel("Feature Importance")
ax.set_title(f"First-timer Success Prediction Features (RF, AUC={ft_auc:.3f})",
             fontsize=13, fontweight="bold")
save_fig(fig, "firsttimer_success_features")
plt.close(fig)

# Best repos for first-timers (real software only, exclude awesome lists etc.)
print("  firsttimer_best_repos")
# Filter to repos with language (proxy for "real software")
ft_with_lang = ft_all[ft_all["language"].notna()].copy()
repo_ft_stats = ft_with_lang.groupby("repo_name").agg(
    ft_prs=("pr_number", "count"),
    ft_merged=("is_merged", "sum"),
).reset_index()
repo_ft_stats["ft_merge_rate"] = repo_ft_stats["ft_merged"] / repo_ft_stats["ft_prs"]
# Require at least 20 first-timer PRs
repo_ft_stats = repo_ft_stats[repo_ft_stats["ft_prs"] >= 20]
repo_ft_stats = repo_ft_stats.sort_values("ft_merge_rate", ascending=False).head(20)

fig, ax = plt.subplots(figsize=(12, 8))
ax.barh(
    repo_ft_stats["repo_name"].values[::-1],
    repo_ft_stats["ft_merge_rate"].values[::-1] * 100,
    color=BAR_BLUE, edgecolor="white",
)
for i, (_, row) in enumerate(repo_ft_stats.iloc[::-1].iterrows()):
    ax.text(
        row["ft_merge_rate"] * 100 + 0.5, i,
        f"{row['ft_merge_rate'] * 100:.1f}% ({row['ft_prs']:.0f} PRs)",
        va="center", fontsize=8,
    )
ax.set_xlabel("First-timer Merge Rate (%)")
ax.set_title("Top 20 Repos for First-timer Success (min 20 FT PRs, real software)",
             fontsize=13, fontweight="bold")
save_fig(fig, "firsttimer_best_repos")
plt.close(fig)


# ===================================================================
# 10. SURVIVAL (KM curves, save as PNG)
# ===================================================================
print("\n--- Survival figures ---")

from oss_pulse.analyze.abandonment import build_survival_data, fit_kaplan_meier
from lifelines import KaplanMeierFitter

surv_data = build_survival_data(pr_df)
print(f"  Survival data: {len(surv_data):,} PRs")

wada_survival = [WADA["red"], WADA["blue"], WADA["green"]]

# 08_survival_by_author_type.png
print("  08_survival_by_author_type.png")
fig, ax = plt.subplots(figsize=(12, 7))
for i, atype in enumerate(["maintainer", "regular", "first-timer"]):
    subset = surv_data[surv_data["author_type"] == atype]
    if len(subset) < 10:
        continue
    kmf = KaplanMeierFitter()
    kmf.fit(subset["duration_days"], subset["event"], label=atype)
    kmf.plot_survival_function(ax=ax, color=wada_survival[i], linewidth=1.5)

ax.set_title("PR Survival by Author Type", fontsize=14, fontweight="bold")
ax.set_xlabel("Days")
ax.set_ylabel("Survival Probability")
ax.legend()
save_fig(fig, "08_survival_by_author_type", fmt="png")
plt.close(fig)

# 08_survival_by_size.png
print("  08_survival_by_size.png")
fig, ax = plt.subplots(figsize=(12, 7))
for i, size in enumerate(["small", "medium", "large"]):
    subset = surv_data[surv_data["pr_size_bucket"] == size]
    if len(subset) < 10:
        continue
    kmf = KaplanMeierFitter()
    kmf.fit(subset["duration_days"], subset["event"], label=size)
    kmf.plot_survival_function(ax=ax, color=wada_survival[i], linewidth=1.5)

ax.set_title("PR Survival by Size", fontsize=14, fontweight="bold")
ax.set_xlabel("Days")
ax.set_ylabel("Survival Probability")
ax.legend()
save_fig(fig, "08_survival_by_size", fmt="png")
plt.close(fig)


# ===================================================================
# PRINT KEY UPDATED NUMBERS
# ===================================================================
print("\n" + "=" * 60)
print("KEY DATASET STATISTICS")
print("=" * 60)

total_prs = len(pr_unique)
n_repos = pr_unique["repo_name"].nunique()
merged_count = (pr_unique["pr_outcome"] == "merged").sum()
merge_rate = merged_count / total_prs
median_merge_h = pr_unique["time_to_merge_hours"].median()
unique_contributors = pr_unique["author"].nunique()

# Funnel stats
from oss_pulse.analyze.funnel import build_contributor_funnel, compute_retention_rates
funnel = build_contributor_funnel(pr_unique)
funnel = compute_retention_rates(funnel)

ft_prs_total = len(pr_unique[pr_unique["author_class"] == "first-timer"])
ft_merge_rate = (
    (pr_unique[pr_unique["author_class"] == "first-timer"]["pr_outcome"] == "merged").sum()
    / ft_prs_total
)
rejection_rate = (pr_unique["pr_outcome"] == "closed").sum() / total_prs
abandoned_count = (pr_unique["pr_outcome"] == "abandoned").sum()

print(f"Total PRs:              {total_prs:,}")
print(f"Repositories:           {n_repos}")
print(f"Overall merge rate:     {merge_rate:.4f} ({merge_rate*100:.1f}%)")
print(f"Median merge time:      {median_merge_h:.1f} hours")
print(f"Unique contributors:    {unique_contributors:,}")
print(f"Rejection rate:         {rejection_rate:.4f} ({rejection_rate*100:.1f}%)")
print(f"Abandoned PRs:          {abandoned_count:,}")
print(f"First-timer PRs:        {ft_prs_total:,}")
print(f"First-timer merge rate: {ft_merge_rate:.4f} ({ft_merge_rate*100:.1f}%)")
print()
print("Contributor Funnel:")
for _, row in funnel.iterrows():
    rr = f" (retention: {row['retention_rate']:.1%})" if row["retention_rate"] is not None else ""
    print(f"  {row['stage']:>10s}: {row['count']:>7,} ({row['percentage']:.1f}%){rr}")

print(f"\nDate range: {pr_unique['pr_created_at'].min().strftime('%Y-%m-%d')} to {pr_unique['pr_created_at'].max().strftime('%Y-%m-%d')}")
print(f"Languages: {pr_unique['language'].nunique()} distinct")

print("\n--- ALL DONE ---")
print(f"Figures saved to: {FIGURES}")
