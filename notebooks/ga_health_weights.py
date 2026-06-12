"""Genetic Algorithm to optimize Health Index weights for oss-pulse.

Uses DEAP to find weight vectors that maximize Spearman correlation
between the health index (computed on historical data) and actual PR
growth measured 6+ months later.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from deap import algorithms, base, creator, tools
from scipy import stats as scipy_stats

# ---------------------------------------------------------------------------
# Make oss_pulse importable
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from oss_pulse.analyze.health_index import (
    DEFAULT_WEIGHTS,
    compute_health_components,
    compute_health_index,
)
from oss_pulse.visualize.style import save_fig, setup_style

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
DATA_PATH = ROOT / "data" / "processed" / "repo_monthly.parquet"
df = pd.read_parquet(DATA_PATH)
print(f"Loaded {len(df)} monthly records for {df['repo_name'].nunique()} repos")

# ---------------------------------------------------------------------------
# Temporal split
# ---------------------------------------------------------------------------
# Historical period: everything before 2025-01
# Future period: 2025-01 to 2025-12
hist_df = df[
    (df["year"] < 2025)
    | ((df["year"] == 2025) & (df["month"] == 0))  # always False, just for clarity
].copy()
hist_df = df[df["year"] < 2025].copy()

future_df = df[(df["year"] == 2025) & (df["month"] >= 1) & (df["month"] <= 12)].copy()

print(f"Historical records: {len(hist_df)} ({hist_df['repo_name'].nunique()} repos)")
print(f"Future records:     {len(future_df)} ({future_df['repo_name'].nunique()} repos)")

# ---------------------------------------------------------------------------
# Compute health components on historical data
# ---------------------------------------------------------------------------
components = compute_health_components(hist_df)
print(f"Health components computed for {len(components)} repos")

# ---------------------------------------------------------------------------
# Compute actual growth: average monthly PR count in 2025 vs last 6 months
# of historical data (2024-H2)
# ---------------------------------------------------------------------------
future_growth = (
    future_df.groupby("repo_name")["pr_count"].mean().rename("future_pr_avg")
)

last_6mo = hist_df[
    (hist_df["year"] == 2024) & (hist_df["month"] >= 7)
].copy()
baseline = last_6mo.groupby("repo_name")["pr_count"].mean().rename("baseline_pr_avg")

growth = pd.DataFrame({"future": future_growth, "baseline": baseline}).dropna()
# Relative growth: (future - baseline) / (baseline + 1) to avoid div-by-zero
growth["growth"] = (growth["future"] - growth["baseline"]) / (
    growth["baseline"] + 1
)

# Keep only repos that appear in both components and growth
common_repos = set(components["repo_name"]) & set(growth.index)
print(f"Repos with both health scores and growth data: {len(common_repos)}")

components_aligned = (
    components[components["repo_name"].isin(common_repos)]
    .set_index("repo_name")
    .sort_index()
)
growth_aligned = growth.loc[growth.index.isin(common_repos)].sort_index()

# ---------------------------------------------------------------------------
# Fitness function
# ---------------------------------------------------------------------------
COMPONENT_NAMES = ["response_time", "merge_rate", "diversity", "trend", "bus_factor"]


def evaluate(individual: list[float]) -> tuple[float]:
    """Compute Spearman correlation between weighted health index and growth."""
    raw = np.array(individual, dtype=float)
    raw = np.abs(raw)
    total = raw.sum()
    if total == 0:
        return (-1.0,)
    normed = raw / total

    weights = dict(zip(COMPONENT_NAMES, normed))
    health = compute_health_index(components_aligned, weights)

    rho, _ = scipy_stats.spearmanr(health.values, growth_aligned["growth"].values)
    if np.isnan(rho):
        return (-1.0,)
    return (rho,)


# ---------------------------------------------------------------------------
# Quick-check with original weights
# ---------------------------------------------------------------------------
original_fitness = evaluate([DEFAULT_WEIGHTS[c] for c in COMPONENT_NAMES])
print(f"\nOriginal weights Spearman rho: {original_fitness[0]:.4f}")

# ---------------------------------------------------------------------------
# DEAP GA setup
# ---------------------------------------------------------------------------
random.seed(42)
np.random.seed(42)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, 0.0, 1.0)
toolbox.register(
    "individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=5
)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# ---------------------------------------------------------------------------
# Run the GA
# ---------------------------------------------------------------------------
POP_SIZE = 100
N_GEN = 50
CXPB = 0.7  # crossover probability
MUTPB = 0.2  # mutation probability

pop = toolbox.population(n=POP_SIZE)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("max", np.max)
stats.register("avg", np.mean)
stats.register("min", np.min)

hof = tools.HallOfFame(1)

print(f"\nRunning GA: pop={POP_SIZE}, gens={N_GEN}, cxpb={CXPB}, mutpb={MUTPB}")
print("-" * 60)

pop, logbook = algorithms.eaSimple(
    pop,
    toolbox,
    cxpb=CXPB,
    mutpb=MUTPB,
    ngen=N_GEN,
    stats=stats,
    halloffame=hof,
    verbose=True,
)

# ---------------------------------------------------------------------------
# Extract best solution
# ---------------------------------------------------------------------------
best_raw = np.array(hof[0], dtype=float)
best_raw = np.abs(best_raw)
best_normed = best_raw / best_raw.sum()
best_weights = dict(zip(COMPONENT_NAMES, best_normed))

optimized_fitness = evaluate(hof[0])

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

print("\n{:<20s} {:>12s} {:>12s}".format("Component", "Original", "Optimized"))
print("-" * 44)
for name in COMPONENT_NAMES:
    print(
        "{:<20s} {:>11.1f}% {:>11.1f}%".format(
            name, DEFAULT_WEIGHTS[name] * 100, best_weights[name] * 100
        )
    )

print(f"\nSpearman rho (original):  {original_fitness[0]:.4f}")
print(f"Spearman rho (optimized): {optimized_fitness[0]:.4f}")
print(f"Improvement:              {optimized_fitness[0] - original_fitness[0]:+.4f}")

# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------
sorted_components = sorted(best_weights.items(), key=lambda x: x[1], reverse=True)
print("\n--- Interpretation ---")
print(
    "The GA's optimized weights rank the health components by predictive "
    "power for future PR growth as follows:"
)
for i, (name, w) in enumerate(sorted_components, 1):
    diff = w - DEFAULT_WEIGHTS[name]
    direction = "UP" if diff > 0.01 else ("DOWN" if diff < -0.01 else "~same")
    print(f"  {i}. {name:<20s} {w*100:5.1f}%  ({direction} from {DEFAULT_WEIGHTS[name]*100:.0f}%)")

biggest_gain = sorted_components[0]
biggest_loss = sorted_components[-1]
print(
    f"\nThe GA thinks '{biggest_gain[0]}' matters most for predicting growth "
    f"({biggest_gain[1]*100:.1f}%), while '{biggest_loss[0]}' matters least "
    f"({biggest_loss[1]*100:.1f}%)."
)
delta = optimized_fitness[0] - original_fitness[0]
if abs(delta) < 0.02:
    print(
        "The improvement is modest, suggesting the original weights were "
        "a reasonable starting point."
    )
elif delta > 0:
    print(
        f"The optimized weights improve predictive correlation by {delta:+.4f}, "
        "a meaningful gain that suggests re-weighting is worthwhile."
    )
else:
    print(
        "The GA did not find better weights than the original, "
        "suggesting the hand-tuned values are already near-optimal."
    )

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
setup_style()

# --- Figure 1: Convergence curve ---
gen = logbook.select("gen")
fit_max = logbook.select("max")
fit_avg = logbook.select("avg")

fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(gen, fit_max, color="#a4133c", linewidth=2, label="Best fitness")
ax1.plot(gen, fit_avg, color="#92b1f5", linewidth=1.5, linestyle="--", label="Average fitness")
ax1.set_xlabel("Generation")
ax1.set_ylabel("Spearman Correlation (rho)")
ax1.set_title("GA Convergence: Health Index Weight Optimization")
ax1.legend()
ax1.set_xlim(0, N_GEN)

path1 = save_fig(fig1, "ga_convergence")
print(f"\nSaved: {path1}")

# --- Figure 2: Weight comparison ---
x = np.arange(len(COMPONENT_NAMES))
width = 0.35

original_vals = [DEFAULT_WEIGHTS[c] for c in COMPONENT_NAMES]
optimized_vals = [best_weights[c] for c in COMPONENT_NAMES]

labels = [c.replace("_", " ").title() for c in COMPONENT_NAMES]

fig2, ax2 = plt.subplots(figsize=(10, 6))
bars1 = ax2.bar(x - width / 2, original_vals, width, label="Original", color="#92b1f5")
bars2 = ax2.bar(x + width / 2, optimized_vals, width, label="Optimized (GA)", color="#a4133c")

ax2.set_xlabel("Health Component")
ax2.set_ylabel("Weight")
ax2.set_title("Health Index Weights: Original vs GA-Optimized")
ax2.set_xticks(x)
ax2.set_xticklabels(labels, rotation=15, ha="right")
ax2.legend()

# Add value labels on bars
for bar in bars1:
    h = bar.get_height()
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        h + 0.005,
        f"{h:.0%}",
        ha="center",
        va="bottom",
        fontsize=9,
    )
for bar in bars2:
    h = bar.get_height()
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        h + 0.005,
        f"{h:.0%}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

path2 = save_fig(fig2, "ga_weights_comparison")
print(f"Saved: {path2}")

plt.close("all")
print("\nDone.")
