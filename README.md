# oss-pulse

**A time-series study of 4.0 million Pull Requests across 580 top open-source software projects (2016-2026).**

Open source runs the world, but how does it actually work? Who contributes, how fast do projects respond, and what happens to the thousands of developers who open their first PR? This study analyzes a decade of Pull Request activity to find out.

---

## Research Questions

1. **bus_factor_score** (10.9%): projects outside the top 8 languages are more likely to decline, possibly due to smaller contributor pools.
2. **diversity_score** (10.5%): repos with fewer lifetime PRs are more vulnerable, suggesting that a deep contribution history acts as a buffer.
3. **trend_score** (10.3%): projects with uneven contributor distributions (high Gini) are at greater risk. When one or two people drive most of the activity, their departure hits harder.

<img src="output/figures/project_decline_features.svg" width="100%">

### 12. Forecasting: Which Model Predicts PR Volume Best?

We benchmarked four time-series models on the aggregate monthly PR volume (80/20 temporal split):

| Model | MAE | RMSE | MAPE |
|-------|-----|------|------|
| **ETS** | **2,595** | **4,269** | **21.7%** |
| ARIMA | 3,174 | 4,927 | 24.4% |
| XGBoost | 3,177 | 5,024 | 24.1% |

ETS wins with 21.7% MAPE. The traditional ARIMA and ML-based XGBoost perform similarly on this data, likely because the series has strong trend and seasonality that ETS handles natively.

<img src="output/figures/forecast_benchmark.svg" width="100%">

---

## Advanced Models

### 13. Can We Predict If a Contributor Will Come Back? (LSTM)

Section 10 predicts whether a *single PR* will be abandoned (AUC=0.916). This section asks a different question: will a *contributor* come back after their latest PR? Contributor behavior is *sequential*: a developer whose last 3 PRs were merged quickly is different from one with growing gaps and recent rejections. We trained an LSTM on the chronological sequence of each contributor's PRs.

The LSTM achieved **AUC=0.848** on the contributor-return task, capturing temporal patterns that flat-feature models cannot express: merge momentum, growing gaps between PRs, and rejection streaks.

<img src="output/figures/nn_contributor_return.svg" width="100%">

### 14. The Hidden Map of Open Source (Repo Embeddings)

We trained a neural network to learn 16-dimensional vector representations of repos, based solely on *who contributes to them*. Repos with overlapping contributor bases end up close in embedding space. We then projected these embeddings to 2D with t-SNE.

The result reveals that **language is a weak signal for repo similarity**. The real structure is social:

- `fastapi` (Python) and `airbnb/javascript` (JavaScript) are neighbors: same web-developer community
- `code-server` (TypeScript) and `rustdesk` (Rust) cluster together: remote-access tool users
- `karpathy/autoresearch` (Python) and `papers-we-love` (Shell) are close: ML research community
- 324 of 341 software repos form one massive supercluster of "generalist GitHub participants" who contribute across all languages

The contributor overlap graph exposes communities of practice that language tags hide.

<img src="output/figures/nn_repo_embeddings.svg" width="100%">

### 15. Optimizing the Health Index with a Genetic Algorithm

The Health Index weights (Section 6) were hand-picked. But which weights actually predict future repo growth? We used a Genetic Algorithm to search for weights that maximize Spearman correlation between today's health score and PR growth 6 months later.

| Component | Original weight | GA-optimized weight |
|-----------|----------------|-------------------|
| Response time | 25.0% | 0.0% |
| Bus factor | 20.0% | 26.8% |
| Diversity | 20.0% | 0.3% |
| Merge rate | 20.0% | 0.0% |
| Trend | 15.0% | 72.8% |

The GA nearly doubled the weight on response time (25.0% to 0%) and eliminated trend (15.0% to 72.8%). The optimized weights improved Spearman correlation from -0.0492 to 0.0669.

**The insight**: The single best predictor of whether a project will grow is **how fast it responds to contributions**. Not its current momentum, not its merge rate. Speed of response. This aligns with the contributor return finding (Section 13): contributors come back when they get fast feedback.

<img src="output/figures/ga_weights_comparison.svg" width="100%">
<img src="output/figures/ga_convergence.svg" width="100%">

### 16. Where Should You Submit Your First PR? (Decision Trees by Language)

We trained decision trees on first-timer PRs for each major programming language to predict which contributions get merged. The question is simple: **if you program in Python, Rust, Go, or any other language, where should you start contributing to maximize your chances of getting merged?**

Repos are ranked using the Wilson score lower bound, which penalizes small samples: a repo with 5/5 merges ranks lower than one with 200/300, because we need statistical confidence, not lucky streaks.

**Per-language model results:**

| Language | AUC | Top predictor | Actionable rule | Best repo |
|----------|-----|--------------|-----------------|-----------|
| Python | 0.784 | repo_merge_rate (75%) | repo_merge_rate > 0.33 AND repo_merge_rate > 0.72 AND pr_size <= 88.50 AND additions <= 5.50 -> likely merged (67%) | bregman-arie/devops-exercises (71% FT merge) |
| TypeScript | 0.745 | repo_merge_rate (54%) | repo_merge_rate > 0.59 AND additions <= 108.50 AND repo_total_prs <= 20586.50 AND repo_merge_rate > 0.80 -> likely merged (66%) | storybookjs/storybook (70% FT merge) |
| Go | 0.808 | repo_merge_rate (76%) | repo_merge_rate > 0.70 AND additions <= 44.50 AND hour <= 18.50 AND deletions > 0.50 -> likely merged (75%) | avelino/awesome-go (65% FT merge) |
| Rust | 0.725 | pr_size (46%) | pr_size <= 112.50 AND repo_merge_rate > 0.82 AND pr_size <= 26.50 AND repo_total_prs <= 6947.50 -> likely merged (79%) | tauri-apps/tauri (68% FT merge) |
| C++ | 0.823 | repo_merge_rate (66%) | repo_merge_rate > 0.73 AND pr_size <= 157.50 AND deletions > 0.50 AND additions <= 3.50 -> likely merged (68%) | opencv/opencv (56% FT merge) |
| Java | 0.888 | repo_merge_rate (88%) | repo_merge_rate > 0.29 AND deletions > 0.50 AND day_of_week <= 3.50 -> likely merged (59%) | iluwatar/java-design-patterns (44% FT merge) |
| C | 0.768 | repo_total_prs (98%) | repo_total_prs <= 629.00 -> likely not merged (38%) | ventoy/Ventoy (36% FT merge) |
| JavaScript | 0.747 | repo_merge_rate (62%) | repo_merge_rate > 0.69 AND repo_total_prs <= 2049.50 AND additions <= 4.50 AND hour <= 10.50 -> likely merged (86%) | Snailclimb/JavaGuide (80% FT merge) |
| C# | 0.714 | repo_total_prs (60%) | repo_total_prs > 4348.50 AND pr_size <= 131.50 AND changed_files > 1.50 -> likely merged (71%) | microsoft/PowerToys (55% FT merge) |

**If you write Python, start here** (ranked by Wilson score, min 20 FT PRs):

| Repo | First-timer merge rate | First-timer PRs | Wilson score |
|------|----------------------|-----------------|--------------|
| bregman-arie/devops-exercises | **71%** | 168 | 0.636 |
| langchain-ai/langchain | **58%** | 2494 | 0.564 |
| infiniflow/ragflow | **59%** | 291 | 0.530 |
| huggingface/transformers | **54%** | 2312 | 0.522 |
| EbookFoundation/free-programming-books | **54%** | 1361 | 0.513 |

**If you write TypeScript, start here** (ranked by Wilson score, min 20 FT PRs):

| Repo | First-timer merge rate | First-timer PRs | Wilson score |
|------|----------------------|-----------------|--------------|
| storybookjs/storybook | **70%** | 936 | 0.672 |
| Stirling-Tools/Stirling-PDF | **70%** | 170 | 0.627 |
| ant-design/ant-design | **64%** | 1246 | 0.609 |
| yangshun/tech-interview-handbook | **70%** | 102 | 0.601 |
| mermaid-js/mermaid | **64%** | 398 | 0.587 |

**If you write Go, start here** (ranked by Wilson score, min 20 FT PRs):

| Repo | First-timer merge rate | First-timer PRs | Wilson score |
|------|----------------------|-----------------|--------------|
| avelino/awesome-go | **65%** | 965 | 0.616 |
| jesseduffield/lazygit | **54%** | 237 | 0.481 |
| junegunn/fzf | **45%** | 166 | 0.378 |
| gin-gonic/gin | **40%** | 291 | 0.341 |
| ollama/ollama | **31%** | 636 | 0.277 |

**If you write Rust, start here** (ranked by Wilson score, min 20 FT PRs):

| Repo | First-timer merge rate | First-timer PRs | Wilson score |
|------|----------------------|-----------------|--------------|
| tauri-apps/tauri | **68%** | 214 | 0.617 |
| denoland/deno | **66%** | 330 | 0.602 |
| zed-industries/zed | **49%** | 992 | 0.459 |
| astral-sh/uv | **52%** | 213 | 0.454 |
| oven-sh/bun | **42%** | 480 | 0.371 |

**If you write C++, start here** (ranked by Wilson score, min 20 FT PRs):

| Repo | First-timer merge rate | First-timer PRs | Wilson score |
|------|----------------------|-----------------|--------------|
| opencv/opencv | **56%** | 1004 | 0.524 |
| electron/electron | **50%** | 686 | 0.464 |
| microsoft/terminal | **48%** | 283 | 0.420 |
| ggml-org/llama.cpp | **42%** | 1026 | 0.394 |
| nomic-ai/gpt4all | **33%** | 85 | 0.239 |

**If you write Java, start here** (ranked by Wilson score, min 20 FT PRs):

| Repo | First-timer merge rate | First-timer PRs | Wilson score |
|------|----------------------|-----------------|--------------|
| iluwatar/java-design-patterns | **44%** | 397 | 0.393 |
| spring-projects/spring-boot | **1%** | 948 | 0.008 |
| macrozheng/mall | **0%** | 67 | 0.000 |

**If you write C, start here** (ranked by Wilson score, min 20 FT PRs):

| Repo | First-timer merge rate | First-timer PRs | Wilson score |
|------|----------------------|-----------------|--------------|
| ventoy/Ventoy | **36%** | 86 | 0.267 |
| Genymobile/scrcpy | **3%** | 187 | 0.011 |

**If you write JavaScript, start here** (ranked by Wilson score, min 20 FT PRs):

| Repo | First-timer merge rate | First-timer PRs | Wilson score |
|------|----------------------|-----------------|--------------|
| Snailclimb/JavaGuide | **80%** | 477 | 0.760 |
| sveltejs/svelte | **52%** | 483 | 0.471 |
| louislam/uptime-kuma | **51%** | 439 | 0.466 |
| axios/axios | **38%** | 442 | 0.336 |
| trekhleb/javascript-algorithms | **32%** | 297 | 0.266 |

**If you write C#, start here** (ranked by Wilson score, min 20 FT PRs):

| Repo | First-timer merge rate | First-timer PRs | Wilson score |
|------|----------------------|-----------------|--------------|
| microsoft/PowerToys | **55%** | 314 | 0.499 |
| 2dust/v2rayN | **24%** | 106 | 0.165 |

<img src="output/figures/first_timer_tree_python.svg" width="100%">

<img src="output/figures/first_timer_recommendations.svg" width="100%">

**The decision tree distilled:** Rust is the only language where PR size matters more than the repo itself. Everywhere else, choosing the right repo is the single most important decision a first-timer can make. Check the repo's merged-vs-closed ratio before investing effort.
---

## What's Next

Dataset covers 580 repos spanning 2016-2026. Planned updates:

- **Comparative by organization type**: company-backed vs community vs foundation
- **Cross-correlation**: do repos that respond faster retain more contributors?

---

## Limitations

**GitHub PRs are not all of open source.** This study only captures projects that use GitHub Pull Requests as their contribution mechanism. Some of the most important open-source projects in history are invisible to this analysis:

- **Linux kernel** (torvalds/linux): 0 PRs on GitHub. Development happens via email patches on the Linux Kernel Mailing List (LKML). Reviews are done by email. Merges are `git pull` commands, not GitHub PRs. The GitHub repo is a read-only mirror.
- **Git** itself, **FFmpeg**, **QEMU**, and other foundational projects use similar mailing-list workflows.

A study based on GitHub PRs inherently measures "GitHub-native open source," not all open source. Projects that predate GitHub or chose to resist its workflow are excluded by design. This biases the dataset toward newer, web-era projects and away from systems-level infrastructure.

**Stars as a selection criterion.** We selected repos by GitHub stars, which is the best available proxy for "widely known open-source projects" but not a perfect one. Stars measure developer audience: how many people found a project interesting enough to bookmark. This correlates with but is not identical to software importance. ~30% of the top-200 by stars were curated lists and educational resources, not software projects. We filtered these out, but the remaining dataset is still biased toward projects with high visibility rather than critical infrastructure that runs quietly (e.g., OpenSSL has far fewer stars than many tutorial repos).

**Snapshot in time.** This analysis was run in June 2026. The AI-era findings (especially 2025-2026 data) may reflect early adoption patterns that will stabilize as tools mature.

---

## Reproducibility

### Setup

```bash
git clone https://github.com/EdenRochmanSharabi/oss-pulse.git
cd oss-pulse
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### Extract data

```bash
make extract-search            # Download repos via GitHub Search API
make extract-mega              # Year-chunked download for mega-repos
make extract-status            # Check extraction progress
```

### Run the full pipeline

```bash
make pipeline                  # combine + transform + analyze + figures + notebook
```

Or step by step:

```bash
make combine                   # Merge per-repo parquets into single dataset
make transform                 # Clean, classify, feature engineering
make analyze                   # Core analysis (seasonal, forecast, health, etc.)
make analyze-advanced          # LSTM contributor return + repo embeddings
make figures                   # Generate dashboard figures
make notebook                  # Execute narrative notebook
```

### Run tests

```bash
make test                      # 134 tests, 85% coverage
make lint                      # ruff check + format
make typecheck                 # mypy --strict
```

---

## Project Structure

```
src/oss_pulse/
  extract/       GitHub Search API, GraphQL API, BigQuery, repo discovery, status monitor
  transform/     Cleaning, bot detection, author classification, feature engineering
  analyze/       Seasonal, forecasting, health index, abandonment, funnel,
                 changepoint, comparative, LSTM contributor return, repo embeddings
  visualize/     matplotlib/seaborn plots, heatmaps, survival curves, dashboards

scripts/
  download_search_api.py     Main extractor (Search API with date filtering)
  download_mega_repos.py     Year-chunked extractor for mega-repos
  retry_until_done.py        Persistent retry loop

notebooks/
  00_narrative_analysis.ipynb   Main analysis (start here)

tests/
  fixtures/      Real data sample for integration tests
  test_*.py      134 tests covering all modules

docs/
  process_log.md Project diary with challenges and decisions

config/
  repos.csv      Repo classification
  events.csv     AI tool releases, Hacktoberfest dates, conferences
```

## License

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
