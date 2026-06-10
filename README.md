# oss-pulse

Time-series analysis of Pull Requests across the top open-source GitHub projects.

This study examines 10 years of PR activity (2016-2026) in the 200 most-starred repositories on GitHub, uncovering seasonal patterns, forecasting trends, measuring project health, and detecting the structural impact of AI coding tools on open-source contribution dynamics.

<!-- FINDINGS -->
## Key Findings

> Findings will be populated after running the full analysis with real data. Run `make demo` for a preview using synthetic data.

<!-- /FINDINGS -->

## Analysis Modules

| Module | Description |
|--------|-------------|
| **Seasonal Patterns** | STL decomposition, stationarity tests (ADF/KPSS), autocorrelation analysis |
| **Forecasting** | Benchmark of ARIMA, Prophet, ETS, and XGBoost on PR volume prediction |
| **Response Time** | Time-to-merge and time-to-first-review distributions and trends |
| **Comparative** | Cross-ecosystem, cross-org-type, and cross-size statistical comparisons |
| **AI Effect** | Automatic changepoint detection aligned with AI tool release dates |
| **Health Index** | Composite score ranking projects by responsiveness, diversity, trend, and bus factor |
| **Abandonment Prediction** | Survival analysis (Kaplan-Meier, Cox PH) and ML classifiers for project decline |
| **Contributor Funnel** | Retention analysis from first PR through regular contributor stages |
| **Micro-patterns** | Hour-of-day and day-of-week activity heatmaps |

## Setup

### Prerequisites

- Python 3.11+
- (Optional) Google Cloud project with BigQuery API enabled for real data

### Installation

```bash
git clone https://github.com/EdenRochmanSharabi/oss-pulse.git
cd oss-pulse
pip install -e ".[dev]"
```

For BigQuery access (real data):
```bash
pip install -e ".[dev,bigquery]"
```

### BigQuery Configuration (optional)

1. Create a project at [console.cloud.google.com](https://console.cloud.google.com)
2. Enable the BigQuery API
3. Create a service account and download the JSON key
4. Place the key at `data/credentials/service_account.json`

## Usage

### Quick demo with synthetic data

```bash
make demo
```

This generates realistic synthetic PR data, runs the full ETL and analysis pipeline, and produces figures in `output/figures/`.

### Full analysis with real data

```bash
make extract      # Pull data from BigQuery (requires credentials)
make transform    # Clean, classify, and engineer features
make analyze      # Run all analysis modules
make figures      # Generate publication-quality plots
```

### Development

```bash
make test         # Run tests (coverage > 80%)
make lint         # ruff check + format
make typecheck    # mypy --strict
make all          # install + demo + test + lint + typecheck
```

## Project Structure

```
src/oss_pulse/
├── extract/       BigQuery client, SQL queries, synthetic data generator
├── transform/     Cleaning, bot detection, author classification, feature engineering
├── analyze/       Time-series decomposition, forecasting, survival analysis, health index
└── visualize/     Academic-clean matplotlib plots, heatmaps, dashboards

notebooks/         Narrative analysis (01-10), one per module
tests/             pytest suite with synthetic data fixtures
config/            Repo classification and external events timeline
```

## Methodology

**Data source:** [GH Archive](https://www.gharchive.org/) via Google BigQuery (`githubarchive.day.*` public dataset).

**PR classification:** Each PR is categorized by outcome (merged / closed without merge / abandoned after 90 days of inactivity) and author type (first-timer / regular / maintainer / bot).

**Health Index:** Composite score (0-100) based on five weighted components:
- Median response time to PRs (25%)
- Merge rate excluding bots (20%)
- Contributor diversity per month (20%)
- Activity trend slope (15%)
- Bus factor via Gini coefficient (20%)

**Forecasting:** Four models benchmarked on 80/20 temporal split with MAE, RMSE, and MAPE.

**Changepoint detection:** PELT algorithm (ruptures library) identifies structural breaks, then cross-referenced with AI tool release dates within a 4-week window.

**Abandonment prediction:** Dual approach using Kaplan-Meier / Cox proportional hazards survival analysis alongside Random Forest and XGBoost classifiers.

## Tech Stack

pandas, statsmodels, scikit-learn, XGBoost, lifelines, Prophet, ruptures, matplotlib, seaborn

## License

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
