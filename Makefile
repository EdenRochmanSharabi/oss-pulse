.PHONY: install extract extract-search extract-mega extract-synthetic \
       combine transform analyze analyze-advanced figures readme \
       demo test lint typecheck clean all

# ── Setup ──────────────────────────────────────────────
install:
	pip install -e ".[dev]"

# ── Extract ────────────────────────────────────────────
extract: extract-search
	@echo "Extraction complete. Run 'make combine' next."

extract-search:
	python scripts/download_search_api.py

extract-mega:
	python scripts/download_mega_repos.py

extract-retry:
	python scripts/retry_until_done.py

extract-synthetic:
	python -m oss_pulse.extract.synthetic

extract-status:
	python -m oss_pulse.extract.status

# ── Combine raw repos into single dataset ──────────────
combine:
	python -c "\
	import os, pandas as pd; \
	dfs = [pd.read_parquet(f'data/raw/repos/{f}') for f in os.listdir('data/raw/repos/') if f.endswith('.parquet')]; \
	dfs = [d for d in dfs if len(d) > 0]; \
	raw = pd.concat(dfs, ignore_index=True); \
	raw.to_parquet('data/raw/pr_events.parquet', index=False); \
	print(f'Combined: {len(raw):,} PRs, {raw[\"repo_name\"].nunique()} repos')"

# ── Process (combine + clean + classify + filter + features) ──
process:
	python scripts/process_pipeline.py

# ── Transform (individual steps, for debugging) ───────
transform:
	python -m oss_pulse.transform.clean
	python -m oss_pulse.transform.classify
	python -m oss_pulse.transform.features

# ── Analyze (core) ─────────────────────────────────────
analyze:
	python -m oss_pulse.analyze.seasonal
	python -m oss_pulse.analyze.forecast
	python -m oss_pulse.analyze.response_time
	python -m oss_pulse.analyze.comparative
	python -m oss_pulse.analyze.changepoint
	python -m oss_pulse.analyze.health_index
	python -m oss_pulse.analyze.abandonment
	python -m oss_pulse.analyze.funnel
	python -m oss_pulse.analyze.productivity
	python -m oss_pulse.analyze.first_timer_tree

# ── Analyze (advanced: NN, GA) ─────────────────────────
analyze-advanced:
	python -m oss_pulse.analyze.contributor_return
	python -m oss_pulse.analyze.repo_embeddings

# ── Figures ────────────────────────────────────────────
figures:
	python -m oss_pulse.visualize.dashboard

# ── Notebook ───────────────────────────────────────────
notebook:
	cd notebooks && jupyter nbconvert --to notebook --execute \
		00_narrative_analysis.ipynb \
		--output 00_narrative_analysis_executed.ipynb \
		--ExecutePreprocessor.timeout=600

# ── Generate README from stats.json ───────────────────
readme:
	python scripts/generate_readme.py

# ── Full pipeline (real data) ──────────────────────────
pipeline: process analyze analyze-advanced figures notebook readme
	@echo "Full pipeline complete."

# ── Demo (synthetic data) ─────────────────────────────
demo: extract-synthetic transform analyze figures
	@echo "Demo pipeline complete. Check output/figures/"

# ── Quality ────────────────────────────────────────────
test:
	python -m pytest

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

typecheck:
	mypy src/

# ── Cleanup ────────────────────────────────────────────
clean:
	rm -rf data/processed/*.parquet
	rm -rf output/figures/*.png
	rm -rf output/figures/*.svg

# ── Everything ─────────────────────────────────────────
all: install pipeline test lint typecheck
