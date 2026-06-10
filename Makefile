.PHONY: install extract transform analyze demo test lint typecheck figures clean all

install:
	pip install -e ".[dev,bigquery]"

extract:
	python -m oss_pulse.extract.bigquery

extract-synthetic:
	python -m oss_pulse.extract.synthetic

transform:
	python -m oss_pulse.transform.clean
	python -m oss_pulse.transform.classify
	python -m oss_pulse.transform.features

analyze:
	python -m oss_pulse.analyze.seasonal
	python -m oss_pulse.analyze.forecast
	python -m oss_pulse.analyze.response_time
	python -m oss_pulse.analyze.comparative
	python -m oss_pulse.analyze.changepoint
	python -m oss_pulse.analyze.health_index
	python -m oss_pulse.analyze.abandonment
	python -m oss_pulse.analyze.funnel

figures:
	python -m oss_pulse.visualize.dashboard

demo: extract-synthetic transform analyze figures
	@echo "Demo pipeline complete. Check output/figures/"

test:
	python -m pytest

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

typecheck:
	mypy src/

clean:
	rm -rf data/processed/*.parquet
	rm -rf output/figures/*.png

all: install demo test lint typecheck
