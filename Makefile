.PHONY: setup fetch build train backtest scan test lint pipeline

setup:
	pip install -e ".[dev]"

fetch:
	python -m pipeline.fetch_data --all

build:
	python -m pipeline.build_dataset

train:
	python -m models.dixon_coles --train

backtest:
	python -m backtest.run_walkforward

scan:
	python scan_now.py

test:
	pytest tests/ -v

lint:
	ruff check .

pipeline: fetch build train backtest
