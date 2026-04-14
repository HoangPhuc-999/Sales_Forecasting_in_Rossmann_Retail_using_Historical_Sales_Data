# Rossmann Sales Forecasting - MLOps Pipeline

This repository contains an end-to-end MLOps baseline for Rossmann sales forecasting:

- Train -> evaluate -> save model artifacts
- Inference API with FastAPI
- Automated pipeline script
- Monitoring and robustness checks
- CI workflow with GitHub Actions
- YAML-based centralized configuration
- Reproducibility with fixed seed, requirements file, version logs

## Project Structure

```text
.
|-- app/
|   `-- main.py
|-- configs/
|   `-- config.yaml
|-- scripts/
|   |-- run_pipeline.py
|   |-- monitor.py
|   |-- retrain.py
|   `-- run_pipeline.bat
|-- requirements.txt
|-- src/rossmann_mlops/
|   |-- config.py
|   |-- features.py
|   |-- predict.py
|   `-- train.py
|-- tests/
|   `-- test_pipeline_smoke.py
`-- .github/workflows/ci.yml
```

## Setup

1. Install dependencies

```bash
pip install -e .
pip install -e .[dev]
```

2. Pull data from DVC (required)

```bash
dvc pull data/train.csv data/store.csv
```

## Run Training Pipeline

```bash
python scripts/run_pipeline.py --config configs/config.yaml
```

Or on Windows:

```bat
scripts\run_pipeline.bat
```

Output artifacts:

- Model: `artifacts/models/rossmann_model.joblib`
- Metrics: `artifacts/metrics/metrics.json`

## Run API

Start service:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

Prediction example:

```bash
curl -X POST http://localhost:8000/predict \
	-H "Content-Type: application/json" \
	-d '{
		"records": [
			{
				"Store": 1,
				"DayOfWeek": 5,
				"Date": "2015-09-17",
				"Open": 1,
				"Promo": 1,
				"StateHoliday": "0",
				"SchoolHoliday": 0
			}
		]
	}'
```

## Monitoring & Robustness

The monitoring layer covers:

- data drift check with PSI
- performance logging
- alert creation when model quality is low
- retrain script
- input error handling in API

Run monitoring on a new CSV:

```bash
python scripts/monitor.py --current data/test.csv
```

This reads the reference data from `data/train.csv`, loads `data/store.csv`, compares features, and writes logs into `logs/`.

Run retraining:

```bash
python scripts/retrain.py --config configs/config.yaml
```

Output logs:

- `logs/monitoring_report.jsonl`
- `logs/performance.jsonl`
- `logs/alerts.jsonl`

## Reproducibility

This project is set up to support reproducible runs:

- `requirements.txt` lists the main dependencies for local setup.
- `configs/config.yaml` keeps the training seed and experiment log path in one place.
- `src/rossmann_mlops/train.py` fixes the global seed before training.
- Each training run writes an experiment record to `logs/experiments.jsonl`.
- The experiment record stores the model hash and DVC data hashes from `data/*.dvc`.

If you want to re-run the exact same setup, keep the same data version from DVC and the same seed value.

## Run Tests

```bash
pytest -q
```

## CI

GitHub Actions workflow file:

- `.github/workflows/ci.yml`

Pipeline actions:

- install dependencies
- run test suite

## Rubric Mapping (MLOps Pipeline)

- Pipeline train -> eval -> save model: `src/rossmann_mlops/train.py`
- API (FastAPI): `app/main.py`
- GitHub Actions CI: `.github/workflows/ci.yml`
- Automation script: `scripts/run_pipeline.py` and `scripts/run_pipeline.bat`
- Config file: `configs/config.yaml`
- Monitoring scripts: `scripts/monitor.py` and `scripts/retrain.py`
- Reproducibility: `requirements.txt`, `configs/config.yaml`, `logs/experiments.jsonl`
