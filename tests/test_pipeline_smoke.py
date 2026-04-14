from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rossmann_mlops.train import train_pipeline


def test_train_pipeline_smoke(tmp_path: Path) -> None:
    train_df = pd.DataFrame(
        {
            "Store": [1, 1, 1, 2, 2, 2],
            "DayOfWeek": [1, 2, 3, 1, 2, 3],
            "Date": [
                "2015-05-20",
                "2015-05-21",
                "2015-06-20",
                "2015-05-20",
                "2015-05-21",
                "2015-06-20",
            ],
            "Open": [1, 1, 1, 1, 1, 1],
            "Promo": [0, 1, 1, 0, 1, 1],
            "StateHoliday": ["0", "0", "0", "0", "0", "0"],
            "SchoolHoliday": [0, 0, 1, 0, 0, 1],
            "Sales": [5000, 5200, 5500, 3000, 3200, 3600],
        }
    )
    store_df = pd.DataFrame(
        {
            "Store": [1, 2],
            "StoreType": ["a", "b"],
            "Assortment": ["a", "a"],
            "CompetitionDistance": [100.0, 200.0],
            "Promo2": [0, 1],
        }
    )

    train_path = tmp_path / "train.csv"
    store_path = tmp_path / "store.csv"
    model_path = tmp_path / "models" / "model.joblib"
    metrics_path = tmp_path / "metrics" / "metrics.json"
    config_path = tmp_path / "config.yaml"

    train_df.to_csv(train_path, index=False)
    store_df.to_csv(store_path, index=False)

    config = {
        "paths": {
            "train_data": str(train_path),
            "store_data": str(store_path),
            "model_file": str(model_path),
            "metrics_file": str(metrics_path),
        },
        "training": {
            "validation_start_date": "2015-06-01",
            "n_estimators": 10,
            "random_state": 42,
            "n_jobs": 1,
        },
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = train_pipeline(config)

    assert model_path.exists()
    assert metrics_path.exists()
    assert result["metrics"]["rmse"] >= 0

    saved_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "mae" in saved_metrics
    assert "r2" in saved_metrics
