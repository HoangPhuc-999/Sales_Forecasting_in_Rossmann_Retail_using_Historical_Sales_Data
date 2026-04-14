from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from rossmann_mlops.config import resolve_path
from rossmann_mlops.features import build_features, merge_store_data


class PredictionInputError(ValueError):
    pass


class Predictor:
    def __init__(self, model_path: str | Path, store_data_path: str | Path) -> None:
        model_file = resolve_path(model_path)
        store_file = resolve_path(store_data_path)

        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}. Run training pipeline first.")
        if not store_file.exists():
            raise FileNotFoundError(
                f"Store data not found: {store_file}. Run `dvc pull data/store.csv` before serving API."
            )

        self.model = joblib.load(model_file)
        self.store_df = pd.read_csv(store_file)

    def predict(self, records: list[dict[str, Any]]) -> list[float]:
        if not records:
            raise PredictionInputError("records must contain at least one item")

        frame = pd.DataFrame(records)
        required_columns = ["Store", "DayOfWeek", "Date", "Open", "Promo", "StateHoliday", "SchoolHoliday"]
        missing = [column for column in required_columns if column not in frame.columns]
        if missing:
            raise PredictionInputError(f"Missing required fields: {missing}")

        merged = merge_store_data(frame, self.store_df)
        features = build_features(merged)
        predictions = self.model.predict(features)
        return [float(value) for value in predictions]
