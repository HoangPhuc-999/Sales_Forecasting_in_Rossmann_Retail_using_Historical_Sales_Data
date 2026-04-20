from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from rossmann_mlops.config import resolve_path
from rossmann_mlops.features import build_features, merge_store_data


class PredictionInputError(ValueError):
    pass


REQUIRED_COLUMNS = ["Store", "DayOfWeek", "Date", "Open", "Promo", "StateHoliday", "SchoolHoliday"]


class Predictor:
    def __init__(
        self,
        model_path: str | Path,
        store_data_path: str | Path,
        artifacts_dir: str | Path | None = None,
    ) -> None:
        model_file = resolve_path(model_path)
        store_file = resolve_path(store_data_path)
        resolved_artifacts_dir = resolve_path(artifacts_dir) if artifacts_dir is not None else model_file.parent

        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}. Run training pipeline first.")
        if not store_file.exists():
            raise FileNotFoundError(
                f"Store data not found: {store_file}. Run 'dvc pull data/raw/store.csv' before serving API."
            )

        self.model = joblib.load(model_file)
        self.store_df = pd.read_csv(store_file)

        try:
            self.store_dw_promo_mapping = joblib.load(resolved_artifacts_dir / "store_dw_promo_mapping.pkl")
            self.month_mapping = joblib.load(resolved_artifacts_dir / "month_mapping.pkl")
            self.global_mean = joblib.load(resolved_artifacts_dir / "global_mean_sales.pkl")
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Missing mapping file in {resolved_artifacts_dir}: {exc}. Run training pipeline first."
            )

    @staticmethod
    def _validate_request_frame(frame: pd.DataFrame) -> None:
        missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
        if missing:
            raise PredictionInputError(f"Missing required fields: {missing}")

    def _apply_mappings(self, frame: pd.DataFrame) -> pd.DataFrame:
        data = frame.merge(self.store_dw_promo_mapping, on=["Store", "DayOfWeek", "Promo"], how="left")
        data = data.merge(self.month_mapping, on="Month", how="left")

        data["Store_DW_Promo_Avg"] = data["Store_DW_Promo_Avg"].fillna(self.global_mean)
        data["Month_Avg_Sales"] = data["Month_Avg_Sales"].fillna(self.global_mean)
        return data

    @staticmethod
    def _align_model_columns(frame: pd.DataFrame, model: Any) -> pd.DataFrame:
        expected_columns: list[str] | None = None
        if hasattr(model, "feature_names_in_"):
            expected_columns = list(model.feature_names_in_)
        elif hasattr(model, "get_booster"):
            try:
                booster = model.get_booster()
                if booster is not None and booster.feature_names:
                    expected_columns = list(booster.feature_names)
            except Exception:
                expected_columns = None

        if not expected_columns:
            return frame

        aligned = frame.copy()
        for column in expected_columns:
            if column not in aligned.columns:
                aligned[column] = 0
        return aligned[expected_columns]

    def predict(self, records: list[dict[str, Any]]) -> list[float]:
        if not records:
            raise PredictionInputError("records must contain at least one item")

        frame = pd.DataFrame(records)
        self._validate_request_frame(frame)

        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
        if frame["Date"].isna().any():
            raise PredictionInputError("Invalid Date value in request records")

        merged = merge_store_data(frame, self.store_df)

        # Save Open before build_features drops it; model was trained with Open included
        open_values = merged["Open"].values.copy() if "Open" in merged.columns else None

        features = build_features(merged)
        features = self._apply_mappings(features)

        # Restore Open (build_features drops it but the trained model requires it)
        if open_values is not None:
            features["Open"] = open_values

        cols_to_drop = ["Sales", "Sales_log", "Customers", "Month", "Promo2", "Date", "Id"]
        features = features.drop(columns=[col for col in cols_to_drop if col in features.columns], errors="ignore")
        features = self._align_model_columns(features, self.model)

        predictions_log = self.model.predict(features)
        predictions = np.exp(predictions_log)
        predictions = np.maximum(predictions, 0)
        return [round(float(value), 2) for value in predictions]
