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
        self.expected_columns = self._get_expected_columns(self.model)

        try:
            self.store_dw_promo_mapping = joblib.load(resolved_artifacts_dir / "store_dw_promo_mapping.pkl")
            self.month_mapping = joblib.load(resolved_artifacts_dir / "month_mapping.pkl")
            self.global_mean = joblib.load(resolved_artifacts_dir / "global_mean_sales.pkl")
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Missing mapping file in {resolved_artifacts_dir}: {exc}. Run training pipeline first."
            )

        # Advanced mappings are produced by newer training logic.
        self.store_median_mapping = self._safe_load_mapping(resolved_artifacts_dir / "store_median_mapping.pkl")
        self.store_dow_stats_mapping = self._safe_load_mapping(resolved_artifacts_dir / "store_dow_stats_mapping.pkl")
        self.promo_lift_mapping = self._safe_load_mapping(resolved_artifacts_dir / "promo_lift_mapping.pkl")

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
    def _safe_load_mapping(path: Path) -> Any | None:
        try:
            return joblib.load(path)
        except FileNotFoundError:
            return None

    @staticmethod
    def _get_expected_columns(model: Any) -> list[str] | None:
        if hasattr(model, "feature_names_in_"):
            return list(model.feature_names_in_)
        if hasattr(model, "get_booster"):
            try:
                booster = model.get_booster()
                if booster is not None and booster.feature_names:
                    return list(booster.feature_names)
            except Exception:
                return None
        return None

    def _requires_advanced_features(self) -> bool:
        advanced_columns = {
            "Store_Avg_Sales",
            "Store_DoW_Median",
            "Promo_Lift",
            "day_sin",
            "day_cos",
            "week_sin",
            "week_cos",
        }
        if not self.expected_columns:
            return False
        return any(column in advanced_columns for column in self.expected_columns)

    def _validate_advanced_mapping_availability(self) -> None:
        if not self._requires_advanced_features():
            return

        if self.store_median_mapping is None or self.store_dow_stats_mapping is None or self.promo_lift_mapping is None:
            raise FileNotFoundError(
                "Model expects advanced feature mappings but one or more files are missing: "
                "store_median_mapping.pkl, store_dow_stats_mapping.pkl, promo_lift_mapping.pkl. "
                "Retrain model to regenerate artifacts."
            )

    def _apply_advanced_mappings(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self._requires_advanced_features():
            return frame

        self._validate_advanced_mapping_availability()

        data = frame.copy()

        # Mirror training-time advanced features.
        data["Store_Avg_Sales"] = data["Store"].map(self.store_median_mapping)

        store_dow_stats = self.store_dow_stats_mapping
        if isinstance(store_dow_stats, pd.DataFrame):
            data = data.merge(store_dow_stats, on=["Store", "DayOfWeek"], how="left")
        else:
            data["Store_DoW_Median"] = np.nan

        data["Promo_Lift"] = data["Store"].map(self.promo_lift_mapping)

        day_of_week = pd.to_numeric(data["DayOfWeek"], errors="coerce")
        week_of_year = pd.to_numeric(data["WeekOfYear"], errors="coerce")
        data["day_sin"] = np.sin(2 * np.pi * day_of_week / 7)
        data["day_cos"] = np.cos(2 * np.pi * day_of_week / 7)
        data["week_sin"] = np.sin(2 * np.pi * week_of_year / 52)
        data["week_cos"] = np.cos(2 * np.pi * week_of_year / 52)

        global_sales_median = float(np.nanmedian(list(self.store_median_mapping.values())))
        data["Store_Avg_Sales"] = pd.to_numeric(data["Store_Avg_Sales"], errors="coerce").fillna(global_sales_median)
        data["Store_DoW_Median"] = pd.to_numeric(data["Store_DoW_Median"], errors="coerce").fillna(data["Store_Avg_Sales"])
        data["Promo_Lift"] = pd.to_numeric(data["Promo_Lift"], errors="coerce").fillna(1.0)

        data["day_sin"] = pd.to_numeric(data["day_sin"], errors="coerce").fillna(0.0)
        data["day_cos"] = pd.to_numeric(data["day_cos"], errors="coerce").fillna(0.0)
        data["week_sin"] = pd.to_numeric(data["week_sin"], errors="coerce").fillna(0.0)
        data["week_cos"] = pd.to_numeric(data["week_cos"], errors="coerce").fillna(0.0)

        return data

    @staticmethod
    def _align_model_columns(frame: pd.DataFrame, model: Any) -> pd.DataFrame:
        expected_columns = Predictor._get_expected_columns(model)

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
        features = self._apply_advanced_mappings(features)

        # Restore Open (build_features drops it but the trained model requires it)
        if open_values is not None:
            features["Open"] = open_values

        cols_to_drop = ["Sales", "Sales_log", "Customers", "Promo2", "Date", "Id"]
        features = features.drop(columns=[col for col in cols_to_drop if col in features.columns], errors="ignore")
        features = self._align_model_columns(features, self.model)

        predictions_log = self.model.predict(features)
        predictions = np.exp(predictions_log)
        predictions = np.maximum(predictions, 0)
        # Stores that are closed (Open=0) always have 0 sales
        if open_values is not None:
            predictions[open_values == 0] = 0.0
        return [round(float(value), 2) for value in predictions]
