from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.rossmann_mlops.config import load_config, resolve_path
from src.rossmann_mlops.features import build_features, merge_store_data


class TrainingError(ValueError):
    """Raised when training pipeline input or split is invalid."""


STORE_DEFAULTS: dict[str, Any] = {
    "StoreType": "a",
    "Assortment": "a",
    "CompetitionDistance": 0.0,
    "Promo2": 0,
    "Promo2SinceWeek": 0,
    "Promo2SinceYear": 0,
    "CompetitionOpenSinceMonth": 0,
    "CompetitionOpenSinceYear": 0,
    "PromoInterval": "None",
}

REQUIRED_TRAIN_COLUMNS = [
    "Store",
    "DayOfWeek",
    "Date",
    "Open",
    "Promo",
    "StateHoliday",
    "SchoolHoliday",
    "Sales",
]


def _ensure_columns(frame: pd.DataFrame, defaults: dict[str, Any]) -> pd.DataFrame:
    updated = frame.copy()
    for column, default_value in defaults.items():
        if column not in updated.columns:
            updated[column] = default_value
    return updated


def _ensure_required_columns(frame: pd.DataFrame, required_columns: list[str], source_name: str) -> None:
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise TrainingError(f"Missing required columns in {source_name}: {missing}")


def _prepare_training_frame(train_df: pd.DataFrame, store_df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    _ensure_required_columns(train_df, REQUIRED_TRAIN_COLUMNS, "training data")

    prepared_train = train_df.copy()
    prepared_train["Date"] = pd.to_datetime(prepared_train["Date"], errors="coerce")
    prepared_train = prepared_train.dropna(subset=["Date"])

    prepared_train["StateHoliday"] = prepared_train["StateHoliday"].astype(str)
    prepared_store = _ensure_columns(store_df, STORE_DEFAULTS)

    merged = merge_store_data(prepared_train, prepared_store)
    merged = _ensure_columns(merged, STORE_DEFAULTS)
    features = build_features(merged)

    target = np.log1p(merged["Sales"].astype(float).clip(lower=0.0).to_numpy())
    return features, target


def _time_split_mask(date_series: pd.Series, validation_start_date: str | None) -> np.ndarray:
    parsed = pd.to_datetime(date_series, errors="coerce")
    if validation_start_date:
        start = pd.to_datetime(validation_start_date, errors="coerce")
        if pd.notna(start):
            mask = (parsed >= start).fillna(False).to_numpy()
            if mask.any() and (~mask).any():
                return mask

    length = len(parsed)
    fallback = np.zeros(length, dtype=bool)
    if length > 1:
        split_point = max(1, int(length * 0.8))
        if split_point >= length:
            split_point = length - 1
        fallback[split_point:] = True
    return fallback


def _load_training_data(paths: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = resolve_path(paths["train_data"])
    store_path = resolve_path(paths["store_data"])

    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")
    if not store_path.exists():
        raise FileNotFoundError(f"Store data not found: {store_path}")

    return pd.read_csv(train_path), pd.read_csv(store_path)


def train_pipeline(config: dict[str, Any]) -> dict[str, Any]:
    paths = config.get("paths", {})
    training = config.get("training", {})

    train_path = resolve_path(paths["train_data"])
    model_path = resolve_path(paths["model_file"])
    metrics_path = resolve_path(paths["metrics_file"])

    train_df, store_df = _load_training_data(paths)

    features, target = _prepare_training_frame(train_df, store_df)
    if len(features) < 2:
        raise TrainingError("Training data must contain at least 2 valid rows after preprocessing")

    validation_mask = _time_split_mask(train_df.loc[features.index, "Date"], training.get("validation_start_date"))
    if not validation_mask.any() or (~validation_mask).sum() == 0:
        raise TrainingError("Unable to create a valid train/validation split")

    x_train = features.loc[~validation_mask]
    x_val = features.loc[validation_mask]
    y_train = target[~validation_mask]
    y_val = target[validation_mask]

    model = RandomForestRegressor(
        n_estimators=int(training.get("n_estimators", 300)),
        random_state=int(training.get("random_state", 42)),
        n_jobs=int(training.get("n_jobs", -1)),
    )
    model.fit(x_train, y_train)

    pred_val = np.expm1(model.predict(x_val))
    true_val = np.expm1(y_val)

    rmse = float(np.sqrt(mean_squared_error(true_val, pred_val)))
    mae = float(mean_absolute_error(true_val, pred_val))
    r2 = float(r2_score(true_val, pred_val))

    metrics = {"rmse": rmse, "mae": mae, "r2": r2}

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return {
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "metrics": metrics,
        "n_train": int((~validation_mask).sum()),
        "n_validation": int(validation_mask.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Rossmann model from YAML config")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    config = load_config(args.config)
    result = train_pipeline(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
