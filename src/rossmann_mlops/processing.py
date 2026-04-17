from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.rossmann_mlops.config import load_config, resolve_path


class ProcessingError(ValueError):
    """Raised when raw Rossmann data cannot be processed safely."""


DEFAULT_PATHS: dict[str, str] = {
    "store_raw": "data/raw/store.csv",
    "train_raw": "data/raw/train.csv",
    "test_raw": "data/raw/test.csv",
    "train_processed": "data/processed/train_processed.csv",
    "test_processed": "data/processed/test_processed.csv",
}

BASE_REQUIRED_COLUMNS = ["Store", "DayOfWeek", "Date", "Open", "Promo", "StateHoliday", "SchoolHoliday"]
TRAIN_ONLY_REQUIRED_COLUMNS = ["Sales"]
STORE_REQUIRED_COLUMNS = [
    "Store",
    "StoreType",
    "Assortment",
    "CompetitionDistance",
    "Promo2",
    "Promo2SinceWeek",
    "Promo2SinceYear",
    "CompetitionOpenSinceMonth",
    "CompetitionOpenSinceYear",
    "PromoInterval",
]


def _ensure_required_columns(frame: pd.DataFrame, required_columns: list[str], source_name: str) -> None:
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ProcessingError(f"Missing required columns in {source_name}: {missing}")


def load_data(store_path: str | Path, train_path: str | Path, test_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw store, train, and test datasets."""
    resolved_store = resolve_path(store_path)
    resolved_train = resolve_path(train_path)
    resolved_test = resolve_path(test_path)

    for source_name, source_path in [
        ("store data", resolved_store),
        ("train data", resolved_train),
        ("test data", resolved_test),
    ]:
        if not source_path.exists():
            raise FileNotFoundError(f"{source_name.capitalize()} not found: {source_path}")

    store_df = pd.read_csv(resolved_store)
    train_df = pd.read_csv(resolved_train, dtype={"StateHoliday": str})
    test_df = pd.read_csv(resolved_test, dtype={"StateHoliday": str})

    _ensure_required_columns(store_df, STORE_REQUIRED_COLUMNS, "store data")
    _ensure_required_columns(train_df, BASE_REQUIRED_COLUMNS + TRAIN_ONLY_REQUIRED_COLUMNS, "train data")
    _ensure_required_columns(test_df, BASE_REQUIRED_COLUMNS, "test data")

    return store_df, train_df, test_df


def merge_data(train_df: pd.DataFrame, test_df: pd.DataFrame, store_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge train/test rows with store metadata."""
    train_merged = pd.merge(train_df, store_df, on="Store", how="left")
    test_merged = pd.merge(test_df, store_df, on="Store", how="left")
    return train_merged, test_merged


def handle_outliers(train_df: pd.DataFrame) -> pd.DataFrame:
    """Apply target transformation used downstream by exploratory workflows."""
    processed = train_df.copy()
    processed["Sales_log"] = np.log1p(processed["Sales"].clip(lower=0))
    return processed


def _normalize_common_columns(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["Date"]).copy()

    data["CompetitionDistance"] = pd.to_numeric(data["CompetitionDistance"], errors="coerce").fillna(0)
    data["CompetitionDistance_log"] = np.log1p(data["CompetitionDistance"])

    fill_zero_cols = [
        "Promo2SinceWeek",
        "Promo2SinceYear",
        "CompetitionOpenSinceMonth",
        "CompetitionOpenSinceYear",
    ]
    for column in fill_zero_cols:
        data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0)

    data["PromoInterval"] = data["PromoInterval"].fillna("None")
    data["StateHoliday"] = data["StateHoliday"].astype(str)
    return data


def preprocess_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean merged train/test frames and enforce train filtering rules."""
    cleaned_train = _normalize_common_columns(train_df)
    cleaned_test = _normalize_common_columns(test_df)

    cleaned_train = cleaned_train[(cleaned_train["Open"] != 0) & (cleaned_train["Sales"] > 0)].copy()
    cleaned_train = handle_outliers(cleaned_train)
    return cleaned_train, cleaned_test


def _resolve_processing_paths(config: dict[str, Any]) -> dict[str, Path]:
    config_paths = config.get("paths", {})
    raw_paths = config.get("raw_paths", {})
    processed_paths = config.get("processed_paths", {})

    return {
        "store_raw": resolve_path(raw_paths.get("store_data", config_paths.get("store_data", DEFAULT_PATHS["store_raw"]))),
        "train_raw": resolve_path(raw_paths.get("train_data", DEFAULT_PATHS["train_raw"])),
        "test_raw": resolve_path(raw_paths.get("test_data", DEFAULT_PATHS["test_raw"])),
        "train_processed": resolve_path(processed_paths.get("train_processed", DEFAULT_PATHS["train_processed"])),
        "test_processed": resolve_path(processed_paths.get("test_processed", DEFAULT_PATHS["test_processed"])),
    }


def run_processing(config_source: str | Path | dict[str, Any] = "configs/config.yaml") -> dict[str, Any]:
    """Run raw-to-processed data pipeline and persist artifacts."""
    config = config_source if isinstance(config_source, dict) else load_config(config_source)
    resolved_paths = _resolve_processing_paths(config)

    store_df, train_df, test_df = load_data(
        resolved_paths["store_raw"],
        resolved_paths["train_raw"],
        resolved_paths["test_raw"],
    )

    train_merged, test_merged = merge_data(train_df, test_df, store_df)
    train_processed, test_processed = preprocess_data(train_merged, test_merged)

    resolved_paths["train_processed"].parent.mkdir(parents=True, exist_ok=True)
    resolved_paths["test_processed"].parent.mkdir(parents=True, exist_ok=True)
    train_processed.to_csv(resolved_paths["train_processed"], index=False)
    test_processed.to_csv(resolved_paths["test_processed"], index=False)

    return {
        "train_processed_path": str(resolved_paths["train_processed"]),
        "test_processed_path": str(resolved_paths["test_processed"]),
        "train_rows": int(len(train_processed)),
        "test_rows": int(len(test_processed)),
        "store_rows": int(len(store_df)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Process Rossmann raw datasets into cleaned merged datasets")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    result = run_processing(args.config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
