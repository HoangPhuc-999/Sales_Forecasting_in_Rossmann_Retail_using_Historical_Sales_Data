from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from rossmann_mlops.config import load_config, resolve_path
from rossmann_mlops.monitoring import log_jsonl
from rossmann_mlops.features import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, build_features, merge_store_data


def _load_training_data(train_path: Path, store_path: Path) -> pd.DataFrame:
    if not train_path.exists():
        raise FileNotFoundError(
            f"Training data file not found: {train_path}. Run `dvc pull data/train.csv data/store.csv` first."
        )
    if not store_path.exists():
        raise FileNotFoundError(
            f"Store data file not found: {store_path}. Run `dvc pull data/store.csv` first."
        )

    train_df = pd.read_csv(train_path)
    store_df = pd.read_csv(store_path)

    if "Sales" not in train_df.columns:
        raise ValueError("Training data must contain Sales target column")

    merged = merge_store_data(train_df, store_df)
    return merged


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_dvc_md5(dvc_path: Path) -> str | None:
    if not dvc_path.exists():
        return None

    with dvc_path.open("r", encoding="utf-8") as f:
        metadata = yaml.safe_load(f)

    outs = metadata.get("outs") if isinstance(metadata, dict) else None
    if not outs:
        return None

    first_out = outs[0]
    if isinstance(first_out, dict):
        return first_out.get("md5")
    return None


def _split_train_validation(df: pd.DataFrame, validation_start_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_date = pd.to_datetime(validation_start_date)
    dated_df = df.copy()
    dated_df["Date"] = pd.to_datetime(dated_df["Date"], errors="coerce")

    train_df = dated_df[dated_df["Date"] < split_date]
    val_df = dated_df[dated_df["Date"] >= split_date]

    if train_df.empty or val_df.empty:
        raise ValueError(
            "Train/validation split is empty. Check training.validation_start_date in config and data date range."
        )

    return train_df, val_df


def train_pipeline(config: dict[str, Any]) -> dict[str, Any]:
    paths_cfg = config["paths"]
    train_cfg = config["training"]
    reproducibility_cfg = config.get("reproducibility", {})

    train_path = resolve_path(paths_cfg["train_data"])
    store_path = resolve_path(paths_cfg["store_data"])
    model_file = resolve_path(paths_cfg["model_file"])
    metrics_file = resolve_path(paths_cfg["metrics_file"])
    experiment_log_file = resolve_path(reproducibility_cfg.get("experiment_log_file", "logs/experiments.jsonl"))

    seed = int(reproducibility_cfg.get("seed", train_cfg["random_state"]))
    set_global_seed(seed)

    merged = _load_training_data(train_path, store_path)
    merged = merged[(merged["Open"] != 0) & (merged["Sales"] > 0)].copy()

    train_df, val_df = _split_train_validation(merged, train_cfg["validation_start_date"])

    X_train = build_features(train_df)
    y_train = train_df["Sales"]

    X_val = build_features(val_df)
    y_val = val_df["Sales"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
            ("numerical", "passthrough", NUMERIC_COLUMNS),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=int(train_cfg["n_estimators"]),
        random_state=int(train_cfg["random_state"]),
        n_jobs=int(train_cfg["n_jobs"]),
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_val)

    rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
    mae = mean_absolute_error(y_val, preds)
    r2 = r2_score(y_val, preds)

    metrics = {
        "rmse": rmse,
        "mae": float(mae),
        "r2": float(r2),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "seed": seed,
    }

    model_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_file.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, model_file)
    model_hash = _hash_file(model_file)
    metrics_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log_jsonl(
        experiment_log_file,
        {
            "seed": seed,
            "train_data": str(train_path),
            "store_data": str(store_path),
            "model_file": str(model_file),
            "metrics_file": str(metrics_file),
            "model_hash": model_hash,
            "data_versions": {
                "train_csv_md5": _read_dvc_md5(train_path.with_suffix(train_path.suffix + ".dvc")),
                "store_csv_md5": _read_dvc_md5(store_path.with_suffix(store_path.suffix + ".dvc")),
                "test_csv_md5": _read_dvc_md5(resolve_path("data/test.csv.dvc")),
            },
            "metrics": metrics,
            "training_params": {
                "validation_start_date": train_cfg["validation_start_date"],
                "n_estimators": int(train_cfg["n_estimators"]),
                "random_state": int(train_cfg["random_state"]),
                "n_jobs": int(train_cfg["n_jobs"]),
            },
        },
    )

    return {
        "metrics": metrics,
        "model_path": str(model_file),
        "metrics_path": str(metrics_file),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train, evaluate, and persist Rossmann sales forecasting model")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)
    result = train_pipeline(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
