from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any
from datetime import datetime

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import yaml

from rossmann_mlops.config import load_config, resolve_path


class TrainingError(ValueError):
    """Raised when training input/split/configuration is invalid."""


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


REQUIRED_TRAIN_COLUMNS = [
    "Store",
    "DayOfWeek",
    "Promo",
    "Month",
    "Year",
    "WeekOfYear",
    "Sales_log",
]


def rmspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    safe_true = np.maximum(np.asarray(y_true, dtype=float), 1e-8)
    safe_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean(np.square((safe_true - safe_pred) / safe_true))))


def _ensure_required_columns(frame: pd.DataFrame, required_columns: list[str], source_name: str) -> None:
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise TrainingError(f"Missing required columns in {source_name}: {missing}")


def _prepare_training_columns(train_df: pd.DataFrame) -> pd.DataFrame:
    frame = train_df.copy()

    if "Date" in frame.columns:
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
        frame = frame.dropna(subset=["Date"]).copy()
        if "Year" not in frame.columns:
            frame["Year"] = frame["Date"].dt.year
        if "WeekOfYear" not in frame.columns:
            frame["WeekOfYear"] = frame["Date"].dt.isocalendar().week.astype(int)
        if "Month" not in frame.columns:
            frame["Month"] = frame["Date"].dt.month
        if "DayOfWeek" not in frame.columns:
            frame["DayOfWeek"] = frame["Date"].dt.weekday + 1

    if "Sales_log" not in frame.columns:
        if "Sales" not in frame.columns:
            raise TrainingError("Training data must include either 'Sales_log' or 'Sales'")
        frame["Sales_log"] = np.log(np.maximum(pd.to_numeric(frame["Sales"], errors="coerce"), 1.0))

    _ensure_required_columns(frame, REQUIRED_TRAIN_COLUMNS, "training data")

    frame["Year"] = pd.to_numeric(frame["Year"], errors="coerce")
    frame["WeekOfYear"] = pd.to_numeric(frame["WeekOfYear"], errors="coerce")
    frame["Month"] = pd.to_numeric(frame["Month"], errors="coerce")
    frame["DayOfWeek"] = pd.to_numeric(frame["DayOfWeek"], errors="coerce")
    frame["Promo"] = pd.to_numeric(frame["Promo"], errors="coerce")
    frame["Sales_log"] = pd.to_numeric(frame["Sales_log"], errors="coerce")

    return frame.dropna(subset=["Year", "WeekOfYear", "Month", "DayOfWeek", "Promo", "Sales_log"]).copy()


def apply_feature_engineering(
    train_set: pd.DataFrame,
    val_set: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    logger.info("Computing target-encoding mapping features")

    store_dw_promo_avg = (
        train_set.groupby(["Store", "DayOfWeek", "Promo"])["Sales_log"].mean().reset_index()
    )
    store_dw_promo_avg.rename(columns={"Sales_log": "Store_DW_Promo_Avg"}, inplace=True)

    month_avg = train_set.groupby("Month")["Sales_log"].mean().reset_index()
    month_avg.rename(columns={"Sales_log": "Month_Avg_Sales"}, inplace=True)

    train_set = train_set.merge(store_dw_promo_avg, on=["Store", "DayOfWeek", "Promo"], how="left")
    train_set = train_set.merge(month_avg, on="Month", how="left")

    val_set = val_set.merge(store_dw_promo_avg, on=["Store", "DayOfWeek", "Promo"], how="left")
    val_set = val_set.merge(month_avg, on="Month", how="left")

    global_mean_train = float(train_set["Sales_log"].mean())
    train_set["Store_DW_Promo_Avg"] = train_set["Store_DW_Promo_Avg"].fillna(global_mean_train)
    train_set["Month_Avg_Sales"] = train_set["Month_Avg_Sales"].fillna(global_mean_train)
    val_set["Store_DW_Promo_Avg"] = val_set["Store_DW_Promo_Avg"].fillna(global_mean_train)
    val_set["Month_Avg_Sales"] = val_set["Month_Avg_Sales"].fillna(global_mean_train)

    return train_set, val_set, store_dw_promo_avg, month_avg, global_mean_train


def _load_training_data(paths: dict[str, Any]) -> pd.DataFrame:
    train_path = resolve_path(paths["train_data"])

    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")

    return pd.read_csv(train_path)


def _split_train_validation(df: pd.DataFrame, training: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_year = int(training.get("validation_year", 2015))
    split_week = int(training.get("validation_week_min", 26))

    if {"Year", "WeekOfYear"}.issubset(df.columns):
        mask = (df["Year"] == split_year) & (df["WeekOfYear"] >= split_week)
        if mask.any() and (~mask).any():
            return df.loc[~mask].copy(), df.loc[mask].copy()

    if "Date" in df.columns:
        validation_start_date = training.get("validation_start_date", "2015-06-01")
        start = pd.to_datetime(validation_start_date, errors="coerce")
        parsed = pd.to_datetime(df["Date"], errors="coerce")
        if pd.notna(start):
            mask = parsed >= start
            if mask.any() and (~mask).any():
                return df.loc[~mask].copy(), df.loc[mask].copy()

    if len(df) < 2:
        raise TrainingError("Training data must contain at least 2 rows")

    split_index = max(1, int(len(df) * 0.8))
    if split_index >= len(df):
        split_index = len(df) - 1
    return df.iloc[:split_index].copy(), df.iloc[split_index:].copy()


def _resolve_artifacts_dir(paths: dict[str, Any], model_path: Path) -> Path:
    artifacts_dir = paths.get("artifacts_dir")
    if artifacts_dir:
        return resolve_path(artifacts_dir)
    return model_path.parent


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _compact_model_params(params: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key, value in params.items():
        if _is_missing_value(value):
            continue
        if isinstance(value, (np.generic,)):
            compact[key] = value.item()
        else:
            compact[key] = value
    return compact


def _prepare_xgb_inputs(x_train: pd.DataFrame, x_val: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ensure XGBoost-compatible numeric dtypes with stable encoding across splits."""
    prepared_train = x_train.copy()
    prepared_val = x_val.copy()

    categorical_cols = prepared_train.select_dtypes(include=["object", "category"]).columns.tolist()
    for column in categorical_cols:
        # Normalize to string to avoid mixed-type category issues (e.g., int + str) in XGBoost.
        train_values = prepared_train[column].astype(str)
        val_values = prepared_val[column].astype(str)

        categories = pd.Index(train_values.unique())
        mapping = {value: idx for idx, value in enumerate(categories)}

        prepared_train[column] = train_values.map(mapping).fillna(-1).astype(np.int32)
        prepared_val[column] = val_values.map(mapping).fillna(-1).astype(np.int32)

    return prepared_train, prepared_val


def _resolve_model_config_output_path(
    *,
    paths: dict[str, Any],
    training: dict[str, Any],
    artifacts_dir: Path,
    default_model_config_path: Path,
) -> tuple[Path, bool]:
    is_production_train = bool(training.get("production_train", False))
    if is_production_train:
        return default_model_config_path, True

    candidate_override = paths.get("model_config_candidate_file")
    if candidate_override:
        return resolve_path(candidate_override), False

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return artifacts_dir / f"model_config_candidate_{timestamp}.yaml", False


def _log_mlflow_payload(
    mlflow_cfg: dict[str, Any],
    model: xgb.XGBRegressor,
    metrics: dict[str, float],
    run_name: str,
) -> None:
    if not bool(mlflow_cfg.get("enabled", False)):
        return

    tracking_uri = mlflow_cfg.get("tracking_uri") or "http://127.0.0.1:5000"
    experiment_name = mlflow_cfg.get("experiment_name") or "Rossmann_Final_Training"

    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(model.get_params())
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, float(metric_value))
            mlflow.sklearn.log_model(model, "model")
    except Exception as exc:  # pragma: no cover
        logger.warning("Skipping MLflow logging due to error: %s", exc)


def train_pipeline(config: dict[str, Any]) -> dict[str, Any]:
    paths = config.get("paths", {})
    training = config.get("training", {})
    mlflow_cfg = config.get("mlflow", {})

    model_path = resolve_path(paths["model_file"])
    metrics_path = resolve_path(paths["metrics_file"])
    artifacts_dir = _resolve_artifacts_dir(paths, model_path)
    default_model_config_path = resolve_path(paths.get("model_config_file", "configs/model_config.yaml"))

    raw_df = _load_training_data(paths)
    prepared_df = _prepare_training_columns(raw_df)
    if len(prepared_df) < 2:
        raise TrainingError("Training data must contain at least 2 valid rows after preprocessing")

    train_df, val_df = _split_train_validation(prepared_df, training)
    if len(train_df) == 0 or len(val_df) == 0:
        raise TrainingError("Unable to create a valid train/validation split")

    train_df, val_df, store_dw_promo_mapping, month_mapping, global_mean = apply_feature_engineering(train_df, val_df)

    drop_cols = ["Sales", "Sales_log", "Customers", "Month", "Promo2", "Date", "Id"]
    x_train = train_df.drop(columns=drop_cols, errors="ignore")
    x_val = val_df.drop(columns=drop_cols, errors="ignore")
    x_train, x_val = _prepare_xgb_inputs(x_train, x_val)

    y_train = train_df["Sales_log"].astype(float)
    y_val = val_df["Sales_log"].astype(float)
    y_train_true = np.exp(y_train)
    y_val_true = np.exp(y_val)

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        tree_method=training.get("tree_method", "hist"),
        n_estimators=int(training.get("n_estimators", 1000)),
        max_depth=int(training.get("max_depth", 11)),
        learning_rate=float(training.get("learning_rate", 0.025)),
        min_child_weight=float(training.get("min_child_weight", 14)),
        random_state=int(training.get("random_state", 42)),
        n_jobs=int(training.get("n_jobs", -1)),
    )

    logger.info("Training XGBoost model")
    model.fit(x_train, y_train)

    y_train_pred = np.exp(model.predict(x_train))
    y_val_pred = np.exp(model.predict(x_val))

    train_rmspe = rmspe(y_train_true, y_train_pred)
    val_rmspe = rmspe(y_val_true, y_val_pred)
    rmspe_gap = float(val_rmspe - train_rmspe)
    rmse = float(np.sqrt(mean_squared_error(y_val_true, y_val_pred)))
    mae = float(mean_absolute_error(y_val_true, y_val_pred))
    r2 = float(r2_score(y_val_true, y_val_pred))

    metrics = {
        "train_rmspe": float(train_rmspe),
        "val_rmspe": float(val_rmspe),
        "rmspe_gap": rmspe_gap,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }

    _log_mlflow_payload(mlflow_cfg, model, metrics, run_name=training.get("run_name", "Production_Train_Logic"))

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_params = _compact_model_params(model.get_params())
    model_config_path, config_overwritten = _resolve_model_config_output_path(
        paths=paths,
        training=training,
        artifacts_dir=artifacts_dir,
        default_model_config_path=default_model_config_path,
    )
    model_config_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)
    joblib.dump(store_dw_promo_mapping, artifacts_dir / "store_dw_promo_mapping.pkl")
    joblib.dump(month_mapping, artifacts_dir / "month_mapping.pkl")
    joblib.dump(global_mean, artifacts_dir / "global_mean_sales.pkl")

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    model_config_payload = {
        "project": "Rossmann Store Sales",
        "best_model": {
            "name": "XGBoost",
            "val_rmspe": float(val_rmspe),
            "params": model_params,
        },
        "features": {
            "input_columns": list(x_train.columns),
            "target": "Sales_log",
        },
    }
    model_config_path.write_text(yaml.safe_dump(model_config_payload, sort_keys=False), encoding="utf-8")

    return {
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "model_config_path": str(model_config_path),
        "artifacts_dir": str(artifacts_dir),
        "metrics": metrics,
        "n_train": int(len(x_train)),
        "n_validation": int(len(x_val)),
        "model_config_overwritten": config_overwritten,
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
