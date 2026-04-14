from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rossmann_mlops.config import load_config, resolve_path
from rossmann_mlops.features import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, build_features, merge_store_data


@dataclass(frozen=True)
class DriftResult:
    column: str
    psi: float
    status: str


@dataclass(frozen=True)
class MonitoringReport:
    timestamp: str
    drift: list[DriftResult]
    performance: dict[str, float]
    alert: str | None


class MonitoringError(ValueError):
    pass


EXPECTED_REQUEST_COLUMNS = [
    "Store",
    "DayOfWeek",
    "Date",
    "Open",
    "Promo",
    "StateHoliday",
    "SchoolHoliday",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_required_columns(df: pd.DataFrame, required_columns: list[str], source_name: str) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise MonitoringError(f"Missing required columns in {source_name}: {missing}")


def _psi_from_distributions(reference: pd.Series, current: pd.Series, buckets: int = 10) -> float:
    reference = reference.replace([np.inf, -np.inf], np.nan).dropna()
    current = current.replace([np.inf, -np.inf], np.nan).dropna()

    if reference.empty or current.empty:
        return 0.0

    if pd.api.types.is_numeric_dtype(reference):
        quantiles = np.linspace(0, 1, buckets + 1)
        edges = np.unique(reference.quantile(quantiles).to_numpy())
        if len(edges) < 3:
            edges = np.array([reference.min(), reference.max()])
        if len(edges) == 2:
            edges = np.array([edges[0], edges[1] + 1e-9])

        ref_bins = pd.cut(reference, bins=edges, include_lowest=True, duplicates="drop")
        cur_bins = pd.cut(current, bins=edges, include_lowest=True, duplicates="drop")
        ref_dist = ref_bins.value_counts(normalize=True, sort=False)
        cur_dist = cur_bins.value_counts(normalize=True, sort=False)
    else:
        ref_dist = reference.astype(str).value_counts(normalize=True)
        cur_dist = current.astype(str).value_counts(normalize=True)

    categories = sorted(set(ref_dist.index.astype(str)) | set(cur_dist.index.astype(str)))
    psi = 0.0
    for category in categories:
        ref_pct = float(ref_dist.get(category, 0.0))
        cur_pct = float(cur_dist.get(category, 0.0))
        ref_pct = max(ref_pct, 1e-6)
        cur_pct = max(cur_pct, 1e-6)
        psi += (cur_pct - ref_pct) * np.log(cur_pct / ref_pct)

    return float(psi)


def detect_data_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    store_df: pd.DataFrame | None = None,
) -> list[DriftResult]:
    _ensure_required_columns(reference_df, EXPECTED_REQUEST_COLUMNS, "reference data")
    _ensure_required_columns(current_df, EXPECTED_REQUEST_COLUMNS, "current data")

    reference_augmented = reference_df.copy()
    current_augmented = current_df.copy()
    reference_augmented["Date"] = pd.to_datetime(reference_augmented["Date"], errors="coerce")
    current_augmented["Date"] = pd.to_datetime(current_augmented["Date"], errors="coerce")

    if store_df is not None:
        _ensure_required_columns(store_df, ["Store", "StoreType", "Assortment", "CompetitionDistance", "Promo2"], "store data")
        reference_augmented = merge_store_data(reference_augmented, store_df)
        current_augmented = merge_store_data(current_augmented, store_df)
    else:
        _ensure_required_columns(
            reference_augmented,
            ["StoreType", "Assortment", "CompetitionDistance", "Promo2"],
            "reference data",
        )
        _ensure_required_columns(
            current_augmented,
            ["StoreType", "Assortment", "CompetitionDistance", "Promo2"],
            "current data",
        )

    reference_features = build_features(reference_augmented.assign(StateHoliday=reference_augmented["StateHoliday"].astype(str)))
    current_features = build_features(current_augmented.assign(StateHoliday=current_augmented["StateHoliday"].astype(str)))

    results: list[DriftResult] = []
    for column in CATEGORICAL_COLUMNS + NUMERIC_COLUMNS:
        psi = _psi_from_distributions(reference_features[column], current_features[column])
        if psi >= 0.3:
            status = "severe_drift"
        elif psi >= 0.2:
            status = "moderate_drift"
        else:
            status = "stable"
        results.append(DriftResult(column=column, psi=psi, status=status))

    return results


def summarize_performance(metrics: dict[str, Any], thresholds: dict[str, float]) -> str | None:
    rmse_threshold = thresholds.get("rmse_alert_threshold")
    if rmse_threshold is not None and float(metrics.get("rmse", 0.0)) > rmse_threshold:
        return f"RMSE {metrics['rmse']:.4f} exceeds threshold {rmse_threshold:.4f}"

    mae_threshold = thresholds.get("mae_alert_threshold")
    if mae_threshold is not None and float(metrics.get("mae", 0.0)) > mae_threshold:
        return f"MAE {metrics['mae']:.4f} exceeds threshold {mae_threshold:.4f}"

    return None


def summarize_drift(drift_results: list[DriftResult], drift_threshold: float = 0.3) -> str | None:
    severe = [result for result in drift_results if result.psi >= drift_threshold]
    if severe:
        columns = ", ".join(result.column for result in severe)
        return f"Data drift detected in columns: {columns}"
    return None


def log_jsonl(log_path: str | Path, payload: dict[str, Any]) -> Path:
    resolved_path = resolve_path(log_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return resolved_path


def load_metrics(metrics_path: str | Path) -> dict[str, Any]:
    resolved_path = resolve_path(metrics_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {resolved_path}")
    with resolved_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    if not isinstance(metrics, dict):
        raise MonitoringError("Metrics file must contain a JSON object")
    return metrics


def _load_monitoring_config(config_source: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(config_source, dict):
        return config_source
    return load_config(config_source)


def run_monitoring(
    reference_data_path: str | Path,
    current_data_path: str | Path,
    config_path: str | Path | dict[str, Any] = "configs/config.yaml",
) -> MonitoringReport:
    config = _load_monitoring_config(config_path)
    reference_path = resolve_path(reference_data_path)
    current_path = resolve_path(current_data_path)
    store_path = resolve_path(config["paths"]["store_data"])

    if not reference_path.exists():
        raise FileNotFoundError(f"Reference data not found: {reference_path}")
    if not current_path.exists():
        raise FileNotFoundError(f"Current data not found: {current_path}")
    if not store_path.exists():
        raise FileNotFoundError(f"Store data not found: {store_path}")

    reference_df = pd.read_csv(reference_path)
    current_df = pd.read_csv(current_path)
    store_df = pd.read_csv(store_path)
    drift_results = detect_data_drift(reference_df, current_df, store_df=store_df)

    metrics_path = config["paths"]["metrics_file"]
    metrics = load_metrics(metrics_path) if resolve_path(metrics_path).exists() else {}

    thresholds = config.get("monitoring", {})
    drift_alert = summarize_drift(drift_results, float(thresholds.get("drift_alert_threshold", 0.3)))
    performance_alert = summarize_performance(metrics, thresholds)
    alert = drift_alert or performance_alert

    report = MonitoringReport(
        timestamp=_utc_now(),
        drift=drift_results,
        performance={k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
        alert=alert,
    )

    report_path = log_jsonl(
        thresholds.get("monitoring_report_file", "logs/monitoring_report.jsonl"),
        {
            "timestamp": report.timestamp,
            "drift": [result.__dict__ for result in report.drift],
            "performance": report.performance,
            "alert": report.alert,
        },
    )

    if alert:
        log_jsonl(
            thresholds.get("alert_file", "logs/alerts.jsonl"),
            {
                "timestamp": report.timestamp,
                "alert": alert,
                "report_path": str(report_path),
            },
        )

    return report


def retrain_from_config(config_path: str | Path = "configs/config.yaml") -> dict[str, Any]:
    from rossmann_mlops.train import train_pipeline

    config = load_config(config_path)
    result = train_pipeline(config)
    log_jsonl(
        config.get("monitoring", {}).get("performance_log_file", "logs/performance.jsonl"),
        {"timestamp": _utc_now(), **result["metrics"]},
    )
    return result
