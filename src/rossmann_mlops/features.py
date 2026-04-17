from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd


class FeatureEngineeringError(ValueError):
    """Raised when feature engineering input is invalid."""


@dataclass(frozen=True)
class FeatureSpec:
    required_columns: list[str]
    drop_columns: list[str]


CATEGORICAL_COLUMNS: Final[list[str]] = [
    "StoreType",
    "Assortment",
    "StateHoliday",
    "Promo",
    "SchoolHoliday",
    "Promo2",
    "Is_Promo2_Month",
]
NUMERIC_COLUMNS: Final[list[str]] = [
    "Store",
    "DayOfWeek",
    "Month",
    "Day",
    "Year",
    "WeekOfYear",
    "CompetitionDistance",
    "Promo2Open_Month",
    "CompetitionOpen_Month",
]

FEATURE_SPEC: Final[FeatureSpec] = FeatureSpec(
    required_columns=[
        "Store",
        "Date",
        "Promo",
        "StateHoliday",
        "SchoolHoliday",
        "StoreType",
        "Assortment",
        "CompetitionDistance",
        "Promo2",
        "Promo2SinceWeek",
        "Promo2SinceYear",
        "CompetitionOpenSinceMonth",
        "CompetitionOpenSinceYear",
        "PromoInterval",
    ],
    drop_columns=[
        "Date",
        "CompetitionOpenSinceMonth",
        "CompetitionOpenSinceYear",
        "Promo2SinceWeek",
        "Promo2SinceYear",
        "PromoInterval",
        "Customers",
        "Open",
    ],
)

STATE_HOLIDAY_MAP: Final[dict[str, int]] = {"0": 0, "a": 1, "b": 2, "c": 3}
STORE_TYPE_MAP: Final[dict[str, int]] = {"a": 0, "b": 1, "c": 2, "d": 3}
ASSORTMENT_MAP: Final[dict[str, int]] = {"a": 0, "b": 1, "c": 2}
MONTH_NAME_MAP: Final[dict[int, str]] = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sept",
    10: "Oct",
    11: "Nov",
    12: "Dec",
}


def _ensure_required_columns(frame: pd.DataFrame, required_columns: list[str]) -> None:
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise FeatureEngineeringError(f"Missing required columns for feature engineering: {missing}")


def _coerce_input_types(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["Date"]).copy()

    data["Promo2SinceWeek"] = pd.to_numeric(data["Promo2SinceWeek"], errors="coerce").fillna(0)
    data["Promo2SinceYear"] = pd.to_numeric(data["Promo2SinceYear"], errors="coerce").fillna(0)
    data["CompetitionOpenSinceMonth"] = pd.to_numeric(data["CompetitionOpenSinceMonth"], errors="coerce").fillna(0)
    data["CompetitionOpenSinceYear"] = pd.to_numeric(data["CompetitionOpenSinceYear"], errors="coerce").fillna(0)
    data["CompetitionDistance"] = pd.to_numeric(data["CompetitionDistance"], errors="coerce").fillna(0)
    data["Promo2"] = pd.to_numeric(data["Promo2"], errors="coerce").fillna(0).astype(int)

    data["StateHoliday"] = data["StateHoliday"].astype(str)
    data["PromoInterval"] = data["PromoInterval"].fillna("None").astype(str)
    data["StoreType"] = data["StoreType"].astype(str).str.lower()
    data["Assortment"] = data["Assortment"].astype(str).str.lower()
    return data


def _add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["Year"] = data["Date"].dt.year
    data["Month"] = data["Date"].dt.month
    data["Day"] = data["Date"].dt.day
    data["WeekOfYear"] = data["Date"].dt.isocalendar().week.astype(int)
    data["DayOfWeek"] = data["Date"].dt.weekday + 1
    return data


def _add_promo_competition_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()

    sales_weeks = data["Year"] * 52 + data["WeekOfYear"]
    promo_weeks = data["Promo2SinceYear"] * 52 + data["Promo2SinceWeek"]
    data["Promo2Open_Month"] = (sales_weeks - promo_weeks) / 4.0

    sales_months = data["Year"] * 12 + data["Month"]
    competition_months = data["CompetitionOpenSinceYear"] * 12 + data["CompetitionOpenSinceMonth"]
    data["CompetitionOpen_Month"] = sales_months - competition_months

    data.loc[(data["Promo2"] == 0) | (data["Promo2SinceYear"] == 0), "Promo2Open_Month"] = 0
    data.loc[data["CompetitionOpenSinceYear"] == 0, "CompetitionOpen_Month"] = 0

    data["Promo2Open_Month"] = data["Promo2Open_Month"].clip(0, 24)
    data["CompetitionOpen_Month"] = data["CompetitionOpen_Month"].clip(0, 24)
    return data


def _add_promo_interval_feature(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["month_tmp"] = data["Month"].map(MONTH_NAME_MAP)

    promo_interval_normalized = (
        data["PromoInterval"].fillna("None").astype(str).str.replace(" ", "", regex=False).str.split(",")
    )
    has_promo_interval = (data["Promo2"] == 1) & data["PromoInterval"].ne("None") & data["PromoInterval"].ne("")

    data["Is_Promo2_Month"] = pd.Series(0, index=data.index, dtype="int64")
    if has_promo_interval.any():
        promo_flags = [
            int(month_name in allowed_months)
            for month_name, allowed_months in zip(
                data.loc[has_promo_interval, "month_tmp"],
                promo_interval_normalized.loc[has_promo_interval],
            )
        ]
        data.loc[has_promo_interval, "Is_Promo2_Month"] = np.asarray(promo_flags, dtype="int64")

    return data.drop(columns=["month_tmp"])


def _encode_categorical_columns(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["StateHoliday"] = data["StateHoliday"].map(STATE_HOLIDAY_MAP).fillna(0).astype(int)
    data["StoreType"] = data["StoreType"].map(STORE_TYPE_MAP).fillna(0).astype(int)
    data["Assortment"] = data["Assortment"].map(ASSORTMENT_MAP).fillna(0).astype(int)
    return data


def extract_row_logic(df: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible API that applies feature derivation to one dataframe."""
    _ensure_required_columns(df, FEATURE_SPEC.required_columns)
    data = _coerce_input_types(df)
    data = _add_time_features(data)
    data = _add_promo_competition_features(data)
    data = _add_promo_interval_feature(data)
    data = _encode_categorical_columns(data)
    return data


def merge_store_data(df: pd.DataFrame, store_df: pd.DataFrame) -> pd.DataFrame:
    """Merge row-level records with store metadata by Store ID."""
    if "Store" not in df.columns:
        raise FeatureEngineeringError("Input data must include 'Store' column")
    if "Store" not in store_df.columns:
        raise FeatureEngineeringError("Store data must include 'Store' column")
    return df.merge(store_df, on="Store", how="left")


def _finalize_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.drop(columns=FEATURE_SPEC.drop_columns, errors="ignore").copy()
    data = data.drop(columns=["Sales"], errors="ignore")

    expected_columns = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS
    missing = [column for column in expected_columns if column not in data.columns]
    if missing:
        raise FeatureEngineeringError(f"Feature frame missing expected columns: {missing}")

    return data[expected_columns]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build model features from merged Rossmann rows."""
    transformed = extract_row_logic(df)
    return _finalize_feature_frame(transformed)


def run_feature_engineering(train_merged: pd.DataFrame, test_merged: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backward-compatible API returning transformed train/test frames."""
    train_features = build_features(train_merged)
    test_features = build_features(test_merged)
    return train_features, test_features