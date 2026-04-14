from __future__ import annotations

from typing import Iterable

import pandas as pd

FEATURE_COLUMNS: list[str] = [
    "Store",
    "DayOfWeek",
    "Open",
    "Promo",
    "SchoolHoliday",
    "StateHoliday",
    "StoreType",
    "Assortment",
    "CompetitionDistance",
    "Promo2",
    "Month",
    "Day",
    "WeekOfYear",
    "Year",
]

CATEGORICAL_COLUMNS: list[str] = ["StateHoliday", "StoreType", "Assortment"]
NUMERIC_COLUMNS: list[str] = [
    "Store",
    "DayOfWeek",
    "Open",
    "Promo",
    "SchoolHoliday",
    "CompetitionDistance",
    "Promo2",
    "Month",
    "Day",
    "WeekOfYear",
    "Year",
]

REQUIRED_BASE_COLUMNS: list[str] = [
    "Store",
    "Date",
    "DayOfWeek",
    "Open",
    "Promo",
    "StateHoliday",
    "SchoolHoliday",
]


def _ensure_columns(df: pd.DataFrame, required_columns: Iterable[str], source_name: str) -> None:
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {source_name}: {missing}")


def merge_store_data(df: pd.DataFrame, store_df: pd.DataFrame) -> pd.DataFrame:
    _ensure_columns(df, ["Store"], "input data")
    _ensure_columns(store_df, ["Store", "StoreType", "Assortment", "CompetitionDistance", "Promo2"], "store data")

    merged = df.merge(store_df, on="Store", how="left")
    merged["CompetitionDistance"] = merged["CompetitionDistance"].fillna(0.0)
    merged["Promo2"] = merged["Promo2"].fillna(0)
    merged["StoreType"] = merged["StoreType"].fillna("unknown")
    merged["Assortment"] = merged["Assortment"].fillna("unknown")
    return merged


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    _ensure_columns(df, REQUIRED_BASE_COLUMNS, "input data")

    featured = df.copy()
    featured["Date"] = pd.to_datetime(featured["Date"], errors="coerce")
    if featured["Date"].isna().any():
        raise ValueError("Date column contains invalid values")

    featured["Month"] = featured["Date"].dt.month
    featured["Day"] = featured["Date"].dt.day
    featured["WeekOfYear"] = featured["Date"].dt.isocalendar().week.astype(int)
    featured["Year"] = featured["Date"].dt.year

    featured["Open"] = featured["Open"].fillna(1).astype(int)
    featured["Promo"] = featured["Promo"].fillna(0).astype(int)
    featured["SchoolHoliday"] = featured["SchoolHoliday"].fillna(0).astype(int)
    featured["StateHoliday"] = featured["StateHoliday"].fillna("0").astype(str)

    return featured[FEATURE_COLUMNS]
