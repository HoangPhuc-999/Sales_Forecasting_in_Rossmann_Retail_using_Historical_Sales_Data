from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rossmann_mlops.monitoring import detect_data_drift, summarize_drift


def test_detect_data_drift_smoke() -> None:
    reference = pd.DataFrame(
        {
            "Store": [1, 1, 2, 2],
            "DayOfWeek": [1, 2, 3, 4],
            "Date": ["2015-01-01", "2015-01-02", "2015-01-03", "2015-01-04"],
            "Open": [1, 1, 1, 1],
            "Promo": [0, 1, 0, 1],
            "StateHoliday": ["0", "0", "0", "0"],
            "SchoolHoliday": [0, 0, 1, 1],
            "StoreType": ["a", "a", "b", "b"],
            "Assortment": ["a", "a", "a", "a"],
            "CompetitionDistance": [100.0, 110.0, 200.0, 210.0],
            "Promo2": [0, 0, 1, 1],
        }
    )
    current = pd.DataFrame(
        {
            "Store": [1, 1, 2, 2],
            "DayOfWeek": [6, 6, 7, 7],
            "Date": ["2015-02-01", "2015-02-02", "2015-02-03", "2015-02-04"],
            "Open": [1, 1, 1, 1],
            "Promo": [1, 1, 1, 1],
            "StateHoliday": ["0", "0", "0", "0"],
            "SchoolHoliday": [1, 1, 1, 1],
            "StoreType": ["a", "a", "b", "b"],
            "Assortment": ["a", "a", "a", "a"],
            "CompetitionDistance": [100.0, 120.0, 240.0, 260.0],
            "Promo2": [0, 0, 1, 1],
        }
    )

    drift = detect_data_drift(reference, current)
    assert len(drift) > 0
    assert summarize_drift(drift, drift_threshold=0.0) is not None
