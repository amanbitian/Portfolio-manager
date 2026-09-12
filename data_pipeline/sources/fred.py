"""FRED (St. Louis Fed) series. Needs a free key in ``FRED_API_KEY``.

https://fred.stlouisfed.org/docs/api/api_key.html  (30 seconds to register)

We request the standard observation endpoint and keep ``realtime_start`` so a later
point-in-time layer can respect data vintages (ALFRED-style).
"""

from __future__ import annotations

import logging

import pandas as pd

from ..http import HttpClient

LOG = logging.getLogger("data_pipeline.fred")

BASE = "https://api.stlouisfed.org/fred/series/observations"

SERIES: dict[str, str] = {
    "DFF": "US effective federal funds rate",
    "DGS10": "US 10Y treasury yield",
    "DGS2": "US 2Y treasury yield",
    "T10Y2Y": "US 10Y-2Y spread",
    "T10Y3M": "US 10Y-3M spread",
    "CPIAUCSL": "US CPI (SA)",
    "UNRATE": "US unemployment rate",
    "INDPRO": "US industrial production",
    "DTWEXBGS": "US dollar index (broad)",
    "VIXCLS": "CBOE VIX",
    "DCOILBRENTEU": "Brent crude USD/bbl",
    "DCOILWTICO": "WTI crude USD/bbl",
    "DEXINUS": "India / US foreign exchange rate (INR per USD)",
    "BAMLH0A0HYM2": "US high-yield OAS",
    "NFCI": "Chicago Fed financial conditions index",
}


def fetch_series(client: HttpClient, api_key: str, series_id: str, label: str) -> pd.DataFrame:
    payload = client.get(
        BASE,
        host_key="fred",
        params={
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": "1990-01-01",
        },
        retries=3,
        timeout=25,
    )
    obs = (payload or {}).get("observations", []) or []
    rows = [
        {
            "series_id": series_id,
            "label": label,
            "date": pd.Timestamp(o["date"]),
            "value": pd.to_numeric(o["value"], errors="coerce"),
            "realtime_start": o.get("realtime_start"),
            "source": "fred",
        }
        for o in obs
    ]
    df = pd.DataFrame(rows).dropna(subset=["value"]).reset_index(drop=True)
    return df
