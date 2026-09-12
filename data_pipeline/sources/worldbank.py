"""World Bank indicators API (no auth). https://api.worldbank.org/v2

Slow-moving macro for regime / country context, not daily trading signals.
"""

from __future__ import annotations

import logging

import pandas as pd

from ..http import HttpClient

LOG = logging.getLogger("data_pipeline.worldbank")

BASE = "https://api.worldbank.org/v2"

# (country_iso2, indicator_code, label)
SERIES: list[tuple[str, str, str]] = [
    ("IN", "NY.GDP.MKTP.KD.ZG", "India real GDP growth %"),
    ("IN", "NY.GDP.MKTP.CD", "India GDP current USD"),
    ("IN", "FP.CPI.TOTL.ZG", "India CPI inflation %"),
    ("IN", "FR.INR.RINR", "India real interest rate %"),
    ("IN", "FR.INR.LEND", "India lending rate %"),
    ("IN", "BX.KLT.DINV.WD.GD.ZS", "India FDI net inflows %GDP"),
    ("IN", "NE.TRD.GNFS.ZS", "India trade %GDP"),
    ("IN", "GC.DOD.TOTL.GD.ZS", "India central govt debt %GDP"),
    ("IN", "SL.UEM.TOTL.ZS", "India unemployment %"),
    ("IN", "PA.NUS.FCRF", "INR/USD official rate"),
    ("WLD", "NY.GDP.MKTP.KD.ZG", "World real GDP growth %"),
    ("WLD", "FP.CPI.TOTL.ZG", "World CPI inflation %"),
    ("US", "NY.GDP.MKTP.KD.ZG", "US real GDP growth %"),
    ("US", "FP.CPI.TOTL.ZG", "US CPI inflation %"),
]


def fetch_series(client: HttpClient, country: str, code: str, label: str) -> pd.DataFrame:
    url = f"{BASE}/country/{country}/indicator/{code}"
    payload = client.get(
        url,
        host_key="worldbank",
        params={"format": "json", "per_page": 1000},
        retries=3,
        timeout=30,
    )
    if not isinstance(payload, list) or len(payload) < 2 or not payload[1]:
        return pd.DataFrame()
    rows = [
        {
            "series_id": f"{country}:{code}",
            "label": label,
            "date": pd.Timestamp(int(r["date"]), 12, 31),
            "value": r["value"],
            "unit": (r.get("indicator") or {}).get("value"),
            "source": "worldbank",
        }
        for r in payload[1]
        if r.get("value") is not None
    ]
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df
