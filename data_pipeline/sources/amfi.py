"""AMFI mutual fund NAV data via mfapi.in.

mfapi.in is a free, no-auth community wrapper around AMFI's own daily
``NAVAll.txt`` feed (https://portal.amfiindia.com/spages/NAVAll.txt). It
publishes no numeric rate limit - its docs just ask callers to cache
responses, which this ingest already does (one-time historical pull per
scheme, written to Parquet, resumable via manifest).
"""

from __future__ import annotations

import pandas as pd

from ..http import HttpClient

BASE = "https://api.mfapi.in/mf"


def fetch_scheme_list(client: HttpClient) -> pd.DataFrame:
    """Every AMFI scheme code + name mfapi.in knows about (~18k, all plans/options)."""
    data = client.get(BASE, host_key="mfapi")
    df = pd.DataFrame(data)
    df = df.rename(columns={"schemeCode": "scheme_code", "schemeName": "scheme_name"})
    df["scheme_code"] = pd.to_numeric(df["scheme_code"], errors="coerce")
    df = df.dropna(subset=["scheme_code"])
    df["scheme_code"] = df["scheme_code"].astype(int)
    return df.drop_duplicates(subset=["scheme_code"]).reset_index(drop=True)


def fetch_nav_history(client: HttpClient, scheme_code: int) -> pd.DataFrame:
    """Full daily NAV history for one scheme, oldest first."""
    payload = client.get(f"{BASE}/{scheme_code}", host_key="mfapi")
    rows = payload.get("data") or []
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    df = df.dropna(subset=["date", "nav"])
    if df.empty:
        return df

    meta = payload.get("meta") or {}
    df["scheme_code"] = int(scheme_code)
    df["fund_house"] = meta.get("fund_house")
    df["scheme_type"] = meta.get("scheme_type")
    df["scheme_category"] = meta.get("scheme_category")
    return df.sort_values("date").reset_index(drop=True)
