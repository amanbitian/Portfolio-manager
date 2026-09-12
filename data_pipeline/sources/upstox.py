"""Upstox V3 historical-candle client (no API key required for historical data).

Endpoint:
    GET https://api.upstox.com/v3/historical-candle/{instrument_key}/{unit}/{interval}/{to}/{from}

Retrieval caps enforced by the API (so we chunk the request range):
    minutes, interval<=15 : 1 month per call
    minutes, interval>15  : 1 quarter per call
    hours                 : 1 quarter per call
    days                  : 1 decade per call
    weeks / months        : unlimited

Candle row order in the response is newest-first:
    [timestamp, open, high, low, close, volume, open_interest]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd
from dateutil.relativedelta import relativedelta  # provided by pandas' dep chain

from ..http import HttpClient

LOG = logging.getLogger("data_pipeline.upstox")

BASE = "https://api.upstox.com/v3/historical-candle"

CANDLE_COLS = ["ts", "open", "high", "low", "close", "volume", "open_interest"]


@dataclass(frozen=True)
class Chunk:
    start: date
    end: date

    @property
    def key(self) -> str:
        return f"{self.start.isoformat()}_{self.end.isoformat()}"


def month_chunks(start: date, end: date) -> list[Chunk]:
    """Calendar-month chunks covering [start, end]."""
    out: list[Chunk] = []
    cur = start.replace(day=1)
    while cur <= end:
        nxt = cur + relativedelta(months=1)
        out.append(Chunk(max(cur, start), min(nxt - timedelta(days=1), end)))
        cur = nxt
    return out


def year_chunks(start: date, end: date, span_years: int = 9) -> list[Chunk]:
    """Multi-year chunks (default 9y) staying under the 1-decade daily cap."""
    out: list[Chunk] = []
    cur = start
    while cur <= end:
        stop = min(date(cur.year + span_years, 12, 31), end)
        out.append(Chunk(cur, stop))
        cur = stop + timedelta(days=1)
    return out


def chunks_for(unit: str, interval: int, start: date, end: date) -> list[Chunk]:
    if unit == "minutes":
        return month_chunks(start, end)
    if unit == "hours":
        return year_chunks(start, end, span_years=0)  # ~1 quarter is safe within a year
    if unit == "days":
        return year_chunks(start, end, span_years=9)
    return [Chunk(start, end)]


class UpstoxClient:
    def __init__(self, client: HttpClient) -> None:
        self.http = client

    def fetch_chunk(
        self,
        instrument_key: str,
        unit: str,
        interval: int,
        chunk: Chunk,
    ) -> pd.DataFrame:
        url = (
            f"{BASE}/{instrument_key}/{unit}/{interval}"
            f"/{chunk.end.isoformat()}/{chunk.start.isoformat()}"
        )
        payload = self.http.get(url, host_key="upstox")
        candles = (payload or {}).get("data", {}).get("candles", []) or []
        if not candles:
            return pd.DataFrame(columns=CANDLE_COLS)
        df = pd.DataFrame(candles, columns=CANDLE_COLS)
        ts = pd.to_datetime(df["ts"], utc=False, format="ISO8601")
        df["ts_ist"] = ts.dt.tz_convert("Asia/Kolkata")
        df["ts_utc"] = ts.dt.tz_convert("UTC")
        df = df.drop(columns=["ts"])
        for c in ("open", "high", "low", "close"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").astype("Int64")
        df["open_interest"] = pd.to_numeric(
            df["open_interest"], errors="coerce"
        ).astype("Int64")
        return df.sort_values("ts_utc").reset_index(drop=True)
