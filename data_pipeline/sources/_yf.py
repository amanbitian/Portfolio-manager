"""Shared yfinance plumbing.

yfinance silently returns empty frames when Yahoo throttles it. A ``curl_cffi``
session that impersonates a real browser avoids most throttling; ``fetch_frame``
adds retry-on-empty on top.
"""

from __future__ import annotations

import logging
import time

import pandas as pd
import yfinance as yf

LOG = logging.getLogger("data_pipeline.yf")

try:  # curl_cffi ships as a yfinance dependency
    from curl_cffi import requests as _cr

    _SESSION = _cr.Session(impersonate="chrome")
except Exception:  # noqa: BLE001 - fall back to yfinance's default session
    _SESSION = None


_last_call = [0.0]
_MIN_GAP = 0.7  # seconds between any two yfinance HTTP calls in this process


def _throttle() -> None:
    now = time.monotonic()
    wait = _MIN_GAP - (now - _last_call[0])
    if wait > 0:
        time.sleep(wait)
    _last_call[0] = time.monotonic()


def ticker(yf_ticker: str) -> yf.Ticker:
    return yf.Ticker(yf_ticker, session=_SESSION) if _SESSION else yf.Ticker(yf_ticker)


def throttle() -> None:
    _throttle()


def fetch_frame(tk: yf.Ticker, attr: str, *, retries: int = 3, pause: float = 2.5) -> pd.DataFrame:
    """Return ``getattr(tk, attr)``; retry while it comes back empty (Yahoo throttle)."""
    last = pd.DataFrame()
    for i in range(retries):
        _throttle()
        try:
            df = getattr(tk, attr)
        except Exception as exc:  # noqa: BLE001
            LOG.debug("%s.%s raised %s", getattr(tk, "ticker", "?"), attr, exc)
            df = pd.DataFrame()
        if df is not None and not df.empty:
            return df
        last = df if df is not None else last
        time.sleep(pause * (i + 1))
    return last
