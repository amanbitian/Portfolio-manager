"""NSE India's own public JSON API (unofficial, no auth) - shareholding pattern.

NSE fronts its ``/api/*`` endpoints with basic bot-protection: a request with no
prior cookies gets a 401. Visiting a normal HTML page first lets ``requests``
pick up the anti-bot cookies via ``Set-Cookie``, same trick this codebase
already uses for BSE (see ``http.HttpClient.prime_cookies``). Rate limit lives
in ``config.py`` under the "nse" host key - kept conservative (1 req/s) because
NSE's own reference client (github.com/BennyThadikaran/NseIndiaApi) throttles
itself to 3 req/s and this site is known to block aggressively otherwise.
"""

from __future__ import annotations

import pandas as pd

from ..http import HttpClient

BASE = "https://www.nseindia.com"
_SHAREHOLDING_URL = f"{BASE}/api/corporate-share-holdings-master"
_PRIME_URL = f"{BASE}/companies-listing/corporate-filings-shareholding-pattern"

_COLUMNS = ["symbol", "date", "pr_and_prgrp", "public_val", "employeeTrusts"]
_NUMERIC = ["pr_and_prgrp", "public_val", "employeeTrusts"]


def prime(client: HttpClient) -> None:
    """Hit a normal NSE page once before any /api/* calls so the session has cookies."""
    client.prime_cookies(_PRIME_URL, host_key="nse")


def fetch_shareholding(client: HttpClient, symbol: str) -> pd.DataFrame:
    """One symbol's quarterly shareholding history.

    Only the summary split NSE's "master" endpoint exposes: promoter+group %,
    public %, employee-trust %. Finer FII/DII/pledge detail lives in each
    quarter's linked XBRL filing, not this endpoint - not fetched here.
    """
    data = client.get(
        _SHAREHOLDING_URL,
        host_key="nse",
        params={"index": "equities", "symbol": symbol.upper()},
        headers={"Referer": f"{_PRIME_URL}?symbol={symbol.upper()}&tabIndex=equity"},
    )
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    keep = [c for c in _COLUMNS if c in df.columns]
    df = df[keep].copy()
    df["date"] = pd.to_datetime(df.get("date"), format="%d-%b-%Y", errors="coerce")
    for c in _NUMERIC:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
