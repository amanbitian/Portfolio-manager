"""Company fundamentals from yfinance.

Free and structured but shallow (~4-5 years). Deep history needs NSE XBRL filings -
see data_pipeline/README.md for that future pipeline.

Output is long-form so new line items never break the schema:
    symbol, period_end, freq, statement, line_item, value, retrieved_at
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd

from ._yf import fetch_frame, throttle, ticker

LOG = logging.getLogger("data_pipeline.fundamentals")

_STATEMENTS = {
    "income_stmt": ("income_stmt", "quarterly_income_stmt"),
    "balance_sheet": ("balance_sheet", "quarterly_balance_sheet"),
    "cash_flow": ("cashflow", "quarterly_cashflow"),
}

_INFO_KEYS = [
    "marketCap",
    "enterpriseValue",
    "trailingPE",
    "forwardPE",
    "priceToBook",
    "priceToSalesTrailing12Months",
    "returnOnEquity",
    "returnOnAssets",
    "profitMargins",
    "operatingMargins",
    "debtToEquity",
    "currentRatio",
    "quickRatio",
    "dividendYield",
    "payoutRatio",
    "beta",
    "bookValue",
    "sharesOutstanding",
    "floatShares",
    "sector",
    "industry",
]


def _melt(df: pd.DataFrame, symbol: str, freq: str, statement: str) -> pd.DataFrame:
    if df is None or getattr(df, "empty", True):
        return pd.DataFrame()
    frame = df.copy()
    frame.index = frame.index.map(str)
    frame.columns = [pd.to_datetime(c, errors="coerce") for c in frame.columns]
    stacked = frame.stack(future_stack=True)
    stacked.index = stacked.index.set_names(["line_item", "period_end"])
    long = stacked.rename("value").reset_index()
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long["symbol"] = symbol
    long["freq"] = freq
    long["statement"] = statement
    long["period_end"] = pd.to_datetime(long["period_end"], errors="coerce")
    long = long.dropna(subset=["value", "period_end"])
    return long[
        ["symbol", "period_end", "freq", "statement", "line_item", "value"]
    ]


def fetch_statements(yf_ticker: str, symbol: str) -> dict[str, pd.DataFrame]:
    tk = ticker(yf_ticker)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out: dict[str, pd.DataFrame] = {}

    for name, (annual_attr, quarterly_attr) in _STATEMENTS.items():
        parts = []
        for attr, freq in ((annual_attr, "A"), (quarterly_attr, "Q")):
            try:
                parts.append(_melt(fetch_frame(tk, attr), symbol, freq, name))
            except Exception as exc:  # noqa: BLE001
                LOG.debug("%s %s failed: %s", symbol, attr, exc)
        nonempty = [p for p in parts if not p.empty]
        df = pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()
        if not df.empty:
            df["retrieved_at"] = now
        out[name] = df

    try:
        throttle()
        info = tk.get_info() or {}
        stats = pd.DataFrame(
            [{"symbol": symbol, "key": k, "raw": info.get(k)} for k in _INFO_KEYS]
        )
        stats["value_str"] = stats["raw"].astype(str)
        stats["value_num"] = pd.to_numeric(stats["raw"], errors="coerce")
        stats = stats.drop(columns=["raw"])
        stats["retrieved_at"] = now
        out["key_stats"] = stats
    except Exception as exc:  # noqa: BLE001
        LOG.debug("%s info failed: %s", symbol, exc)
        out["key_stats"] = pd.DataFrame()

    return out
