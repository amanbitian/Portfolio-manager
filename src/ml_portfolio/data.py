from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO, StringIO
from typing import BinaryIO

import numpy as np
import pandas as pd


DEFAULT_SYMBOLS = [
    "RELIANCE",
    "HDFCBANK",
    "INFY",
    "TCS",
    "ICICIBANK",
    "LT",
    "AXISBANK",
    "SUNPHARMA",
]


@dataclass(frozen=True)
class DataValidation:
    rows: int
    columns: int
    start: pd.Timestamp
    end: pd.Timestamp
    missing_values: int
    notes: list[str]


def make_synthetic_prices(
    symbols: list[str] | None = None,
    periods: int = 1300,
    seed: int = 42,
    end_date: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """Create daily prices with simple bull, bear, and high-volatility regimes."""
    rng = np.random.default_rng(seed)
    symbols = symbols or DEFAULT_SYMBOLS
    anchor = pd.Timestamp(end_date) if end_date is not None else pd.Timestamp.today()
    # Business-day range generation with a weekend/holiday `end` can return one fewer
    # date than `periods` (observed under pandas 3.0.1); rolling back to the nearest
    # business day first keeps the count exact.
    anchor = pd.offsets.BDay().rollback(anchor.normalize())
    dates = pd.bdate_range(end=anchor, periods=periods)
    n_assets = len(symbols)

    base_corr = 0.32
    corr = np.full((n_assets, n_assets), base_corr)
    np.fill_diagonal(corr, 1.0)
    vol = np.linspace(0.012, 0.022, n_assets)
    cov = corr * np.outer(vol, vol)

    market_regime = np.zeros(periods)
    market_regime[periods // 3 : periods // 2] = -1
    market_regime[periods // 2 : periods * 2 // 3] = 1
    market_regime[periods * 2 // 3 :] = 0.5

    asset_alpha = rng.normal(0.00015, 0.00025, n_assets)
    returns = np.empty((periods, n_assets))

    for i in range(periods):
        regime = market_regime[i]
        drift = 0.00035 + asset_alpha
        if regime == -1:
            drift = -0.00045 + asset_alpha * 0.4
        elif regime == 1:
            drift = 0.00065 + asset_alpha

        shock_scale = 1.7 if regime == 0.5 else 1.0
        noise = rng.multivariate_normal(np.zeros(n_assets), cov * shock_scale)
        factor = rng.normal(0, 0.006 * shock_scale)
        returns[i] = drift + noise + factor

    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    return pd.DataFrame(prices, index=dates, columns=symbols).round(2)


def load_price_csv(file: BinaryIO | BytesIO | StringIO) -> pd.DataFrame:
    """Load wide or long CSV prices into a date-indexed close-price matrix."""
    raw = pd.read_csv(file)
    raw.columns = [str(col).strip() for col in raw.columns]
    lower_columns = {col.lower(): col for col in raw.columns}

    if "date" not in lower_columns:
        raise ValueError("CSV must include a date column.")

    date_col = lower_columns["date"]
    raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")
    raw = raw.dropna(subset=[date_col]).copy()

    lower = {col.lower(): col for col in raw.columns}
    if {"symbol", "close"}.issubset(lower):
        prices = raw.pivot_table(
            index=date_col,
            columns=lower["symbol"],
            values=lower["close"],
            aggfunc="last",
        )
    else:
        prices = raw.set_index(date_col)

    prices = prices.apply(pd.to_numeric, errors="coerce")
    prices = prices.sort_index()
    prices = prices.loc[:, prices.notna().sum() > 10]
    prices = prices.dropna(how="all")
    prices = prices.loc[~prices.index.duplicated(keep="last")]
    prices.columns = [str(col).upper() for col in prices.columns]

    if len(prices.columns) != len(set(prices.columns)):
        raise ValueError("CSV contains duplicate symbols after normalizing column names.")
    if "CASH" in prices.columns:
        raise ValueError("CASH is reserved by the allocator and cannot be used as an asset symbol.")
    if (prices <= 0).any().any():
        raise ValueError("CSV contains non-positive prices. Use adjusted positive close prices.")

    prices = prices.ffill(limit=5).dropna(axis=1, how="any")

    if prices.shape[1] < 2:
        raise ValueError("Need at least two assets with usable price history.")
    if len(prices) < 260:
        raise ValueError("Need at least 260 daily rows for walk-forward testing.")

    return prices


def validate_prices(prices: pd.DataFrame) -> DataValidation:
    notes: list[str] = []
    returns = prices.pct_change()

    if prices.index.is_monotonic_increasing:
        notes.append("Dates are sorted in ascending order.")
    else:
        notes.append("Dates were not sorted; sort before research use.")

    if (prices <= 0).any().any():
        notes.append("Some prices are non-positive and should be cleaned.")
    else:
        notes.append("All retained prices are positive.")

    large_moves = int((returns.abs() > 0.25).sum().sum())
    if large_moves:
        notes.append(f"Detected {large_moves} daily moves above 25%; inspect corporate actions.")
    else:
        notes.append("No daily moves above 25% detected.")

    return DataValidation(
        rows=len(prices),
        columns=prices.shape[1],
        start=prices.index.min(),
        end=prices.index.max(),
        missing_values=int(prices.isna().sum().sum()),
        notes=notes,
    )
