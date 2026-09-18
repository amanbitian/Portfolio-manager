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
    """Create daily prices with simple bull, bear, and high-volatility regimes.

    Vectorized: draws every day's noise in one batch against a single Cholesky
    factor of the covariance matrix instead of re-decomposing a covariance
    matrix inside a per-day loop. This changed the RNG call order, so seeded
    numeric output differs from the previous (pre-vectorization) implementation
    for the same seed - see the README note on versioning this generator when
    seeded outputs change. Downstream Parquet caches key off price content
    (storage.price_key), so no cache migration is needed.
    """
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

    # Only two distinct covariance scales ever occur (normal vs. high-vol regime days),
    # so factorize the equicorrelation covariance once and scale batched draws per day
    # instead of calling rng.multivariate_normal (which re-decomposes its covariance
    # argument on every call) inside a Python loop.
    cholesky = np.linalg.cholesky(cov)
    sqrt_scale = np.sqrt(np.where(market_regime == 0.5, 1.7, 1.0))
    noise = (rng.standard_normal((periods, n_assets)) @ cholesky.T) * sqrt_scale[:, None]
    factor = rng.normal(0, 0.006, periods) * sqrt_scale

    drift = np.where(
        (market_regime == -1)[:, None], -0.00045 + asset_alpha * 0.4,
        np.where((market_regime == 1)[:, None], 0.00065 + asset_alpha,
                 0.00035 + asset_alpha),
    )

    returns = drift + noise + factor[:, None]
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    return pd.DataFrame(prices, index=dates, columns=symbols).round(2)


def clean_gappy_columns(prices: pd.DataFrame, limit: int = 5) -> tuple[pd.DataFrame, dict[str, str]]:
    """Forward-fill short gaps; keep columns whose only remaining gaps are at
    the edges of the window, recording where their real data starts/ends;
    drop columns with an interior gap, reporting why instead of silently
    vanishing them.

    Three kinds of remaining gap, handled differently on purpose:

    * A PURELY TRAILING gap (real data, then an unbroken run of missing
      values to the end of the frame - a symbol that stopped trading inside
      the window). Dropping the whole column for that - the old behavior -
      meant a backtest decision on an EARLY date was silently affected by
      whether the symbol still had data at the very END of the window, a
      real lookahead bias in universe construction. The column is kept, its
      post-cutoff cells forward-filled flat, and the last real date recorded
      in `.attrs["eligible_through"]` so the walk-forward loops stop treating
      the symbol as investable from then on (and sell any open position at
      the cutoff, see execution.liquidate_at_cutoff).
    * A PURELY LEADING gap (an unbroken run of missing values from the start
      of the frame, then real data - a symbol that LISTED inside the window).
      Dropping the whole column for that is not a lookahead bias (nothing
      could have been traded before it existed) but it silently removes a
      large share of any multi-year real universe: 266 of 750 Nifty Total
      Market names over 2018-2026 in this project's data lake. The column is
      kept, its pre-listing cells back-filled flat at the first real price
      (so every downstream price consumer stays finite and the first real
      day shows no artificial jump), and the first real date recorded in
      `.attrs["eligible_from"]`. That placeholder is DANGEROUS if trusted: a
      flat price is a zero-return, zero-volatility asset to any risk model,
      so the decision layer excludes a symbol from the optimizer's candidate
      set until it has MIN_LISTING_HISTORY_DAYS of real history
      (decisions.py), the ML feature/target builders mask pre-listing cells
      to NaN (features.py), and Copilot masks its scores through the
      indicator warm-up (strategy_copilot.py). Consumers that need the real
      series only should go through `listed_only`.
    * An INTERIOR gap (real data on both sides of a run longer than `limit`)
      keeps the original behavior exactly - dropped, reported in `excluded`.
      Re-entering a universe after a genuine data outage or trading
      suspension is a separate, more debatable question this doesn't answer;
      the data lake shows it to be rare (22 symbols over 2018-2026, 1 over
      2023-2026). A column with BOTH a leading and a trailing gap and no
      interior one is kept with both attrs set.

    Dates are stored as ISO-8601 strings, not Timestamps, on purpose: pandas
    JSON-encodes a non-empty `.attrs` into parquet metadata with no type
    guard, so a Timestamp value made `to_parquet` raise (crashing the
    persistent feature cache in decisions.py and every artifact save that
    embeds the price frame). A JSON-native value survives parquet, pickle
    and `st.cache_data` round-trips intact; read it back through
    `eligibility_starts` / `eligibility_cutoffs`, never from the raw attrs.
    """
    filled = prices.ffill(limit=limit)
    # Count cells the bounded ffill actually touched (computed before the
    # edge-gap carve-outs below apply their own, separate unlimited fills) -
    # a forward-filled price is otherwise indistinguishable from an
    # observed one to every downstream consumer (volatility/return
    # calculations would silently understate risk on a filled day, since a
    # flat-filled price shows zero return). Exposed via `.attrs["fill_counts"]`
    # so callers can disclose it instead of leaving it invisible.
    fill_counts = ((filled.notna()) & (prices.isna())).sum()
    still_missing = filled.isna()
    excluded: dict[str, str] = {}
    eligible_from: dict[str, str] = {}
    eligible_through: dict[str, str] = {}
    keep = []
    for col in filled.columns:
        missing = still_missing[col].to_numpy()
        if not missing.any():
            keep.append(col)
            continue
        present = ~missing
        if not present.any():
            excluded[col] = "no data in range"
            continue
        first_real = int(np.argmax(present))
        last_real = len(present) - 1 - int(np.argmax(present[::-1]))
        if missing[first_real:last_real + 1].any():
            excluded[col] = "gap longer than the forward-fill limit inside the symbol's own history"
            continue
        keep.append(col)
        if first_real > 0:
            eligible_from[col] = pd.Timestamp(filled.index[first_real]).isoformat()
            filled[col] = filled[col].bfill()
        if last_real < len(present) - 1:
            eligible_through[col] = pd.Timestamp(filled.index[last_real]).isoformat()
            filled[col] = filled[col].ffill()
    cleaned = filled.loc[:, keep]
    cleaned.attrs["eligible_from"] = eligible_from
    cleaned.attrs["eligible_through"] = eligible_through
    cleaned.attrs["fill_counts"] = {col: int(fill_counts[col]) for col in keep if fill_counts[col] > 0}
    return cleaned, excluded


def eligibility_starts(prices: pd.DataFrame) -> dict[str, pd.Timestamp]:
    """Per-symbol first-real-data date recorded by `clean_gappy_columns`'s
    leading-gap carve-out, as Timestamps. The only supported way to read
    `.attrs["eligible_from"]`; see `eligibility_cutoffs` for why the stored
    form is an ISO string and both forms coerce here.
    """
    return {str(symbol): pd.Timestamp(start)
            for symbol, start in (prices.attrs.get("eligible_from") or {}).items()}


def listed_only(prices: pd.DataFrame) -> pd.DataFrame:
    """A copy of `prices` with every edge-placeholder cell (before a symbol's
    `eligible_from`, after its `eligible_through`) set to NaN, so a consumer
    that measures a symbol's OWN history - trailing returns, indicator
    warm-ups, ML features and targets - sees only real prints. Returns the
    input itself, uncopied, when nothing is recorded.
    """
    starts, cutoffs = eligibility_starts(prices), eligibility_cutoffs(prices)
    if not starts and not cutoffs:
        return prices
    masked = prices.copy()
    for symbol, start in starts.items():
        if symbol in masked.columns:
            masked.loc[masked.index < start, symbol] = np.nan
    for symbol, cutoff in cutoffs.items():
        if symbol in masked.columns:
            masked.loc[masked.index > cutoff, symbol] = np.nan
    return masked


def eligibility_cutoffs(prices: pd.DataFrame) -> dict[str, pd.Timestamp]:
    """Per-symbol last-real-data date recorded by `clean_gappy_columns`'s
    trailing-gap carve-out, as Timestamps. The only supported way to read
    `.attrs["eligible_through"]`: the stored value is an ISO string (so it
    survives parquet/pickle round-trips - see `clean_gappy_columns`), and a
    caller-built frame may carry a Timestamp directly; both coerce here.
    """
    return {str(symbol): pd.Timestamp(cutoff)
            for symbol, cutoff in (prices.attrs.get("eligible_through") or {}).items()}


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

    prices, _ = clean_gappy_columns(prices)

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
