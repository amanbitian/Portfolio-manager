from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from .data import clean_gappy_columns, eligibility_cutoffs, eligibility_starts


class MarketDataError(ValueError):
    """Raised when the real data lake is missing, misconfigured, or has no
    usable rows for a requested selection."""


def resolve_data_root(override: str | Path | None = None) -> Path:
    """Resolve the data lake root. Reuses data_pipeline's own STOCK_DATA_ROOT
    resolution (env var + default) instead of re-hardcoding the path here."""
    if override:
        return Path(override).expanduser()
    from data_pipeline.config import load_config
    return load_config().data_root


def load_universe(data_root: Path) -> pd.DataFrame:
    """Instrument identity table: symbol, isin, name, industry, and Upstox/
    yfinance keys. A snapshot of *current* Nifty Total Market membership only -
    no historical constituent history or ticker-rename tracking exists in the
    pipeline (see the universe_caveat in load_market_prices' report)."""
    path = Path(data_root) / "reference" / "universe.parquet"
    if not path.exists():
        raise MarketDataError(
            f"Universe file not found at {path}. Run the data pipeline's `universe` "
            "step first, or point Data root at an existing data lake."
        )
    universe = pd.read_parquet(path)
    return universe[universe["instrument_key"].notna()].sort_values("symbol").reset_index(drop=True)


def dataset_fingerprint(data_root: Path, symbols: list[str], start=None, end=None) -> str:
    """A cheap cache key: no data content is read. Changes when the ingestion
    manifest is updated (new pipeline run) or the selection changes - unlike
    storage.price_key, which hashes an entire loaded frame's content on every
    call and would be wasted cost here since the lake doesn't change mid-session."""
    manifest = Path(data_root) / "_manifests" / "market_1day.json"
    mtime = manifest.stat().st_mtime if manifest.exists() else 0
    payload = repr((str(data_root), sorted(symbols), str(start), str(end), mtime))
    return hashlib.sha256(payload.encode()).hexdigest()


def load_market_prices(
    data_root: str | Path,
    symbols: list[str],
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Read close prices for `symbols` from market/ohlcv_1day into the app's
    standard date-indexed price matrix. Column-pruned per symbol file (only
    `ts_ist`/`close` are read; OHLC/volume/open_interest are never loaded for
    this daily-price use case).

    Prices are already split/bonus-adjusted at source (Upstox) - do not
    re-adjust them. Symbols with no file, no rows in range, or a gap the
    forward-fill limit can't cover are excluded and reported, not silently
    dropped or forward-filled into looking tradable.
    """
    data_root = Path(data_root)
    if not data_root.exists():
        raise MarketDataError(f"Data root not found: {data_root}")

    series: dict[str, pd.Series] = {}
    excluded: dict[str, str] = {}
    for symbol in symbols:
        path = data_root / "market" / "ohlcv_1day" / f"symbol={symbol}" / "part.parquet"
        if not path.exists():
            excluded[symbol] = "no data file for this symbol"
            continue
        frame = pd.read_parquet(path, columns=["ts_ist", "close"])
        index = pd.DatetimeIndex(frame["ts_ist"]).tz_localize(None).normalize()
        closes = pd.Series(frame["close"].to_numpy(dtype=float), index=index)
        # A pipeline-format boundary artifact leaves ~431/750 symbols with one
        # duplicated calendar date (a midnight-timestamp row alongside a
        # same-day intraday-anchored row, identical close) - de-duplicate by
        # normalized date rather than assuming clean daily bars.
        closes = closes[~closes.index.duplicated(keep="last")]
        if start is not None:
            closes = closes[closes.index >= pd.Timestamp(start)]
        if end is not None:
            closes = closes[closes.index <= pd.Timestamp(end)]
        if closes.empty:
            excluded[symbol] = "no rows in the selected date range"
            continue
        series[symbol] = closes

    if not series:
        raise MarketDataError("None of the requested symbols have data in the selected range.")

    prices = pd.DataFrame(series).sort_index()
    prices.index.name = None
    prices, gap_excluded = clean_gappy_columns(prices)
    excluded.update(gap_excluded)

    if prices.shape[1] < 2:
        raise MarketDataError(
            "Fewer than two symbols have usable data for this selection. "
            "Widen the date range or choose different symbols."
        )

    coverage_end = prices.index.max()
    coverage_age_days = int((pd.Timestamp.now().normalize() - coverage_end.normalize()).days)
    report = {
        "data_root": str(data_root),
        "requested_symbols": len(symbols),
        "loaded_symbols": prices.shape[1],
        "excluded": excluded,
        "coverage_start": str(prices.index.min()),
        "coverage_end": str(coverage_end),
        # Honest disclosure, not a hard gate - blocking on data age is a
        # product decision beyond this fix, but the app previously gave no
        # signal at all about how stale a loaded dataset might be.
        "coverage_age_days": coverage_age_days,
        "price_basis": "close, already split/bonus-adjusted at source (Upstox) - do not re-adjust",
        "universe_caveat": (
            "Universe reflects current Nifty Total Market membership applied to all "
            "history; historical constituent changes and delisted-in-the-past names "
            "are not available (survivorship bias)."
        ),
        # Symbols kept via the trailing-gap carve-out in clean_gappy_columns:
        # real data up to this date, held flat and excluded from new
        # allocation after it (see backtest.py/strategy_copilot.py's use of
        # `prices.attrs["eligible_through"]`) rather than dropped from the
        # entire window - avoids a late-window gap silently affecting
        # early-window decisions.
        "truncated": {symbol: str(cutoff) for symbol, cutoff in eligibility_cutoffs(prices).items()},
        # Symbols that LISTED inside the window (leading gap): kept, with a
        # flat back-filled placeholder before this first real date, and not
        # allocatable until decisions.MIN_LISTING_HISTORY_DAYS real sessions
        # later. 266 of 750 names over 2018-2026 in this lake - previously
        # all silently dropped.
        "listed_mid_window": {symbol: str(start) for symbol, start in eligibility_starts(prices).items()},
        # Cells forward-filled to bridge a short gap (<=5 rows) - a filled
        # price is otherwise indistinguishable from an observed one to every
        # downstream calculation.
        "short_gaps_filled": prices.attrs.get("fill_counts", {}),
        "dataset_version": dataset_fingerprint(data_root, symbols, start, end),
    }
    return prices, report


def load_corporate_actions(data_root: str | Path, symbols: list[str]) -> pd.DataFrame:
    """Read-only corporate-action events (dividends/splits/bonuses) for
    display or audit. Not wired into any purchase-ledger cost-basis math -
    prices are already adjusted, and unit/cash reconciliation for a
    transaction ledger is separate, not-yet-built work."""
    data_root = Path(data_root)
    frames = []
    for symbol in symbols:
        path = data_root / "corporate_actions" / f"{symbol}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["symbol", "ex_date", "action_type", "ratio", "amount", "source", "split_adj_factor"])
