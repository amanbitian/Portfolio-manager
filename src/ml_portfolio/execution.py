from __future__ import annotations

import numpy as np
import pandas as pd


def rebalance_cost_rate(target: pd.Series, current: pd.Series, cost_bps: float) -> float:
    """Solve fees against trades measured after fees have reduced investable equity.

    Every caller (backtest.py, strategy_copilot.py, portfolio.py) builds `target`
    and `current` from the same `symbols + ["CASH"]` list, so their indexes are
    already aligned on every call. O03 profiling (1000 assets, daily rebalancing)
    found this function's original `reindex`/`Index.drop` pair cost ~0.39ms/call -
    almost 2x the walk-forward loop's own per-rebalance overhead - purely to
    relabel two Series that were already aligned. The fast path below skips that
    relabeling with a positional `np.delete`, ~2.9x faster measured; a mismatched
    index (nothing in this codebase produces one, but a future caller might) still
    falls back to the original reindex so the result stays correct either way.
    """
    if target.index.equals(current.index):
        cash_pos = target.index.get_loc("CASH")
        w = np.delete(target.to_numpy(), cash_pos)
        old = np.nan_to_num(np.delete(current.to_numpy(), cash_pos), nan=0.0)
    else:
        symbols = target.index.drop("CASH")
        w = target.reindex(symbols).to_numpy()
        old = current.reindex(symbols).fillna(0).to_numpy()
    fee = cost_bps / 10000
    if not np.isfinite(fee) or not 0 <= fee < 1:
        raise ValueError("Cost must be between zero and 10,000 bps.")
    low, high = 0.0, 1.0
    for _ in range(60):
        middle = (low + high) / 2
        cost = fee * np.abs(w * (1 - middle) - old).sum()
        if middle < cost:
            low = middle
        else:
            high = middle
    return (low + high) / 2 if fee else 0.0


def cutoff_positions(prices: pd.DataFrame, cutoffs: dict[str, pd.Timestamp]) -> dict[str, int]:
    """Map each symbol's eligibility cutoff date to its integer position in
    `prices.index` once, so the per-rebalance and per-interval checks in the
    walk-forward loops are integer comparisons rather than Timestamp searches."""
    return {symbol: int(prices.index.searchsorted(pd.Timestamp(cutoff), side="right") - 1)
            for symbol, cutoff in cutoffs.items() if symbol in prices.columns}


def start_positions(prices: pd.DataFrame, starts: dict[str, pd.Timestamp]) -> dict[str, int]:
    """Map each symbol's first-real-data date (`data.eligibility_starts`) to
    its integer position in `prices.index`; symbols with no leading gap are
    simply absent, which callers treat as position 0."""
    return {symbol: int(prices.index.searchsorted(pd.Timestamp(start), side="left"))
            for symbol, start in starts.items() if symbol in prices.columns}


def liquidate_at_cutoff(values: np.ndarray, symbols: list[str], cutoff_idx: dict[str, int],
                        idx: int, cost_bps: float) -> tuple[np.ndarray, list[str]]:
    """Force-sell any position still open in a symbol on the first day after
    its eligibility cutoff, inside one drift interval.

    `values` is the interval's buy-and-hold value matrix (one row per date
    `idx+1 .. idx+len(values)`, one column per symbol plus CASH last), in
    units of the equity at the rebalance on `idx`. A symbol kept via
    clean_gappy_columns's trailing-gap carve-out has real prices only through
    `cutoff_idx[symbol]`; every later cell is a flat-filled placeholder, so
    holding it there would freeze the position at zero return until the
    NEXT rebalance and charge the exit as ordinary rebalance turnover then.
    That is not what happens to a delisted or suspended-to-death name: the
    position is converted to cash at the last real price. So, from the first
    post-cutoff row, the column's (constant) value moves into CASH net of the
    transaction cost, mutating `values` in place.

    Returns the per-row cumulative fee (same units as `values`, zero before
    any liquidation) so a caller can keep the gross-of-fees path separate,
    plus the symbols liquidated in this interval. Positions already zero
    (the rebalance on `idx` sat past the cutoff, so the decision layer had
    already sent that weight to CASH) are left alone.
    """
    fees = np.zeros(len(values))
    liquidated: list[str] = []
    last_position = idx + len(values)
    for column, symbol in enumerate(symbols):
        cutoff = cutoff_idx.get(symbol)
        if cutoff is None or cutoff >= last_position:
            continue
        first_row = max(cutoff - idx, 0)
        proceeds = float(values[first_row, column])
        if proceeds <= 0.0:
            continue
        fee = proceeds * cost_bps / 10000
        values[first_row:, column] = 0.0
        values[first_row:, -1] += proceeds - fee
        fees[first_row:] += fee
        liquidated.append(symbol)
    return fees, liquidated
