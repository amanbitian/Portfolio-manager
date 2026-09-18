from __future__ import annotations

import numpy as np
import pandas as pd

from .data import listed_only

FEATURE_WINDOWS = (21, 63, 126)


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)


def build_feature_panel(prices: pd.DataFrame) -> pd.DataFrame:
    """Return a MultiIndex panel indexed by date and symbol.

    Features are shifted by one business day so the rebalance decision at date t
    only uses information known before that session's forward return.

    Edge placeholders (a symbol's flat back-filled pre-listing cells or
    flat forward-filled post-cutoff cells, see data.clean_gappy_columns) are
    masked to NaN first: otherwise a newly listed name would enter the pooled
    ridge fit with a fabricated zero-momentum/zero-volatility feature row and
    a zero forward return for every pre-listing date, and its first real
    window would blend real and placeholder prints. With NaN, every rolling
    window warms up on real prints only, the cross-sectional z-scores skip
    the symbol, and an all-NaN symbol-day drops out of the panel entirely.
    """
    prices = listed_only(prices)
    # daily_returns fills NaN with 0.0 (a placeholder day would otherwise
    # read as a real zero-volatility day); keep the mask through the returns
    # too, and drop the first real print's NaN-based return as well.
    returns = daily_returns(prices).where(prices.notna() & prices.shift(1).notna())
    frames: list[pd.DataFrame] = []

    for window in FEATURE_WINDOWS:
        momentum = _wide_zscore(prices.pct_change(window).shift(1))
        volatility = _wide_zscore(returns.rolling(window).std().shift(1) * np.sqrt(252))
        trend = _wide_zscore((prices / prices.rolling(window).mean() - 1).shift(1))

        frames.append(_stack_feature(momentum, f"momentum_{window}d"))
        frames.append(_stack_feature(volatility, f"volatility_{window}d"))
        frames.append(_stack_feature(trend, f"trend_{window}d"))

    drawdown = _wide_zscore((prices / prices.rolling(126).max() - 1).shift(1))
    frames.append(_stack_feature(drawdown, "drawdown_126d"))

    panel = pd.concat(frames, axis=1)
    panel = panel.replace([np.inf, -np.inf], np.nan)
    return panel.dropna(how="all").fillna(0.0)


def forward_returns(prices: pd.DataFrame, horizon: int) -> pd.Series:
    # Same masking as build_feature_panel: a placeholder-to-placeholder
    # forward return is 0 by construction, not an observed outcome, and a
    # placeholder-to-real one is a fabricated jump; both become NaN targets,
    # which ridge_daily_statistics already drops.
    prices = listed_only(prices)
    future = prices.shift(-horizon) / prices - 1
    return future.stack().rename("forward_return").rename_axis(["date", "symbol"])


def feature_matrix_at(panel: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    try:
        return panel.xs(date, level="date")
    except KeyError:
        return pd.DataFrame(columns=panel.columns)


def _stack_feature(values: pd.DataFrame, name: str) -> pd.DataFrame:
    stacked = values.stack().rename(name).rename_axis(["date", "symbol"])
    return stacked.to_frame()


def _wide_zscore(values: pd.DataFrame) -> pd.DataFrame:
    mean = values.mean(axis=1)
    std = values.std(axis=1, ddof=0).replace(0, np.nan)
    return values.sub(mean, axis=0).div(std, axis=0)
