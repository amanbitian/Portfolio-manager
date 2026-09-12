from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_WINDOWS = (21, 63, 126)


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)


def build_feature_panel(prices: pd.DataFrame) -> pd.DataFrame:
    """Return a MultiIndex panel indexed by date and symbol.

    Features are shifted by one business day so the rebalance decision at date t
    only uses information known before that session's forward return.
    """
    returns = daily_returns(prices)
    frames: list[pd.DataFrame] = []

    for window in FEATURE_WINDOWS:
        momentum = prices.pct_change(window).shift(1)
        volatility = returns.rolling(window).std().shift(1) * np.sqrt(252)
        trend = (prices / prices.rolling(window).mean() - 1).shift(1)

        frames.append(_stack_feature(momentum, f"momentum_{window}d"))
        frames.append(_stack_feature(volatility, f"volatility_{window}d"))
        frames.append(_stack_feature(trend, f"trend_{window}d"))

    drawdown = (prices / prices.rolling(126).max() - 1).shift(1)
    frames.append(_stack_feature(drawdown, "drawdown_126d"))

    panel = pd.concat(frames, axis=1)
    panel = panel.replace([np.inf, -np.inf], np.nan)
    panel = panel.groupby(level="date", group_keys=False).apply(_cross_sectional_zscore)
    return panel.dropna(how="all").fillna(0.0)


def forward_returns(prices: pd.DataFrame, horizon: int) -> pd.Series:
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


def _cross_sectional_zscore(group: pd.DataFrame) -> pd.DataFrame:
    std = group.std(ddof=0).replace(0, np.nan)
    return (group - group.mean()) / std
