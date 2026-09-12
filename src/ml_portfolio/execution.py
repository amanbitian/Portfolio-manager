from __future__ import annotations

import numpy as np
import pandas as pd


def rebalance_cost_rate(target: pd.Series, current: pd.Series, cost_bps: float) -> float:
    """Solve fees against trades measured after fees have reduced investable equity."""
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
