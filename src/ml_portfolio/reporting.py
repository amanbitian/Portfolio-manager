from __future__ import annotations

import numpy as np
import pandas as pd


def metric_table(results: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for name, result in results.items():
        row = {"strategy": name}
        row.update(result["metrics"])
        rows.append(row)
    table = pd.DataFrame(rows)
    return table.sort_values("Sharpe", ascending=False)


def latest_weights(weights: pd.DataFrame) -> pd.DataFrame:
    if weights.empty:
        return pd.DataFrame(columns=["asset", "weight"])
    latest = weights.iloc[-1].sort_values(ascending=False)
    return latest.rename_axis("asset").reset_index(name="weight")


def drawdown_series(equity_curve: pd.DataFrame) -> pd.Series:
    if equity_curve.empty:
        return pd.Series(dtype=float, name="drawdown")
    running_peak = equity_curve["equity"].cummax().clip(lower=1.0)
    return (equity_curve["equity"] / running_peak - 1).rename("drawdown")


def format_percent(value: float) -> str:
    return f"{value:.2%}"


def return_contributions(result: dict, prices: pd.DataFrame) -> pd.Series:
    """Per-symbol contribution to the backtest's GROSS return over the whole
    evaluation window, dollar-exact.

    For each day t, contribution_i,t = gross_equity[t-1] * start_of_day_weight_i
    * (price_relative_i,t - 1). Summed across assets this equals
    gross_equity[t] - gross_equity[t-1] exactly (weights sum to 1, so
    sum_i(w_i * relative_i) - 1 = gross growth minus one), and summed across
    the whole window it equals gross_equity[-1] - 1 - the total gross return -
    NOT the net/CAGR return, since transaction cost is a single portfolio-level
    scalar (execution.rebalance_cost_rate) never attributed per asset. Net
    return = gross return - metrics["Cost drag"] (already computed elsewhere).

    start_of_day_weight is `holdings` shifted by one row, EXCEPT on the first
    day following each rebalance, where it must be the just-decided `weights`
    target (holdings' own row at the decision date is the pre-trade weight,
    not the new target) - reusing holdings.shift(1) there would silently
    misattribute contributions at every rebalance boundary.
    """
    holdings = result["holdings"]
    weights = result["weights"]
    equity_curve = result["equity_curve"]
    dated = holdings.index[1:]  # exclude the initial-capital anchor row

    start_of_day = holdings.shift(1).loc[dated]
    decision_positions = prices.index.get_indexer(weights.index)
    valid = decision_positions >= 0
    first_day_dates = prices.index[decision_positions[valid] + 1]
    targets = weights.loc[weights.index[valid]].reindex(columns=start_of_day.columns)
    targets.index = first_day_dates
    start_of_day.loc[first_day_dates] = targets

    relatives = prices.reindex(dated) / prices.shift(1).reindex(dated)
    relatives["CASH"] = 1.0
    relatives = relatives.reindex(columns=start_of_day.columns)

    prior_gross_equity = equity_curve["gross_equity"].shift(1).loc[dated]
    contributions = start_of_day.mul(relatives - 1).mul(prior_gross_equity, axis=0)
    return contributions.sum().sort_values(ascending=False).rename("contribution")


def concentration(weights: pd.Series, top_n: tuple[int, ...] = (3, 5)) -> dict:
    """HHI (sum of squared weight shares) and top-N weight share, computed
    over non-CASH positions only - cash isn't a concentration risk."""
    risky = weights.drop("CASH", errors="ignore")
    risky = risky[risky > 0]
    total = float(risky.sum())
    if total <= 0:
        return {"hhi": 0.0, **{f"top_{n}_share": 0.0 for n in top_n}}
    shares = (risky / total).sort_values(ascending=False)
    result = {"hhi": float((shares ** 2).sum())}
    for n in top_n:
        result[f"top_{n}_share"] = float(shares.head(n).sum())
    return result


def weight_drift(current: pd.Series, reference: pd.Series) -> pd.Series:
    """current - reference vs. any reference (equal-weight, a prior rebalance
    target, ...), reindexed to their union with missing values treated as 0."""
    index = current.index.union(reference.index)
    return (current.reindex(index).fillna(0) - reference.reindex(index).fillna(0)).rename("drift")


def downsample_preserving_extrema(series: pd.Series, max_points: int = 2000) -> pd.Series:
    """Downsample a long series for charting while preserving its min and max
    value (e.g. drawdown extrema) - splits the series into `max_points // 2`
    buckets and keeps each bucket's min AND max index, rather than a naive
    stride/every-Nth-point sample, which can silently skip past the actual
    peak or trough. A no-op when the series already fits within max_points."""
    if len(series) <= max_points:
        return series
    buckets = max(1, max_points // 2)
    edges = np.linspace(0, len(series), buckets + 1).astype(int)
    keep = set()
    for i in range(buckets):
        chunk = series.iloc[edges[i]:edges[i + 1]]
        if chunk.empty:
            continue
        keep.add(chunk.index[int(np.argmin(chunk.to_numpy()))])
        keep.add(chunk.index[int(np.argmax(chunk.to_numpy()))])
    return series.loc[sorted(keep)]
