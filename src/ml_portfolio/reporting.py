from __future__ import annotations

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
