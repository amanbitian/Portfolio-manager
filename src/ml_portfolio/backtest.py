from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .decisions import build_context, make_decision
from .execution import rebalance_cost_rate
from .optimizers import AllocationConfig, normalize_holdings


@dataclass(frozen=True)
class BacktestConfig:
    train_days: int = 504
    rebalance_days: int = 21
    forward_days: int = 21
    transaction_cost_bps: float = 10.0
    evaluation_start: str | None = None
    evaluation_end: str | None = None
    min_evaluation_days: int = 63


def evaluation_bounds(prices: pd.DataFrame, config: BacktestConfig) -> tuple[int, int]:
    if config.train_days < 166 or config.rebalance_days < 1 or config.forward_days < 1:
        raise ValueError("Training needs at least 166 rows; rebalance and forecast horizons must be positive.")
    if not np.isfinite(config.transaction_cost_bps) or not 0 <= config.transaction_cost_bps < 10000:
        raise ValueError("Transaction cost must be between zero and 10,000 bps.")
    if config.min_evaluation_days < 63:
        raise ValueError("Evaluation requires at least 63 realized daily returns.")
    if not isinstance(prices.index, pd.DatetimeIndex) or not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError("Backtest requires unique, increasing dates.")
    if prices.shape[1] < 2 or not prices.columns.is_unique or "CASH" in prices.columns:
        raise ValueError("Backtest requires at least two unique risky assets.")
    if not np.isfinite(prices.to_numpy(dtype=float)).all() or (prices <= 0).any().any():
        raise ValueError("Backtest prices must be finite and positive.")
    start = int(prices.index.searchsorted(pd.Timestamp(config.evaluation_start))) if config.evaluation_start else config.train_days
    end = int(prices.index.searchsorted(pd.Timestamp(config.evaluation_end), side="right") - 1) if config.evaluation_end else len(prices) - 1
    if start < config.train_days:
        raise ValueError("The evaluation start must leave the full selected training history.")
    if end - start < config.min_evaluation_days or start >= len(prices) or end < 0:
        raise ValueError(f"Select an evaluation period with at least {config.min_evaluation_days} daily returns after training.")
    return start, end


def run_walk_forward_backtest(prices: pd.DataFrame, allocation_config: AllocationConfig,
                              backtest_config: BacktestConfig) -> dict:
    start, end = evaluation_bounds(prices, backtest_config)
    prices = prices.iloc[:end + 1].copy()
    symbols = list(prices.columns)
    context = build_context(prices, backtest_config.forward_days, allocation_config.method == "ML mean-variance")
    current = normalize_holdings(None, symbols)
    equity = gross_equity = 1.0
    points = [{"date": prices.index[start], "equity": 1.0, "gross_equity": 1.0, "return": np.nan}]
    records, targets, holdings, forecasts, validations = [], [], [], [], []
    holdings.append(current.rename(prices.index[start]))
    for idx in range(start, end, backtest_config.rebalance_days):
        decision = make_decision(context, idx, allocation_config, backtest_config.train_days,
                                 current, backtest_config.transaction_cost_bps)
        weights = decision["weights"]
        turnover = float((weights - current).abs().sum() / 2)
        asset_turnover = float((weights.drop("CASH") - current.drop("CASH")).abs().sum())
        cost_rate = rebalance_cost_rate(weights, current, backtest_config.transaction_cost_bps)
        cost_value = equity * cost_rate
        before = equity
        current = weights.copy()
        next_idx = min(idx + backtest_config.rebalance_days, end)
        for offset, (date, returns) in enumerate(context.returns.iloc[idx+1:next_idx+1].iterrows()):
            gross_return = float(current.reindex(symbols).dot(returns))
            net_return = (1 - cost_rate) * (1 + gross_return) - 1 if offset == 0 else gross_return
            equity *= 1 + net_return
            gross_equity *= 1 + gross_return
            points.append({"date": date, "equity": equity, "gross_equity": gross_equity, "return": net_return})
            current.loc[symbols] *= 1 + returns
            current /= current.sum()
            holdings.append(current.rename(date))
        records.append({"date": decision["date"], "train_start": prices.index[idx-backtest_config.train_days],
                        "train_end": decision["information_through"], "period_return": equity / before - 1,
                        "turnover": turnover, "asset_turnover": asset_turnover, "cost": cost_rate,
                        "cost_value": cost_value, "confidence": decision["confidence"],
                        "regime": decision["regime"], "effective_method": decision["effective_method"],
                        "reason": decision["reason"], "solver_iterations": decision["solver"]["iterations"],
                        "validation_blocks": decision["validation_summary"]["blocks"], **decision["regime_stats"]})
        targets.append(weights.rename(decision["date"]))
        if allocation_config.method == "ML mean-variance":
            actual = (prices.iloc[idx+backtest_config.forward_days] / prices.iloc[idx] - 1
                      if idx+backtest_config.forward_days <= end else pd.Series(np.nan, index=symbols))
            forecasts.append(pd.DataFrame({"date": decision["date"], "symbol": symbols,
                                           "expected_return": decision["expected_returns"].to_numpy(),
                                           "adjusted_return": decision["adjusted_returns"].to_numpy(),
                                           "actual_return": actual.to_numpy(), "horizon_days": backtest_config.forward_days,
                                           "annualized_volatility": np.sqrt(np.diag(decision["covariance"])),
                                           "effective_method": decision["effective_method"]}))
            if not decision["validation"].empty:
                validations.append(decision["validation"].assign(decision_date=decision["date"]))
    curve = pd.DataFrame(points).set_index("date")
    rebalances = pd.DataFrame(records)
    return {"equity_curve": curve, "rebalances": rebalances, "weights": pd.DataFrame(targets),
            "holdings": pd.DataFrame(holdings), "final_holdings": current,
            "forecasts": pd.concat(forecasts, ignore_index=True) if forecasts else pd.DataFrame(),
            "validation": pd.concat(validations, ignore_index=True) if validations else pd.DataFrame(),
            "metrics": calculate_metrics(curve, rebalances)}


def calculate_metrics(equity_curve: pd.DataFrame, rebalances: pd.DataFrame) -> dict[str, float]:
    if equity_curve.empty or len(equity_curve) < 2:
        return {name: float("nan") for name in ("CAGR", "Volatility", "Sharpe", "Max drawdown",
                                                "Average turnover", "Cost drag", "Fees paid", "Final equity")}
    # The first row is initial capital, not an observed holding-period return.
    returns = equity_curve["return"].iloc[1:].dropna()
    years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
    initial, final = float(equity_curve.equity.iloc[0]), float(equity_curve.equity.iloc[-1])
    volatility = float(returns.std(ddof=1) * np.sqrt(252)) if len(returns) >= 2 else float("nan")
    if np.isfinite(volatility) and volatility < 1e-12:
        volatility = 0.0
    sharpe = float(returns.mean() * 252 / volatility) if volatility > 0 else float("nan")
    drawdown = equity_curve.equity / equity_curve.equity.cummax() - 1
    gross = float(equity_curve.gross_equity.iloc[-1]) if "gross_equity" in equity_curve else float("nan")
    return {"CAGR": (final / initial) ** (1 / years) - 1 if years > 0 else float("nan"),
            "Volatility": volatility, "Sharpe": sharpe, "Max drawdown": float(drawdown.min()),
            "Average turnover": float(rebalances.turnover.mean()) if not rebalances.empty else 0.0,
            "Cost drag": (gross - final) / initial,
            "Fees paid": float(rebalances.cost_value.sum()) / initial if "cost_value" in rebalances else 0.0,
            "Final equity": final / initial}
