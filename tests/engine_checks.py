from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.append(str(Path(__file__).resolve().parents[1] / ".app-deps"))

from ml_portfolio.backtest import BacktestConfig, calculate_metrics, run_walk_forward_backtest
from ml_portfolio.data import make_synthetic_prices
from ml_portfolio.optimizers import AllocationConfig, allocate_portfolio
from ml_portfolio.reporting import drawdown_series


def test_future_prices_do_not_change_first_allocation() -> None:
    prices = make_synthetic_prices(periods=630, seed=42)
    config = AllocationConfig(
        method="ML mean-variance",
        max_weight=0.6,
        cash_weight=0.0,
        risk_aversion=6.0,
        forecast_horizon_days=21,
    )
    backtest_config = BacktestConfig(
        train_days=504,
        rebalance_days=21,
        forward_days=21,
        transaction_cost_bps=10,
    )

    baseline = run_walk_forward_backtest(prices, config, backtest_config)
    changed = prices.copy()
    changed.iloc[505:526, 0] *= 1.5
    changed_result = run_walk_forward_backtest(changed, config, backtest_config)

    baseline_first = baseline["weights"].iloc[0].sort_index()
    changed_first = changed_result["weights"].iloc[0].sort_index()
    assert float((baseline_first - changed_first).abs().max()) == 0.0


def test_infeasible_risky_cap_moves_residual_to_cash() -> None:
    symbols = ["A", "B", "C", "D"]
    mu = pd.Series([0.04, 0.03, 0.02, 0.01], index=symbols)
    cov = pd.DataFrame(np.diag([0.04, 0.05, 0.06, 0.07]), index=symbols, columns=symbols)
    trailing = pd.DataFrame(np.ones((126, 4)) * 0.01, columns=symbols)

    weights = allocate_portfolio(
        mu,
        cov,
        trailing,
        AllocationConfig(method="Equal weight", max_weight=0.1, cash_weight=0.0),
    )

    assert abs(float(weights.sum()) - 1.0) < 1e-12
    assert float(weights.drop("CASH").max()) <= 0.1 + 1e-12
    assert abs(float(weights["CASH"]) - 0.6) < 1e-12


def test_confidence_and_risk_change_ml_exposure() -> None:
    symbols = ["A", "B", "C", "D"]
    mu = pd.Series([0.004, 0.003, 0.002, 0.001], index=symbols)
    cov = pd.DataFrame(np.diag([0.04, 0.05, 0.06, 0.07]), index=symbols, columns=symbols)
    trailing = pd.DataFrame(np.ones((126, 4)) * 0.01, columns=symbols)

    low_confidence = allocate_portfolio(
        mu * 0.1,
        cov,
        trailing,
        AllocationConfig(method="ML mean-variance", max_weight=1.0, cash_weight=0.0, risk_aversion=6.0),
    )
    high_confidence = allocate_portfolio(
        mu,
        cov,
        trailing,
        AllocationConfig(method="ML mean-variance", max_weight=1.0, cash_weight=0.0, risk_aversion=6.0),
    )
    low_risk_aversion = allocate_portfolio(
        mu,
        cov,
        trailing,
        AllocationConfig(method="ML mean-variance", max_weight=1.0, cash_weight=0.0, risk_aversion=2.0),
    )
    high_risk_aversion = allocate_portfolio(
        mu,
        cov,
        trailing,
        AllocationConfig(method="ML mean-variance", max_weight=1.0, cash_weight=0.0, risk_aversion=20.0),
    )

    assert float(high_confidence.drop("CASH").sum()) > float(low_confidence.drop("CASH").sum())
    assert float(low_risk_aversion.drop("CASH").sum()) > float(high_risk_aversion.drop("CASH").sum())


def test_drawdown_includes_initial_capital() -> None:
    dates = pd.to_datetime(["2024-01-01", "2024-01-02"])
    equity_curve = pd.DataFrame({"equity": [1.0, 0.9], "return": [0.0, -0.1]}, index=dates)

    assert float(drawdown_series(equity_curve).min()) == -0.09999999999999998
    assert calculate_metrics(equity_curve, pd.DataFrame())["Max drawdown"] == -0.09999999999999998


if __name__ == "__main__":
    checks = [
        test_future_prices_do_not_change_first_allocation,
        test_infeasible_risky_cap_moves_residual_to_cash,
        test_confidence_and_risk_change_ml_exposure,
        test_drawdown_includes_initial_capital,
    ]
    for check in checks:
        check()
    print(f"{len(checks)} engine checks passed")
