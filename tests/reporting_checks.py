from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.append(str(ROOT / ".app-deps"))

import numpy as np
import pandas as pd

from ml_portfolio.backtest import BacktestConfig, run_walk_forward_backtest
from ml_portfolio.optimizers import AllocationConfig
from ml_portfolio.reporting import concentration, downsample_preserving_extrema, return_contributions, weight_drift


def price_fixture(rows=400, n_assets=4, seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=rows)
    returns = rng.normal(0.0003, 0.012, (rows, n_assets))
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    return pd.DataFrame(prices, index=dates, columns=[f"S{i}" for i in range(n_assets)])


class ReturnContributionChecks(unittest.TestCase):
    def test_contributions_reconcile_to_gross_return(self):
        prices = price_fixture()
        config = AllocationConfig(method="Equal weight", max_weight=0.6)
        backtest = BacktestConfig(train_days=200, rebalance_days=21, transaction_cost_bps=15)
        result = run_walk_forward_backtest(prices, config, backtest)

        contributions = return_contributions(result, prices)
        gross_total_return = float(result["equity_curve"]["gross_equity"].iloc[-1]) - 1.0
        self.assertAlmostEqual(float(contributions.sum()), gross_total_return, places=9)

        net_total_return = float(result["equity_curve"]["equity"].iloc[-1]) - 1.0
        cost_drag = result["metrics"]["Cost drag"]
        self.assertAlmostEqual(gross_total_return - cost_drag, net_total_return, places=9)

    def test_cash_contributes_zero(self):
        prices = price_fixture()
        config = AllocationConfig(method="Equal weight", max_weight=0.6, cash_weight=0.1)
        backtest = BacktestConfig(train_days=200, rebalance_days=21, transaction_cost_bps=0)
        result = run_walk_forward_backtest(prices, config, backtest)
        contributions = return_contributions(result, prices)
        self.assertAlmostEqual(float(contributions.get("CASH", 0.0)), 0.0, places=9)

    def test_reconciles_across_multiple_rebalance_boundaries(self):
        # A shorter rebalance interval exercises the weights-vs-holdings.shift(1)
        # override at many more period boundaries - the case that would break
        # first if the override logic were wrong.
        prices = price_fixture(rows=460, n_assets=6)
        config = AllocationConfig(method="Inverse volatility", max_weight=0.5)
        backtest = BacktestConfig(train_days=210, rebalance_days=5, transaction_cost_bps=8)
        result = run_walk_forward_backtest(prices, config, backtest)
        contributions = return_contributions(result, prices)
        gross_total_return = float(result["equity_curve"]["gross_equity"].iloc[-1]) - 1.0
        self.assertAlmostEqual(float(contributions.sum()), gross_total_return, places=8)


class ConcentrationChecks(unittest.TestCase):
    def test_four_equal_weighted_positions(self):
        weights = pd.Series({"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25, "CASH": 0.0})
        result = concentration(weights)
        self.assertAlmostEqual(result["hhi"], 0.25)
        self.assertAlmostEqual(result["top_3_share"], 0.75)

    def test_concentrated_single_position(self):
        weights = pd.Series({"A": 0.9, "B": 0.1, "CASH": 0.0})
        result = concentration(weights)
        self.assertAlmostEqual(result["hhi"], 0.9 ** 2 + 0.1 ** 2)

    def test_all_cash_reports_zero_concentration(self):
        weights = pd.Series({"CASH": 1.0})
        result = concentration(weights)
        self.assertEqual(result["hhi"], 0.0)


class WeightDriftChecks(unittest.TestCase):
    def test_hand_computed_drift_vs_equal_weight(self):
        current = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
        reference = pd.Series({"A": 1 / 3, "B": 1 / 3, "C": 1 / 3})
        drift = weight_drift(current, reference)
        self.assertAlmostEqual(drift["A"], 0.5 - 1 / 3)
        self.assertAlmostEqual(drift["B"], 0.3 - 1 / 3)
        self.assertAlmostEqual(drift["C"], 0.2 - 1 / 3)

    def test_missing_symbol_treated_as_zero(self):
        current = pd.Series({"A": 1.0})
        reference = pd.Series({"A": 0.5, "B": 0.5})
        drift = weight_drift(current, reference)
        self.assertAlmostEqual(drift["A"], 0.5)
        self.assertAlmostEqual(drift["B"], -0.5)


class DownsampleChecks(unittest.TestCase):
    def test_noop_when_already_short(self):
        series = pd.Series(np.arange(10.0))
        pd.testing.assert_series_equal(downsample_preserving_extrema(series, max_points=2000), series)

    def test_preserves_interior_extrema_missed_by_naive_stride(self):
        # A naive every-Nth-point stride sample would land on multiples of 100
        # and completely miss the single spike/trough placed off that stride.
        n = 5000
        series = pd.Series(np.sin(np.linspace(0, 20, n)))
        series.iloc[2503] = 100.0    # far off any round stride
        series.iloc[3771] = -100.0
        reduced = downsample_preserving_extrema(series, max_points=200)
        self.assertLessEqual(len(reduced), 200 + 2)
        self.assertAlmostEqual(reduced.max(), 100.0)
        self.assertAlmostEqual(reduced.min(), -100.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
