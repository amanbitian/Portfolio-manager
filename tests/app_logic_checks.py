from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.append(str(ROOT / ".app-deps"))

import numpy as np
import pandas as pd

from ml_portfolio.backtest import BacktestConfig, calculate_metrics, run_walk_forward_backtest
from ml_portfolio.decisions import build_context, latest_decision, make_decision, validation_summary
from ml_portfolio.execution import rebalance_cost_rate
from ml_portfolio.optimizers import AllocationConfig, AllocationError, allocate_portfolio, risk_checks
from ml_portfolio.portfolio import build_proposal, save_proposal, saved_proposals


def price_fixture(rows=650):
    rng = np.random.default_rng(23)
    return pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(.0003, .012, (rows, 3)), axis=0)),
                        index=pd.bdate_range("2020-01-01", periods=rows), columns=["A", "B", "C"])


class AllocationChecks(unittest.TestCase):
    def setUp(self):
        self.mu = pd.Series([.02, .01], index=["A", "B"])
        self.cov = pd.DataFrame(np.diag([.01, .01]), index=self.mu.index, columns=self.mu.index)
        self.returns = pd.DataFrame({"A": [-.01, .01] * 63, "B": [-.02, .02] * 63})
        self.config = AllocationConfig(max_weight=1, cash_weight=0, risk_aversion=1, forecast_horizon_days=252)

    def test_inverse_volatility_proportions(self):
        w = allocate_portfolio(self.mu, self.cov, self.returns, replace(self.config, method="Inverse volatility"))
        np.testing.assert_allclose(w[["A", "B"]], [2/3, 1/3], atol=1e-6)

    def test_ml_known_optimum(self):
        w = allocate_portfolio(self.mu, self.cov, self.returns, replace(self.config, max_weight=.6))
        np.testing.assert_allclose(w[["A", "B"]], [.6, .4], atol=1e-6)

    def test_minimum_variance_analytic_solution(self):
        cov = self.cov.copy()
        cov.loc["B", "B"] = .04
        w = allocate_portfolio(self.mu, cov, self.returns, replace(self.config, method="Minimum variance"))
        np.testing.assert_allclose(w[["A", "B"]], [.8, .2], atol=1e-5)

    def test_risk_sector_and_turnover_constraints(self):
        config = replace(self.config, max_volatility=.025, max_turnover=.4,
                         sectors={"A": "Tech", "B": "Tech"}, sector_caps={"Tech": .3})
        old = pd.Series({"A": 0., "B": 0., "CASH": 1.})
        w = allocate_portfolio(self.mu, self.cov, self.returns, config, old)
        self.assertTrue(risk_checks(w, self.cov, config, old).passed.all())
        self.assertLessEqual(w.A + w.B, .300001)

    def test_incompatible_limits_raise(self):
        old = pd.Series({"A": 1., "B": 0., "CASH": 0.})
        with self.assertRaises(AllocationError):
            allocate_portfolio(self.mu, self.cov, self.returns,
                               replace(self.config, max_weight=.2, max_turnover=0), old)

    def test_fees_prevent_unprofitable_entry(self):
        mu = self.mu * .01
        cheap = allocate_portfolio(mu, self.cov, self.returns, self.config)
        costly = allocate_portfolio(mu, self.cov, self.returns, self.config, transaction_cost_bps=10)
        self.assertGreater(cheap.drop("CASH").sum(), .001)
        self.assertAlmostEqual(costly.CASH, 1, places=6)

    def test_no_trade_band_preserves_holdings(self):
        old = pd.Series({"A": .5, "B": .5, "CASH": 0.})
        mu = pd.Series([.006, .004], index=self.mu.index)
        w = allocate_portfolio(mu, self.cov, self.returns, self.config, old, 20)
        np.testing.assert_allclose(w, old, atol=1e-6)

    def test_caps_leave_cash(self):
        w = allocate_portfolio(self.mu, self.cov, self.returns,
                               replace(self.config, method="Equal weight", max_weight=.1))
        self.assertAlmostEqual(w.CASH, .8, places=6)

    def test_bad_method_rejected(self):
        with self.assertRaises(AllocationError):
            allocate_portfolio(self.mu, self.cov, self.returns, replace(self.config, method="typo"))


class DecisionChecks(unittest.TestCase):
    def test_validation_rewards_accuracy_not_extremes(self):
        rows = []
        for date in pd.bdate_range("2020-01-01", periods=3):
            for actual in [-.1, .1]:
                rows.append({"date": date, "prediction": actual * .9, "actual": actual, "baseline": 0.})
        records = pd.DataFrame(rows)
        good = validation_summary(records)
        bad = validation_summary(records.assign(prediction=-records.prediction * 10))
        self.assertTrue(good["eligible"])
        self.assertFalse(bad["eligible"])
        self.assertGreater(good["confidence"], bad["confidence"])
        self.assertFalse(validation_summary(records.iloc[:4])["eligible"])

    def test_decision_ignores_future_prices(self):
        prices = price_fixture()
        changed = prices.copy()
        changed.iloc[504:, 0] *= 4
        cfg = AllocationConfig(max_weight=.6)
        a = make_decision(build_context(prices, 21), 504, cfg, 504)
        b = make_decision(build_context(changed, 21), 504, cfg, 504)
        np.testing.assert_allclose(a["weights"], b["weights"])
        pd.testing.assert_frame_equal(a["validation"], b["validation"])
        pd.testing.assert_series_equal(a["expected_returns"], b["expected_returns"])

    def test_latest_uses_final_observation(self):
        prices = price_fixture()
        cfg = AllocationConfig(max_weight=.6)
        a = latest_decision(prices, cfg, 504)
        changed = prices.copy()
        changed.iloc[-1, 0] *= 1.2
        b = latest_decision(changed, cfg, 504)
        self.assertEqual(a["information_through"], prices.index[-1])
        self.assertGreater(a["date"], prices.index[-1])
        self.assertFalse(np.allclose(a["expected_returns"], b["expected_returns"]))

    def test_insufficient_evidence_falls_back(self):
        prices = price_fixture(240)
        decision = latest_decision(prices, AllocationConfig(), 166)
        self.assertEqual(decision["effective_method"], "Equal weight")
        self.assertEqual(decision["confidence"], 0)


class AccountingChecks(unittest.TestCase):
    def test_anchor_excluded_from_statistics(self):
        curve = pd.DataFrame({"equity": [1, 1.01, 1.0201], "return": [0, .01, .01]},
                             index=pd.bdate_range("2020-01-01", periods=3))
        metrics = calculate_metrics(curve, pd.DataFrame())
        self.assertEqual(metrics["Volatility"], 0)
        self.assertTrue(np.isnan(metrics["Sharpe"]))

    def test_initial_loss_drawdown(self):
        curve = pd.DataFrame({"equity": [1, .9], "return": [0, -.1]},
                             index=pd.bdate_range("2020-01-01", periods=2))
        self.assertAlmostEqual(calculate_metrics(curve, pd.DataFrame())["Max drawdown"], -.1)

    def test_holdings_drift(self):
        prices = pd.DataFrame(100., index=pd.bdate_range("2020-01-01", periods=264), columns=["A", "B"])
        prices.loc[prices.index[201]:, "A"] = 200
        result = run_walk_forward_backtest(prices, AllocationConfig(method="Equal weight", cash_weight=0, max_weight=1),
                                           BacktestConfig(train_days=200, rebalance_days=126, transaction_cost_bps=0))
        self.assertAlmostEqual(result["metrics"]["Final equity"], 1.5, places=6)
        np.testing.assert_allclose(result["final_holdings"][["A", "B"]], [2/3, 1/3], atol=1e-6)

    def test_self_financing_fee(self):
        target = pd.Series({"A": 1., "CASH": 0.})
        old = pd.Series({"A": 0., "CASH": 1.})
        self.assertAlmostEqual(rebalance_cost_rate(target, old, 100), .01/1.01, places=10)

    def test_cost_drag_tracks_gross_path(self):
        result = run_walk_forward_backtest(price_fixture(300), AllocationConfig(method="Equal weight", max_weight=.6),
                                           BacktestConfig(train_days=200, transaction_cost_bps=100))
        curve, metrics = result["equity_curve"], result["metrics"]
        self.assertAlmostEqual(metrics["Cost drag"], curve.gross_equity.iloc[-1] - curve.equity.iloc[-1])
        self.assertGreater(metrics["Fees paid"], 0)

    def test_short_evaluation_rejected(self):
        with self.assertRaises(ValueError):
            run_walk_forward_backtest(price_fixture(260), AllocationConfig(), BacktestConfig(train_days=258))

    def test_common_dates_independent_of_training_window(self):
        prices = price_fixture(400)
        cfg = AllocationConfig(method="Equal weight")
        start = str(prices.index[300].date())
        a = run_walk_forward_backtest(prices, cfg, BacktestConfig(train_days=200, evaluation_start=start))
        b = run_walk_forward_backtest(prices, cfg, BacktestConfig(train_days=252, evaluation_start=start))
        pd.testing.assert_index_equal(a["equity_curve"].index, b["equity_curve"].index)

    def test_minimum_variance_with_binding_risk_ceiling(self):
        result = run_walk_forward_backtest(price_fixture(400),
                                           AllocationConfig(method="Minimum variance", max_volatility=.05),
                                           BacktestConfig(train_days=200))
        self.assertGreater(len(result["weights"]), 5)
        self.assertTrue(np.isfinite(result["equity_curve"].equity).all())


class ProposalChecks(unittest.TestCase):
    def test_funding_and_snapshot_roundtrip(self):
        prices = price_fixture(300)
        qty = pd.Series({"A": 100, "B": 0, "C": 0})
        config = AllocationConfig(method="Equal weight", max_weight=.6)
        proposal = build_proposal(prices, qty, 100000, config, 200, 10)
        self.assertTrue(proposal["accepted"])
        self.assertAlmostEqual(proposal["capital_after"], proposal["capital"] - proposal["estimated_fees"])
        trades = proposal["trades"]
        self.assertAlmostEqual(proposal["cash_after"], 100000 - trades.trade_value.sum() - trades.estimated_fee.sum())
        with tempfile.TemporaryDirectory() as temp:
            path = save_proposal(Path(temp), proposal, prices, config, 200, 10)
            payload = json.loads(path.read_text())
            self.assertEqual(payload["schema_version"], 1)
            history = saved_proposals(Path(temp), prices.iloc[-1], prices.index[-1])
            self.assertAlmostEqual(history.iloc[0].marked_value, proposal["capital_after"])
            changed = prices * 2
            history = saved_proposals(Path(temp), changed.iloc[-1], changed.index[-1], changed)
            self.assertEqual(history.iloc[0].status, "Not comparable")

    def test_failed_rounding_cannot_be_saved(self):
        prices = pd.DataFrame(100., index=pd.bdate_range("2020-01-01", periods=300), columns=["A", "B"])
        proposal = build_proposal(prices, pd.Series({"A": 1, "B": 0}), 100,
                                  AllocationConfig(method="Equal weight", cash_weight=0, max_turnover=.25), 200, 0)
        self.assertFalse(proposal["accepted"])
        self.assertFalse(proposal["checks"].set_index("check").loc["One-way turnover", "passed"])
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                save_proposal(Path(temp), proposal, prices, AllocationConfig(), 200, 10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
