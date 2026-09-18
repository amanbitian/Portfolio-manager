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


def listing_fixture(rows=700, listed_at=150):
    """Three seasoned names plus NEW, which lists at row `listed_at`. NEW's
    pre-listing placeholder is flat (as clean_gappy_columns leaves it) and
    its real history is deliberately the CALMEST, highest-drift series in the
    frame - so any risk model that trusted the placeholder, or even just
    liked a smooth young name, would pile into it."""
    prices = price_fixture(rows)
    rng = np.random.default_rng(7)
    real = 50 * np.exp(np.cumsum(rng.normal(.0008, .004, rows - listed_at)))
    prices["NEW"] = np.r_[np.full(listed_at, real[0]), real]
    prices.attrs["eligible_from"] = {"NEW": prices.index[listed_at].isoformat()}
    return prices


def large_price_fixture(rows=260, columns=1000):
    rng = np.random.default_rng(41)
    symbols = [f"S{i:04d}" for i in range(columns)]
    returns = rng.normal(.0002, .015, (rows, columns))
    return pd.DataFrame(100 * np.exp(np.cumsum(returns, axis=0)),
                        index=pd.bdate_range("2020-01-01", periods=rows), columns=symbols)


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

    def test_large_universe_uses_fast_path(self):
        symbols = [f"S{i:04d}" for i in range(1000)]
        mu = pd.Series(0.0, index=symbols)
        cov = pd.DataFrame(np.diag(np.full(1000, .04)), index=symbols, columns=symbols)
        trailing = pd.DataFrame(np.ones((126, 1000)) * .001, columns=symbols)
        weights = allocate_portfolio(mu, cov, trailing,
                                     AllocationConfig(method="Equal weight", max_weight=.02, cash_weight=.05))
        self.assertEqual(weights.attrs["solver"]["mode"], "convex")
        self.assertEqual(len(weights), 1001)
        self.assertAlmostEqual(weights.CASH, .05, places=8)
        self.assertLessEqual(float(weights.drop("CASH").max()), .02)
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=8)


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

    def test_large_universe_latest_decision_runs(self):
        prices = large_price_fixture()
        decision = latest_decision(prices, AllocationConfig(max_weight=.02, cash_weight=.05), 166)
        self.assertEqual(len(decision["weights"]), 1001)
        self.assertEqual(decision["solver"]["mode"], "convex")
        self.assertTrue(decision["checks"].passed.all())


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

    def test_universe_eligibility_cutoff_zeroes_weight_after_but_not_before(self):
        # Regression test for the lookahead-bias fix: a symbol carrying
        # `.attrs["eligible_through"]` (set by clean_gappy_columns's
        # trailing-gap carve-out) must receive real weight on rebalances
        # before its cutoff and exactly zero weight at/after it - the freed
        # weight moves to CASH, and a decision made before the cutoff must
        # never be affected by it.
        prices = price_fixture(400)
        cutoff = prices.index[300]
        prices.attrs["eligible_through"] = {"C": cutoff}
        result = run_walk_forward_backtest(
            prices, AllocationConfig(method="Equal weight", max_weight=1.0, cash_weight=0.0),
            BacktestConfig(train_days=200, rebalance_days=21))
        weights = result["weights"]
        before = weights.loc[weights.index <= cutoff]
        after = weights.loc[weights.index > cutoff]
        self.assertGreater(len(before), 0)
        self.assertGreater(len(after), 0)
        self.assertTrue((before["C"] > 0).any())
        self.assertTrue((after["C"] == 0).all())
        self.assertTrue((after["CASH"] > 0).all())

    def test_open_position_is_liquidated_to_cash_the_day_after_its_cutoff(self):
        # A position still open when its eligibility cutoff falls INSIDE a
        # drift interval must be sold to CASH at the last real price on the
        # first post-cutoff day - fee charged there - not frozen at zero
        # return until the next rebalance (the previously disclosed
        # simplification). Rebalances fall on 200 + 21k here, so a cutoff at
        # position 315 sits mid-interval: the position was legitimately opened
        # on the 305 rebalance and the liquidation has to happen mid-drift.
        prices = price_fixture(400)
        rebalance_days, cost_bps = 21, 50
        cutoff = prices.index[315]
        prices.attrs["eligible_through"] = {"C": cutoff}
        prices.loc[prices.index > cutoff, "C"] = prices.loc[cutoff, "C"]  # flat-filled, as clean_gappy_columns leaves it
        result = run_walk_forward_backtest(
            prices, AllocationConfig(method="Equal weight", max_weight=1.0, cash_weight=0.0),
            BacktestConfig(train_days=200, rebalance_days=rebalance_days, transaction_cost_bps=cost_bps))
        holdings, rebalances = result["holdings"], result["rebalances"].set_index("date")
        opening = rebalances.index[rebalances.index <= cutoff][-1]
        self.assertGreater(result["weights"].loc[opening, "C"], 0)
        self.assertEqual(rebalances.loc[opening, "liquidated"], "C")
        self.assertEqual(rebalances.loc[rebalances.index != opening, "liquidated"].eq("").all(), True)
        first_after = prices.index[prices.index.get_loc(cutoff) + 1]
        self.assertGreater(holdings.loc[cutoff, "C"], 0)          # still held on the last real day
        self.assertEqual(holdings.loc[first_after, "C"], 0.0)     # sold the next day
        self.assertTrue((holdings.loc[holdings.index > cutoff, "C"] == 0).all())
        self.assertGreater(holdings.loc[first_after, "CASH"], holdings.loc[cutoff, "CASH"])
        # Economics: proceeds land in CASH net of the fee, the fee shows up
        # in that rebalance's fees (not the next one's turnover), and the
        # gross path excludes it - so Cost drag still measures fees only.
        curve = result["equity_curve"]
        pre, post = curve.loc[cutoff], curve.loc[first_after]
        liquidated_share = holdings.loc[cutoff, "C"]
        # A and B still move that day, so compare net against gross: the only
        # thing separating the two one-day ratios is the liquidation fee on
        # C's share of the book.
        self.assertAlmostEqual(post.gross_equity / pre.gross_equity - post.equity / pre.equity,
                               liquidated_share * cost_bps / 10000, places=10)
        expected_fee = pre.equity * liquidated_share * cost_bps / 10000
        self.assertGreater(rebalances.loc[opening, "cost_value"], expected_fee)
        self.assertAlmostEqual(result["metrics"]["Cost drag"],
                               curve.gross_equity.iloc[-1] - curve.equity.iloc[-1], places=10)
        # The next rebalance starts from C already at zero, so the exit is
        # not billed a second time as turnover: what remains is only the
        # ordinary A/B/CASH re-equalisation, far smaller than C's old share.
        following = rebalances.index[rebalances.index > cutoff][0]
        day_before = prices.index[prices.index.get_loc(following) - 1]
        self.assertEqual(result["weights"].loc[following, "C"], 0.0)
        self.assertEqual(holdings.loc[day_before, "C"], 0.0)
        self.assertLess(rebalances.loc[following, "turnover"], liquidated_share / 4)

    def test_position_opened_on_the_cutoff_day_itself_is_liquidated_next_day(self):
        # On the cutoff day nobody yet knows it is the last real print, so a
        # rebalance landing exactly there may still buy the name; it must
        # then be sold on the very next day, not held until the next
        # rebalance.
        prices = price_fixture(400)
        cutoff = prices.index[305]  # 200 + 5 * 21: exactly a rebalance date
        prices.attrs["eligible_through"] = {"C": cutoff}
        prices.loc[prices.index > cutoff, "C"] = prices.loc[cutoff, "C"]
        result = run_walk_forward_backtest(
            prices, AllocationConfig(method="Equal weight", max_weight=1.0, cash_weight=0.0),
            BacktestConfig(train_days=200, rebalance_days=21, transaction_cost_bps=50))
        rebalances = result["rebalances"].set_index("date")
        self.assertIn(cutoff, rebalances.index)
        self.assertGreater(result["weights"].loc[cutoff, "C"], 0)
        self.assertEqual(rebalances.loc[cutoff, "liquidated"], "C")
        holdings = result["holdings"]
        self.assertTrue((holdings.loc[holdings.index > cutoff, "C"] == 0).all())

    def test_truncated_prices_survive_the_persistent_feature_cache(self):
        # Regression: pandas JSON-encodes `.attrs` into parquet metadata, and
        # `daily_returns`/`build_feature_panel` propagate attrs from the price
        # frame, so a Timestamp cutoff in `eligible_through` made the
        # persistent cache's `to_parquet` raise TypeError on every ML
        # proposal/backtest for any real-data window with a mid-window
        # delisting. The cutoff is now stored JSON-natively and read back via
        # `eligibility_cutoffs`, so it also survives the round trip.
        from ml_portfolio.data import clean_gappy_columns, eligibility_cutoffs
        raw = price_fixture(400)
        raw.loc[raw.index[350]:, "C"] = np.nan
        prices, excluded = clean_gappy_columns(raw)
        self.assertEqual(excluded, {})
        # The bounded 5-row forward fill extends the last real observation by 5 rows first.
        self.assertEqual(eligibility_cutoffs(prices), {"C": raw.index[354]})
        with tempfile.TemporaryDirectory() as folder:
            context = build_context(prices, 21, cache_dir=Path(folder))
            self.assertEqual(eligibility_cutoffs(context.returns), {"C": raw.index[354]})
            # Second build reads the cache back instead of rebuilding it.
            reloaded = build_context(prices, 21, cache_dir=Path(folder))
            pd.testing.assert_frame_equal(reloaded.returns, context.returns, check_freq=False)
            self.assertEqual(eligibility_cutoffs(reloaded.returns), {"C": raw.index[354]})
            result = run_walk_forward_backtest(
                prices, AllocationConfig(method="ML mean-variance", max_weight=1.0, cash_weight=0.0),
                BacktestConfig(train_days=200, rebalance_days=21), cache_dir=Path(folder))
        after = result["weights"].loc[result["weights"].index > raw.index[354]]
        self.assertGreater(len(after), 0)
        self.assertTrue((after["C"] == 0).all())

    def test_newly_listed_symbol_is_not_a_candidate_until_seasoned(self):
        # The core proof for mid-window listings: NEW gets exactly zero weight
        # on every rebalance before listing + MIN_LISTING_HISTORY_DAYS, then
        # real weight afterwards - and, because it is excluded from the
        # solver's candidate set rather than zeroed after the fact, the
        # minimum-variance solver never even sees its zero-volatility
        # placeholder. The benchmark, forecasts and the decision record all
        # disclose the same thing.
        from ml_portfolio.decisions import MIN_LISTING_HISTORY_DAYS
        prices = listing_fixture(rows=700, listed_at=150)
        eligible_date = prices.index[150 + MIN_LISTING_HISTORY_DAYS]
        for method in ("Minimum variance", "ML mean-variance"):
            with self.subTest(method=method):
                with tempfile.TemporaryDirectory() as folder:
                    result = run_walk_forward_backtest(
                        prices, AllocationConfig(method=method, max_weight=1.0, cash_weight=0.0),
                        BacktestConfig(train_days=200, rebalance_days=21), cache_dir=Path(folder))
                weights = result["weights"]
                before = weights.loc[weights.index < eligible_date, "NEW"]
                after = weights.loc[weights.index >= eligible_date, "NEW"]
                self.assertGreater(len(before), 3)
                self.assertGreater(len(after), 3)
                self.assertTrue((before == 0).all())
                self.assertGreater(after.max(), 0.05)
                holdings = result["holdings"]
                self.assertTrue((holdings.loc[holdings.index < eligible_date, "NEW"] == 0).all())
                if method == "ML mean-variance":
                    forecasts = result["forecasts"]
                    young = forecasts[(forecasts.symbol == "NEW") & (forecasts.date < eligible_date)]
                    seasoned = forecasts[(forecasts.symbol == "NEW") & (forecasts.date >= eligible_date)]
                    self.assertTrue(young.annualized_volatility.isna().all())
                    self.assertTrue(young.expected_return.isna().all())
                    self.assertTrue(seasoned.annualized_volatility.notna().all())

    def test_benchmark_buys_only_names_listed_by_the_window_start(self):
        # Buy-and-hold has no risk model to protect, so it needs no seasoning
        # - but it cannot buy a name that does not exist yet, and holding
        # NEW's flat placeholder would only dilute it with a fabricated 0%.
        prices = listing_fixture(rows=700, listed_at=300)  # lists AFTER the evaluation start (row 200)
        result = run_walk_forward_backtest(
            prices, AllocationConfig(method="Equal weight", max_weight=1.0, cash_weight=0.0),
            BacktestConfig(train_days=200, rebalance_days=21), cache_dir=None)
        benchmark = result["benchmark"]
        self.assertEqual(prices.index.get_loc(benchmark.index[0]), 200)
        window = prices.loc[benchmark.index, ["A", "B", "C"]]
        expected = (window / window.iloc[0]).mean(axis=1)
        pd.testing.assert_series_equal(benchmark, expected.rename("benchmark"), check_freq=False)
        # Listed before the start: an ordinary constituent of the benchmark.
        earlier = listing_fixture(rows=700, listed_at=150)
        result = run_walk_forward_backtest(
            earlier, AllocationConfig(method="Equal weight", max_weight=1.0, cash_weight=0.0),
            BacktestConfig(train_days=200, rebalance_days=21), cache_dir=None)
        benchmark = result["benchmark"]
        window = earlier.loc[benchmark.index]
        pd.testing.assert_series_equal(benchmark, (window / window.iloc[0]).mean(axis=1).rename("benchmark"),
                                       check_freq=False)

    def test_decision_names_ineligible_symbols_and_goes_cash_only_when_none_qualify(self):
        from ml_portfolio.decisions import MIN_LISTING_HISTORY_DAYS
        prices = listing_fixture(rows=700, listed_at=150)
        context = build_context(prices, 21, ml=False, cache_dir=None)
        decision = make_decision(context, 300, AllocationConfig(method="Equal weight", max_weight=1.0), 200)
        self.assertEqual(decision["ineligible"], ["NEW"])
        self.assertEqual(decision["weights"]["NEW"], 0.0)
        self.assertNotIn("NEW", decision["covariance"].index)
        self.assertTrue(decision["checks"].passed.all())
        decision = make_decision(context, 150 + MIN_LISTING_HISTORY_DAYS,
                                 AllocationConfig(method="Equal weight", max_weight=1.0), 200)
        self.assertEqual(decision["ineligible"], [])
        self.assertGreater(decision["weights"]["NEW"], 0)
        # Every name still warming up: cash only, no solver, no crash.
        young = prices.iloc[:, [3]].copy()
        young.attrs["eligible_from"] = prices.attrs["eligible_from"]
        context = build_context(young, 21, ml=False, cache_dir=None)
        decision = make_decision(context, 300, AllocationConfig(method="Equal weight"), 200)
        self.assertEqual(decision["ineligible"], ["NEW"])
        self.assertEqual(decision["weights"]["CASH"], 1.0)
        self.assertEqual(decision["solver"]["mode"], "cash-only")
        self.assertTrue(decision["checks"].passed.all())

    def test_features_and_targets_never_see_the_pre_listing_placeholder(self):
        # The ML path must not learn from fabricated rows: no feature row and
        # no target exists for NEW before it listed, and the same price VALUES
        # without the eligibility attrs are cached under a different key.
        from ml_portfolio.features import build_feature_panel, forward_returns
        prices = listing_fixture(rows=500, listed_at=150)
        panel = build_feature_panel(prices)
        new_rows = panel.xs("NEW", level="symbol")
        self.assertGreaterEqual(new_rows.index.min(), prices.index[150 + 21])
        self.assertLess(new_rows.index.min(), prices.index[150 + 130])
        targets = forward_returns(prices, 21).xs("NEW", level="symbol")
        self.assertTrue(targets.loc[targets.index < prices.index[150]].isna().all())
        self.assertTrue(targets.loc[prices.index[150]:prices.index[300]].notna().all())
        plain = prices.copy()
        plain.attrs = {}
        with tempfile.TemporaryDirectory() as folder:
            with_attrs = build_context(prices, 21, cache_dir=Path(folder))
            without = build_context(plain, 21, cache_dir=Path(folder))
        self.assertGreater(len(without.features.xs("NEW", level="symbol")),
                           len(with_attrs.features.xs("NEW", level="symbol")))

    def test_live_holding_in_unseasoned_listing_is_proposed_for_sale_with_a_reason(self):
        prices = listing_fixture(rows=400, listed_at=150)  # NEW has 250 sessions: two short
        quantities = pd.Series({"A": 100., "B": 100., "C": 100., "NEW": 500.})
        proposal = build_proposal(prices, quantities, 1000., AllocationConfig(method="Equal weight", max_weight=1.0), 200, 10)
        trades = proposal["trades"].set_index("symbol")
        self.assertEqual(trades.loc["NEW", "action"], "Sell")
        self.assertEqual(trades.loc["NEW", "target_quantity"], 0.0)
        self.assertIn("Listed too recently", trades.loc["NEW", "reason"])
        self.assertIn("NEW", proposal["decision"]["ineligible"])
        # The forced exit is real turnover in the risk checks (not hidden by
        # the solver treating that money as cash already).
        checks = proposal["checks"].set_index("check")
        self.assertGreater(checks.loc["One-way turnover", "value"], trades.loc["NEW", "current_weight"] * 0.99)

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
            self.assertEqual(payload["schema_version"], 2)
            history = saved_proposals(Path(temp), prices.iloc[-1], prices.index[-1])
            self.assertAlmostEqual(history.iloc[0].marked_value, proposal["capital_after"])
            changed = prices * 2
            history = saved_proposals(Path(temp), changed.iloc[-1], changed.index[-1], changed)
            self.assertEqual(history.iloc[0].status, "Not comparable")

    def test_large_universe_proposal_saves_and_loads(self):
        # Regression test: above FULL_COVARIANCE_ASSET_LIMIT, models.py's
        # factor risk model stores raw ndarrays (`loadings`/`factor_variance`/
        # `specific_variance`) in the covariance DataFrame's `.attrs` - pandas'
        # own `to_parquet` embeds a non-empty `.attrs` as JSON internally with
        # no type guard, so saving a large-universe proposal used to crash
        # with `TypeError: Object of type ndarray is not JSON serializable`
        # even though the (accepted) proposal itself computed correctly.
        prices = large_price_fixture(rows=260, columns=450)
        qty = pd.Series(0.0, index=prices.columns)
        config = AllocationConfig(method="Minimum variance", max_weight=0.05)
        proposal = build_proposal(prices, qty, 100000, config, 200, 10)
        self.assertTrue(proposal["accepted"])
        with tempfile.TemporaryDirectory() as temp:
            path = save_proposal(Path(temp), proposal, prices, config, 200, 10)
            payload = json.loads(path.read_text())
            self.assertEqual(payload["schema_version"], 2)
            history = saved_proposals(Path(temp), prices.iloc[-1], prices.index[-1])
            self.assertAlmostEqual(history.iloc[0].marked_value, proposal["capital_after"])

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
