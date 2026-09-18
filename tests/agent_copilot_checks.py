from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.append(str(ROOT / ".app-deps"))

import numpy as np
import pandas as pd

from ml_portfolio.holdings import save_purchase_portfolio
from ml_portfolio.ledger import append_transactions, save_transaction_ledger
from ml_portfolio.research_agent import (
    build_price_research_report,
    load_research_report,
    research_reports,
    save_research_report,
)
from ml_portfolio.storage import price_key
from ml_portfolio.strategy_copilot import (
    MAX_SWEEP_COMBINATIONS,
    StrategyParseError,
    compile_strategy_prompt,
    load_strategy_run,
    load_strategy_sweep,
    run_strategy_dsl_backtest,
    save_strategy_run,
    save_strategy_sweep,
    strategy_from_controls,
    strategy_runs,
    strategy_sweeps,
    sweep_strategy,
    validate_strategy_dsl,
)


def price_fixture(rows=320):
    dates = pd.bdate_range("2020-01-01", periods=rows)
    base = np.linspace(100, 180, rows)
    prices = pd.DataFrame({
        "AAA": base,
        "BBB": np.linspace(100, 90, rows),
        "CCC": 100 + 8 * np.sin(np.linspace(0, 10, rows)),
    }, index=dates)
    return prices


class ResearchAgentChecks(unittest.TestCase):
    def test_price_research_report_is_structured_and_peer_aware(self):
        report = build_price_research_report(price_fixture(), "AAA", ["BBB"], "Compare momentum.")
        self.assertEqual(report["symbol"], "AAA")
        self.assertIn("AAA", report["thesis"])
        self.assertIn("Question focus", report["thesis"])
        self.assertEqual(list(report["scores"].dimension)[0], "Overall")
        self.assertTrue(report["scores"].score.between(0, 100).all())
        self.assertEqual(set(report["peers"].symbol), {"AAA", "BBB"})

    def test_short_history_reports_unavailable_not_a_mislabeled_return(self):
        # Regression test: with under a year of history, "1Y return" used to
        # silently show the full-history return under that label instead of
        # reporting it as unavailable - a materially different, differently
        # risky number presented as if it were a genuine one-year return.
        report = build_price_research_report(price_fixture(rows=100), "AAA", ["BBB"])
        one_year_row = report["summary"].set_index("metric").loc["1Y return", "value"]
        self.assertTrue(pd.isna(one_year_row))
        # A window that DOES fit should still report a real number.
        one_month_row = report["summary"].set_index("metric").loc["1M return", "value"]
        self.assertTrue(np.isfinite(one_month_row))

    def test_formulas_and_scenario_range_present_and_consistent(self):
        report = build_price_research_report(price_fixture(), "AAA", ["BBB"])
        self.assertIn("overall_weights", report["formulas"])
        self.assertAlmostEqual(sum(report["formulas"]["overall_weights"].values()), 1.0)
        scenario = report["scenario_range"]
        if "bear_10th_percentile" in scenario:
            self.assertLessEqual(scenario["bear_10th_percentile"], scenario["base_median"])
            self.assertLessEqual(scenario["base_median"], scenario["bull_90th_percentile"])

    def test_save_load_roundtrip_and_isolation_from_other_kinds(self):
        prices = price_fixture()
        report = build_price_research_report(prices, "AAA", ["BBB"])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = save_research_report(root, "AAA brief", prices, report)
            payload = load_research_report(path)
            self.assertEqual(payload["kind"], "research_report")
            self.assertEqual(payload["name"], "AAA brief")
            self.assertEqual(payload["report"]["symbol"], "AAA")
            self.assertIn("AAA brief", research_reports(root)[path])

            strategy = strategy_from_controls("momentum_top", momentum_lookback=63, max_positions=1,
                                              rebalance_days=21, transaction_cost_bps=0)
            result = run_strategy_dsl_backtest(prices, strategy, evaluation_start=prices.index[126])
            save_strategy_run(root, "Linked run", prices, result, source_research_report_id=payload["id"])
            # research_reports() must not pick up the strategy_run file and vice versa.
            self.assertEqual(len(research_reports(root)), 1)
            self.assertEqual(len(strategy_runs(root)), 1)

    def test_source_research_report_id_round_trips(self):
        prices = price_fixture()
        report = build_price_research_report(prices, "AAA", ["BBB"])
        strategy = strategy_from_controls("momentum_top", momentum_lookback=63, max_positions=1,
                                          rebalance_days=21, transaction_cost_bps=0)
        result = run_strategy_dsl_backtest(prices, strategy, evaluation_start=prices.index[126])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            report_path = save_research_report(root, "AAA brief", prices, report)
            report_id = load_research_report(report_path)["id"]

            unlinked_path = save_strategy_run(root, "Unlinked", prices, result)
            self.assertIsNone(load_strategy_run(unlinked_path)["source_research_report_id"])

            linked_path = save_strategy_run(root, "Linked", prices, result,
                                            source_research_report_id=report_id)
            linked_payload = load_strategy_run(linked_path)
            self.assertEqual(linked_payload["source_research_report_id"], report_id)
            resolved = load_research_report(report_path)
            self.assertEqual(resolved["id"], linked_payload["source_research_report_id"])

    def test_research_rejects_unknown_peer(self):
        with self.assertRaises(ValueError):
            build_price_research_report(price_fixture(), "AAA", ["MISSING"])


class StrategyCopilotChecks(unittest.TestCase):
    def test_prompt_compiles_to_validated_json(self):
        strategy = compile_strategy_prompt("Buy when RSI is below 35, hold 5 stocks.", max_positions=5)
        self.assertEqual(strategy["entry"]["template"], "rsi_oversold")
        self.assertEqual(strategy["entry"]["threshold"], 35)
        validated = validate_strategy_dsl(strategy, price_fixture())
        self.assertEqual(validated["portfolio"]["max_positions"], 3)

    def test_rsi_period_in_parens_does_not_break_threshold_parsing(self):
        # Regression test: "(14)" between "rsi" and "below" used to break the
        # threshold regex's no-digit gap, silently falling back to the
        # hardcoded default threshold of 30 instead of reading the 20 the
        # user actually wrote.
        strategy = compile_strategy_prompt("RSI(14) below 20, hold 5 stocks.", max_positions=5)
        self.assertEqual(strategy["entry"]["template"], "rsi_oversold")
        self.assertEqual(strategy["entry"]["period"], 14)
        self.assertEqual(strategy["entry"]["threshold"], 20)

    def test_bare_sma_number_is_ambiguous_not_defaulted(self):
        # Regression test: "SMA 20" used to silently compile to the 50/200
        # default, discarding the "20" the user gave, instead of raising.
        with self.assertRaises(StrategyParseError):
            compile_strategy_prompt("SMA 20, hold 5 stocks.", max_positions=5)

    def test_unsupported_concepts_are_rejected_not_dropped(self):
        for prompt in (
            "momentum strategy, short the bottom decile, hold 5 stocks",
            "momentum strategy with 2x leverage, hold 5 stocks",
            "momentum strategy with a 5% stop loss, hold 5 stocks",
        ):
            with self.assertRaises(StrategyParseError):
                compile_strategy_prompt(prompt, max_positions=5)
        # A supported prompt with none of these words still compiles fine.
        compile_strategy_prompt("momentum strategy, hold 5 stocks.", max_positions=5)

    def test_schema_version_frequency_and_execution_are_validated(self):
        strategy = strategy_from_controls("momentum_top", momentum_lookback=63, max_positions=1)
        prices = price_fixture()
        validate_strategy_dsl(strategy, prices)  # sanity: the real thing still validates

        bad_version = dict(strategy, schema_version=999)
        with self.assertRaises(StrategyParseError):
            validate_strategy_dsl(bad_version, prices)

        bad_frequency = dict(strategy, frequency="weekly")
        with self.assertRaises(StrategyParseError):
            validate_strategy_dsl(bad_frequency, prices)

        bad_execution = dict(strategy, execution="something else entirely")
        with self.assertRaises(StrategyParseError):
            validate_strategy_dsl(bad_execution, prices)

    def test_positive_only_string_is_rejected_not_silently_truthy(self):
        # Regression test: entry.get("positive_only", True) had no type check,
        # so the JSON/CSV string "false" (truthy in Python) silently behaved
        # as enabled instead of disabling the positive-only filter.
        strategy = strategy_from_controls("momentum_top", momentum_lookback=63, max_positions=1)
        strategy["entry"]["positive_only"] = "false"
        with self.assertRaises(StrategyParseError):
            validate_strategy_dsl(strategy, price_fixture())

    def test_allocation_controls_are_applied_and_accurately_reported(self):
        # Regression test: Copilot never received max_weight/cash_weight at
        # all and always went 100% into picks with zero cash, while
        # enforced_constraints unconditionally claimed "cash_buffer": True.
        prices = price_fixture()
        strategy = strategy_from_controls("momentum_top", momentum_lookback=10, max_positions=1,
                                          rebalance_days=10, transaction_cost_bps=0)
        unconstrained = run_strategy_dsl_backtest(prices, strategy, evaluation_start=prices.index[30])
        first_unconstrained = unconstrained["weights"].iloc[0]
        self.assertAlmostEqual(first_unconstrained.drop("CASH").max(), 1.0)
        self.assertAlmostEqual(first_unconstrained.CASH, 0.0)
        self.assertFalse(unconstrained["enforced_constraints"]["cash_buffer"])

        constrained = run_strategy_dsl_backtest(prices, strategy, evaluation_start=prices.index[30],
                                                max_weight=0.6, cash_weight=0.1)
        first_constrained = constrained["weights"].iloc[0]
        self.assertLessEqual(first_constrained.drop("CASH").max(), 0.6 + 1e-9)
        self.assertGreaterEqual(first_constrained.CASH, 0.1 - 1e-9)
        self.assertAlmostEqual(first_constrained.sum(), 1.0)
        self.assertTrue(constrained["enforced_constraints"]["cash_buffer"])
        self.assertTrue(constrained["enforced_constraints"]["max_weight_cap"])

    def test_universe_eligibility_cutoff_excludes_symbol_after_but_not_before(self):
        # Mirrors backtest.py's equivalent test: a symbol carrying
        # `.attrs["eligible_through"]` must still be selectable before its
        # cutoff and never selected at/after it.
        dates = pd.bdate_range("2020-01-01", periods=200)
        prices = pd.DataFrame({
            "A": np.linspace(100, 90, 200),
            "B": np.linspace(100, 80, 200),
            "C": np.linspace(100, 200, 200),  # by far the strongest momentum - would always be picked
        }, index=dates)
        cutoff = dates[120]
        prices.attrs["eligible_through"] = {"C": cutoff}
        strategy = strategy_from_controls("momentum_top", momentum_lookback=10, max_positions=1,
                                          rebalance_days=10, transaction_cost_bps=0)
        result = run_strategy_dsl_backtest(prices, strategy, evaluation_start=dates[30])
        rebalances = result["rebalances"].set_index("date")
        before = rebalances.loc[rebalances.index <= cutoff]
        after = rebalances.loc[rebalances.index > cutoff]
        self.assertGreater(len(before), 0)
        self.assertGreater(len(after), 0)
        self.assertTrue((before["selected"] == "C").any())
        self.assertFalse((after["selected"] == "C").any())

    def test_held_position_is_liquidated_the_day_after_its_cutoff(self):
        # Mirrors backtest.py's forced-liquidation test: with C the strongest
        # momentum name and a cutoff mid-interval (rebalances at 30 + 10k,
        # cutoff at 125), the position opened on the 120 rebalance must be
        # sold to CASH on day 126 with the fee charged there, not carried
        # flat until the 130 rebalance.
        dates = pd.bdate_range("2020-01-01", periods=200)
        prices = pd.DataFrame({
            "A": np.linspace(100, 90, 200),
            "B": np.linspace(100, 80, 200),
            "C": np.linspace(100, 200, 200),
        }, index=dates)
        cutoff = dates[125]
        prices.attrs["eligible_through"] = {"C": cutoff.isoformat()}  # as clean_gappy_columns stores it
        prices.loc[prices.index > cutoff, "C"] = prices.loc[cutoff, "C"]
        cost_bps = 50
        strategy = strategy_from_controls("momentum_top", momentum_lookback=10, max_positions=1,
                                          rebalance_days=10, transaction_cost_bps=cost_bps)
        result = run_strategy_dsl_backtest(prices, strategy, evaluation_start=dates[30])
        rebalances = result["rebalances"].set_index("date")
        opening = rebalances.index[rebalances.index <= cutoff][-1]
        self.assertEqual(rebalances.loc[opening, "selected"], "C")
        self.assertEqual(rebalances.loc[opening, "liquidated"], "C")
        self.assertTrue(rebalances.loc[rebalances.index != opening, "liquidated"].eq("").all())
        curve = result["equity_curve"]
        pre, post = curve.loc[cutoff], curve.loc[dates[126]]
        # Fully invested in C (cash_weight=0 by default), so the fee is on the
        # whole book: net drops by exactly cost_bps while gross is flat.
        self.assertAlmostEqual(post.gross_equity / pre.gross_equity, 1.0, places=10)
        self.assertAlmostEqual(post.equity / pre.equity, 1 - cost_bps / 10000, places=10)
        self.assertGreater(rebalances.loc[opening, "cost_value"], pre.equity * cost_bps / 10000 * 0.999)
        # The following rebalance rotates into A/B from an all-CASH book, so
        # its turnover is a normal buy, and C is never selected again.
        following = rebalances.index[rebalances.index > cutoff][0]
        self.assertNotIn("C", rebalances.loc[following, "selected"])
        self.assertAlmostEqual(result["metrics"]["Cost drag"],
                               curve.gross_equity.iloc[-1] - curve.equity.iloc[-1], places=10)

    def test_newly_listed_symbol_not_selected_until_listing_plus_warmup(self):
        # NEW's real history is the strongest momentum in the frame and its
        # placeholder is flat; it must never be chosen before listing plus
        # the indicator warm-up, and must be chosen afterwards.
        dates = pd.bdate_range("2020-01-01", periods=300)
        prices = pd.DataFrame({
            "A": np.linspace(100, 110, 300),
            "B": np.linspace(100, 95, 300),
            "NEW": np.r_[np.full(120, 40.0), np.linspace(40, 120, 180)],
        }, index=dates)
        prices.attrs["eligible_from"] = {"NEW": dates[120].isoformat()}
        strategy = strategy_from_controls("momentum_top", momentum_lookback=20, max_positions=1,
                                          rebalance_days=5, transaction_cost_bps=0)
        result = run_strategy_dsl_backtest(prices, strategy, evaluation_start=dates[40])
        rebalances = result["rebalances"].set_index("date")
        warmup = int(validate_strategy_dsl(strategy, prices)["_warmup_days"])
        earliest_allowed = dates[120 + warmup + 1]
        picked = rebalances[rebalances.selected == "NEW"]
        self.assertGreater(len(picked), 5)
        self.assertGreaterEqual(picked.index.min(), earliest_allowed)
        self.assertTrue((rebalances.loc[rebalances.index < earliest_allowed, "selected"] != "NEW").all())

    def test_research_agent_measures_only_real_history_for_a_recent_listing(self):
        dates = pd.bdate_range("2020-01-01", periods=400)
        prices = pd.DataFrame({
            "A": np.linspace(100, 110, 400),
            "B": np.linspace(100, 95, 400),
            "NEW": np.r_[np.full(300, 40.0), np.linspace(40, 60, 100)],  # listed 100 sessions ago
        }, index=dates)
        prices.attrs["eligible_from"] = {"NEW": dates[300].isoformat()}
        report = build_price_research_report(prices, "NEW", peers=["A"])
        row = report["peers"].set_index("symbol").loc["NEW"]
        self.assertTrue(np.isnan(row["return_1y"]))     # 100 real sessions: no 1Y return, not a flat-line 0%
        self.assertTrue(np.isnan(row["return_6m"]))
        self.assertAlmostEqual(row["return_3m"], 60 / prices["NEW"].iloc[-64] - 1, places=10)
        self.assertGreater(row["annualized_volatility"], 0)
        self.assertEqual(report["peers"].set_index("symbol").loc["A", "return_1y"],
                         prices["A"].iloc[-1] / prices["A"].iloc[-253] - 1)

    def test_strategy_backtest_runs_and_uses_shifted_information(self):
        prices = price_fixture()
        strategy = strategy_from_controls("momentum_top", momentum_lookback=63,
                                          max_positions=1, rebalance_days=21,
                                          transaction_cost_bps=0)
        result = run_strategy_dsl_backtest(prices, strategy, evaluation_start=prices.index[126])
        self.assertIn("CAGR", result["metrics"])
        self.assertFalse(result["rebalances"].empty)
        first = result["rebalances"].iloc[0]
        self.assertLess(first.information_through, first.date)
        self.assertTrue(np.isfinite(result["equity_curve"].equity).all())

    def test_invalid_dsl_is_rejected(self):
        strategy = strategy_from_controls("sma_cross", fast_window=200, slow_window=50)
        with self.assertRaises(ValueError):
            run_strategy_dsl_backtest(price_fixture(), strategy)

    def test_future_data_does_not_change_first_decision(self):
        prices = price_fixture()
        strategy = strategy_from_controls("momentum_top", momentum_lookback=63,
                                          max_positions=1, rebalance_days=21,
                                          transaction_cost_bps=0)
        baseline = run_strategy_dsl_backtest(prices, strategy, evaluation_start=prices.index[126])
        first_information_through = baseline["rebalances"].iloc[0].information_through
        cutoff = prices.index.get_loc(first_information_through)
        changed = prices.copy()
        changed.iloc[cutoff + 1:, 0] *= 1.5
        changed_result = run_strategy_dsl_backtest(changed, strategy, evaluation_start=prices.index[126])
        pd.testing.assert_series_equal(
            baseline["weights"].iloc[0].sort_index(), changed_result["weights"].iloc[0].sort_index())

    def test_fees_paid_matches_independently_reconstructed_cost(self):
        # Regression test for a real bug: the rebalance loop used to omit
        # `cost_value`, so calculate_metrics silently reported "Fees paid" as
        # 0.0 regardless of the configured cost.
        dates = pd.bdate_range("2020-01-01", periods=100)
        prices = pd.DataFrame({"A": np.linspace(100, 130, 100), "B": np.linspace(100, 80, 100)},
                              index=dates)
        strategy = strategy_from_controls("momentum_top", momentum_lookback=10, max_positions=1,
                                          rebalance_days=10, transaction_cost_bps=50)
        result = run_strategy_dsl_backtest(prices, strategy, evaluation_start=dates[30])
        self.assertGreater(result["metrics"]["Fees paid"], 0.0)
        expected_total = sum(
            result["equity_curve"].loc[row.date, "equity"] * row.cost
            for _, row in result["rebalances"].iterrows()
        )
        self.assertAlmostEqual(result["metrics"]["Fees paid"], expected_total, places=6)

    def test_constant_prices_run_without_error_and_rsi_uses_flat_convention(self):
        dates = pd.bdate_range("2020-01-01", periods=100)
        prices = pd.DataFrame({"A": 100.0, "B": 100.0, "C": 100.0}, index=dates)
        for kwargs in (
            dict(template="momentum_top", momentum_lookback=10),
            dict(template="sma_cross", fast_window=5, slow_window=10),
            dict(template="rsi_oversold", rsi_period=5, rsi_threshold=50),
        ):
            strategy = strategy_from_controls(max_positions=1, rebalance_days=10,
                                              transaction_cost_bps=10, **kwargs)
            result = run_strategy_dsl_backtest(prices, strategy, evaluation_start=dates[30])
            self.assertTrue(np.isfinite(result["equity_curve"].equity).all())

        # RSI=50 (the flat convention) never satisfies "< 50", so no asset is
        # ever selected and equity should stay exactly flat.
        rsi_strategy = strategy_from_controls("rsi_oversold", rsi_period=5, rsi_threshold=50,
                                              max_positions=1, rebalance_days=10,
                                              transaction_cost_bps=10)
        rsi_result = run_strategy_dsl_backtest(prices, rsi_strategy, evaluation_start=dates[30])
        self.assertTrue((rsi_result["equity_curve"].equity == 1.0).all())
        self.assertTrue((rsi_result["rebalances"].positions == 0).all())

    def test_no_eligible_assets_stays_in_cash(self):
        dates = pd.bdate_range("2020-01-01", periods=100)
        prices = pd.DataFrame({"A": np.linspace(100, 50, 100), "B": np.linspace(100, 60, 100)},
                              index=dates)
        strategy = strategy_from_controls("momentum_top", momentum_lookback=10, max_positions=1,
                                          rebalance_days=10, transaction_cost_bps=10)
        result = run_strategy_dsl_backtest(prices, strategy, evaluation_start=dates[30])
        self.assertTrue((result["weights"]["CASH"] == 1.0).all())
        self.assertTrue((result["rebalances"].positions == 0).all())
        self.assertTrue((result["equity_curve"].equity == 1.0).all())

    def test_invalid_evaluation_bounds_rejected(self):
        prices = price_fixture()
        strategy = strategy_from_controls("momentum_top", momentum_lookback=63,
                                          max_positions=1, rebalance_days=21,
                                          transaction_cost_bps=0)
        with self.assertRaises(ValueError):  # before indicator warmup is satisfied
            run_strategy_dsl_backtest(prices, strategy, evaluation_start=prices.index[10])
        with self.assertRaises(ValueError):  # fewer than min_evaluation_days remain
            run_strategy_dsl_backtest(prices, strategy, evaluation_start=prices.index[-5])
        with self.assertRaises(ValueError):  # entirely outside the price index
            run_strategy_dsl_backtest(prices, strategy, evaluation_start="2099-01-01")

    def test_save_load_roundtrip_and_listing(self):
        prices = price_fixture()
        strategy = strategy_from_controls("momentum_top", momentum_lookback=63,
                                          max_positions=1, rebalance_days=21,
                                          transaction_cost_bps=0)
        result = run_strategy_dsl_backtest(prices, strategy, evaluation_start=prices.index[126])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = save_strategy_run(root, "My momentum run", prices, result)
            payload = load_strategy_run(path)
            self.assertEqual(payload["kind"], "strategy_run")
            self.assertEqual(payload["name"], "My momentum run")
            self.assertAlmostEqual(payload["result"]["metrics"]["CAGR"], result["metrics"]["CAGR"])
            self.assertIn("My momentum run", strategy_runs(root)[path])

    def test_data_changed_notice_uses_price_key_mismatch(self):
        # Mirrors saved_proposals' comparability check: a saved run's stored
        # price_hash should differ once the underlying data has changed, which
        # is exactly the signal the UI uses to show a "data changed" notice.
        prices = price_fixture()
        strategy = strategy_from_controls("momentum_top", momentum_lookback=63,
                                          max_positions=1, rebalance_days=21,
                                          transaction_cost_bps=0)
        result = run_strategy_dsl_backtest(prices, strategy, evaluation_start=prices.index[126])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = save_strategy_run(root, "Run", prices, result)
            payload = load_strategy_run(path)
            self.assertEqual(payload["price_hash"], price_key(prices))
            changed_prices = prices * 2
            self.assertNotEqual(payload["price_hash"], price_key(changed_prices))

    def test_strategy_run_listing_does_not_collide_with_other_kinds(self):
        prices = price_fixture()
        strategy = strategy_from_controls("momentum_top", momentum_lookback=63,
                                          max_positions=1, rebalance_days=21,
                                          transaction_cost_bps=0)
        result = run_strategy_dsl_backtest(prices, strategy, evaluation_start=prices.index[126])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            save_strategy_run(root, "Strategy", prices, result)
            purchases = pd.DataFrame([{"symbol": "AAA", "instrument_type": "Stock",
                                       "purchase_date": prices.index[0], "quantity": 1,
                                       "purchase_price": 100}])
            save_purchase_portfolio(root, "Purchases", purchases, 0.0, prices.iloc[-1], prices.index[-1])
            transactions, _ = append_transactions(None, purchases.rename(
                columns={"purchase_date": "date", "purchase_price": "price"}).assign(action="buy"),
                prices.index[-1])
            save_transaction_ledger(root, "Ledger", transactions, prices.iloc[-1], prices.index[-1])

            self.assertEqual(len(strategy_runs(root)), 1)


class ParameterSweepChecks(unittest.TestCase):
    def _fixed(self):
        return {"max_positions": 1, "rebalance_days": 21, "transaction_cost_bps": 0}

    def test_bound_enforced(self):
        prices = price_fixture(400)
        grid = {"momentum_lookback": list(range(21, 21 + 30))}  # 30 candidates > MAX_SWEEP_COMBINATIONS
        self.assertGreater(len(grid["momentum_lookback"]), MAX_SWEEP_COMBINATIONS)
        with self.assertRaises(ValueError):
            sweep_strategy(prices, "momentum_top", self._fixed(), grid,
                           prices.index[250], prices.index[251])

    def test_selection_bias_is_exposed_not_hidden(self):
        prices = price_fixture(400)
        grid = {"momentum_lookback": [21, 42, 63]}
        sweep = sweep_strategy(prices, "momentum_top", self._fixed(), grid,
                               prices.index[250], prices.index[251], selection_metric="Sharpe")
        self.assertEqual(len(sweep["variants"]), 3)
        best_row = sweep["variants"].loc[sweep["variants"]["Sharpe"].idxmax()]
        self.assertEqual(int(best_row["momentum_lookback"]), sweep["selected_params"]["momentum_lookback"])

    def test_final_test_window_is_genuinely_separate_from_selection(self):
        prices = price_fixture(400)
        fixed = self._fixed()
        grid = {"momentum_lookback": [21, 42, 63]}
        selection_end = prices.index[250]
        final_test_start = prices.index[251]
        sweep = sweep_strategy(prices, "momentum_top", fixed, grid, selection_end, final_test_start)
        self.assertGreater(pd.Timestamp(sweep["final_test_result"]["assumptions"]["evaluation_start"]),
                           pd.Timestamp(sweep["selection_end"]))
        champion = strategy_from_controls("momentum_top", **fixed, **sweep["selected_params"])
        selection_only = run_strategy_dsl_backtest(prices, champion, evaluation_end=selection_end)
        self.assertNotEqual(selection_only["metrics"]["CAGR"], sweep["final_test_result"]["metrics"]["CAGR"])

    def test_final_test_must_start_after_selection_ends(self):
        prices = price_fixture(400)
        grid = {"momentum_lookback": [21, 42]}
        with self.assertRaises(ValueError):
            sweep_strategy(prices, "momentum_top", self._fixed(), grid,
                           prices.index[250], prices.index[200])

    def test_save_load_roundtrip_and_isolation(self):
        prices = price_fixture(400)
        grid = {"momentum_lookback": [21, 42, 63]}
        sweep = sweep_strategy(prices, "momentum_top", self._fixed(), grid,
                               prices.index[250], prices.index[251])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = save_strategy_sweep(root, "My sweep", prices, sweep)
            payload = load_strategy_sweep(path)
            self.assertEqual(payload["kind"], "strategy_sweep")
            self.assertEqual(len(payload["sweep"]["variants"]), 3)
            self.assertEqual(payload["sweep"]["selected_params"], sweep["selected_params"])
            self.assertIn("My sweep", strategy_sweeps(root)[path])
            self.assertEqual(len(strategy_runs(root)), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
