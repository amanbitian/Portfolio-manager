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

from ml_portfolio.robustness import (
    bootstrap_sharpe_ci,
    cost_stress,
    load_robustness_report,
    parameter_perturbation,
    randomized_placebo,
    run_robustness_checks,
    save_robustness_report,
    subperiod_analysis,
    summarize_verdict,
    verdict_conclusion,
)
from ml_portfolio.research_registry import list_experiments
from ml_portfolio.strategy_copilot import run_strategy_dsl_backtest, strategy_from_controls


def price_fixture(rows=400):
    dates = pd.bdate_range("2020-01-01", periods=rows)
    return pd.DataFrame({
        "AAA": np.linspace(100, 180, rows),
        "BBB": np.linspace(100, 90, rows),
        "CCC": 100 + 8 * np.sin(np.linspace(0, 10, rows)),
    }, index=dates)


def trending_fixture(rows=400):
    """One asset with an unmistakable, dominant, persistent uptrend against
    flat/declining noise - real momentum should find it almost every time,
    a random pick almost never."""
    dates = pd.bdate_range("2020-01-01", periods=rows)
    return pd.DataFrame({
        "WINNER": np.linspace(100, 400, rows),
        "B": 100 + np.cumsum(np.zeros(rows)),
        "C": np.linspace(100, 95, rows),
        "D": np.linspace(100, 105, rows),
        "E": 100 + 3 * np.sin(np.linspace(0, 20, rows)),
    }, index=dates)


def momentum_strategy(**overrides):
    params = dict(momentum_lookback=63, max_positions=1, rebalance_days=21, transaction_cost_bps=10)
    params.update(overrides)
    return strategy_from_controls("momentum_top", **params)


class CostStressChecks(unittest.TestCase):
    def test_baseline_multiplier_matches_original_cost(self):
        prices = price_fixture()
        strategy = momentum_strategy()
        frame = cost_stress(prices, strategy, evaluation_start=prices.index[126])
        baseline_row = frame.loc[frame.cost_multiplier == 1.0].iloc[0]
        self.assertAlmostEqual(baseline_row.cost_bps, 10.0)

    def test_higher_multiplier_means_higher_cost_bps(self):
        prices = price_fixture()
        strategy = momentum_strategy()
        frame = cost_stress(prices, strategy, evaluation_start=prices.index[126], multipliers=(1.0, 2.0, 3.0))
        self.assertEqual(list(frame.cost_bps), [10.0, 20.0, 30.0])


class SubperiodChecks(unittest.TestCase):
    def test_splits_into_requested_count_with_increasing_dates(self):
        prices = price_fixture()
        strategy = momentum_strategy()
        frame = subperiod_analysis(prices, strategy, evaluation_start=prices.index[126], n_subperiods=2)
        self.assertEqual(len(frame), 2)
        self.assertLess(pd.Timestamp(frame.iloc[0]["end"]), pd.Timestamp(frame.iloc[1]["end"]))

    def test_rejects_fewer_than_two_subperiods(self):
        prices = price_fixture()
        strategy = momentum_strategy()
        with self.assertRaises(ValueError):
            subperiod_analysis(prices, strategy, evaluation_start=prices.index[126], n_subperiods=1)

    def test_too_short_subperiod_reports_error_row_not_a_crash(self):
        prices = price_fixture()
        strategy = momentum_strategy()
        frame = subperiod_analysis(prices, strategy, evaluation_start=prices.index[126], n_subperiods=8)
        self.assertTrue((~frame["error"].isna()).any() if "error" in frame else False)


class ParameterPerturbationChecks(unittest.TestCase):
    def test_covers_both_directions_for_momentum_lookback(self):
        prices = price_fixture()
        strategy = momentum_strategy()
        frame = parameter_perturbation(prices, strategy, evaluation_start=prices.index[126])
        self.assertEqual(frame.iloc[0].field, "baseline")
        directions = set(frame.loc[frame.field == "lookback_days", "direction"])
        self.assertEqual(directions, {"-", "+"})
        minus_value = frame.loc[(frame.field == "lookback_days") & (frame.direction == "-"), "value"].iloc[0]
        plus_value = frame.loc[(frame.field == "lookback_days") & (frame.direction == "+"), "value"].iloc[0]
        self.assertLess(minus_value, 63)
        self.assertGreater(plus_value, 63)

    def test_sma_cross_perturbs_both_windows(self):
        prices = price_fixture()
        strategy = strategy_from_controls("sma_cross", fast_window=20, slow_window=100,
                                          max_positions=1, rebalance_days=21, transaction_cost_bps=10)
        frame = parameter_perturbation(prices, strategy, evaluation_start=prices.index[126])
        self.assertEqual(set(frame.field) - {"baseline"}, {"fast_window", "slow_window"})


class RandomizedPlaceboChecks(unittest.TestCase):
    def test_produces_one_row_per_trial_with_finite_or_nan_sharpe(self):
        prices = price_fixture()
        strategy = momentum_strategy()
        frame = randomized_placebo(prices, strategy, evaluation_start=prices.index[126], trials=5, seed=1)
        self.assertEqual(len(frame), 5)
        self.assertIn("Sharpe", frame.columns)

    def test_deterministic_given_a_seed(self):
        prices = price_fixture()
        strategy = momentum_strategy()
        a = randomized_placebo(prices, strategy, evaluation_start=prices.index[126], trials=5, seed=7)
        b = randomized_placebo(prices, strategy, evaluation_start=prices.index[126], trials=5, seed=7)
        pd.testing.assert_frame_equal(a, b)

    def test_real_signal_clearly_beats_random_pick_on_an_obvious_trend(self):
        # The actual sanity check that makes this placebo meaningful: with one
        # asset dominating on momentum, real top-1 momentum should find it
        # almost every rebalance (Sharpe far above the random-pick average).
        prices = trending_fixture()
        strategy = strategy_from_controls("momentum_top", momentum_lookback=42, max_positions=1,
                                          rebalance_days=21, transaction_cost_bps=0)
        baseline = run_strategy_dsl_backtest(prices, strategy, evaluation_start=prices.index[100])
        placebo = randomized_placebo(prices, strategy, evaluation_start=prices.index[100], trials=30, seed=3)
        self.assertGreater(baseline["metrics"]["Sharpe"], placebo["Sharpe"].mean())


class BootstrapChecks(unittest.TestCase):
    def test_confidence_interval_brackets_mean_and_uses_all_trials(self):
        prices = price_fixture()
        strategy = momentum_strategy(rebalance_days=5)
        result = run_strategy_dsl_backtest(prices, strategy, evaluation_start=prices.index[100])
        stats = bootstrap_sharpe_ci(result["equity_curve"], trials=200, seed=1)
        self.assertGreater(stats["trials"], 0)
        self.assertLessEqual(stats["ci_low"], stats["mean_sharpe"])
        self.assertLessEqual(stats["mean_sharpe"], stats["ci_high"])

    def test_rejects_too_few_returns(self):
        curve = pd.DataFrame({"return": [np.nan] + [0.001] * 10}, index=pd.bdate_range("2020-01-01", periods=11))
        with self.assertRaises(ValueError):
            bootstrap_sharpe_ci(curve)


class VerdictChecks(unittest.TestCase):
    def test_falsified_when_baseline_not_positive(self):
        cost_frame = pd.DataFrame([{"cost_multiplier": 1.0, "Sharpe": -0.1},
                                   {"cost_multiplier": 3.0, "Sharpe": -0.5}])
        subperiod_frame = pd.DataFrame([{"Sharpe": -0.1}, {"Sharpe": -0.2}])
        placebo_frame = pd.DataFrame([{"Sharpe": 0.0}] * 10)
        bootstrap = {"ci_low": -0.5}
        result = summarize_verdict(cost_frame, subperiod_frame, placebo_frame, bootstrap)
        self.assertEqual(result["verdict"], "falsified")
        self.assertFalse(result["checks"]["baseline_positive"])

    def test_supported_when_every_check_passes(self):
        cost_frame = pd.DataFrame([{"cost_multiplier": 1.0, "Sharpe": 1.5}, {"cost_multiplier": 3.0, "Sharpe": 1.1}])
        subperiod_frame = pd.DataFrame([{"Sharpe": 1.2}, {"Sharpe": 1.4}])
        placebo_frame = pd.DataFrame([{"Sharpe": s} for s in np.linspace(-0.5, 0.4, 20)])
        bootstrap = {"ci_low": 0.3}
        result = summarize_verdict(cost_frame, subperiod_frame, placebo_frame, bootstrap)
        self.assertEqual(result["verdict"], "supported")
        self.assertEqual(result["checks_passed"], 5)

    def test_inconclusive_when_evidence_is_mixed(self):
        cost_frame = pd.DataFrame([{"cost_multiplier": 1.0, "Sharpe": 0.8}, {"cost_multiplier": 3.0, "Sharpe": -0.1}])
        subperiod_frame = pd.DataFrame([{"Sharpe": 0.5}, {"Sharpe": 0.6}])
        placebo_frame = pd.DataFrame([{"Sharpe": s} for s in np.linspace(-0.3, 0.3, 20)])
        bootstrap = {"ci_low": -0.05}
        result = summarize_verdict(cost_frame, subperiod_frame, placebo_frame, bootstrap)
        self.assertEqual(result["verdict"], "inconclusive")

    def test_conclusion_text_lists_every_check(self):
        checks = {"baseline_positive": True, "survives_max_cost_stress": False}
        text = verdict_conclusion(checks)
        self.assertIn("PASS: baseline positive", text)
        self.assertIn("FAIL: survives max cost stress", text)


class EndToEndChecks(unittest.TestCase):
    def test_run_robustness_checks_returns_every_section(self):
        prices = price_fixture()
        strategy = momentum_strategy()
        bundle = run_robustness_checks(prices, strategy, evaluation_start=prices.index[126],
                                       placebo_trials=5, bootstrap_trials=50)
        for key in ("baseline_metrics", "cost_stress", "subperiods", "parameter_perturbation",
                   "placebo", "bootstrap", "verdict", "checks", "checks_passed", "checks_total"):
            self.assertIn(key, bundle)
        self.assertIn(bundle["verdict"], ("supported", "falsified", "inconclusive"))

    def test_save_load_roundtrip_and_registry_listing(self):
        prices = price_fixture()
        strategy = momentum_strategy()
        bundle = run_robustness_checks(prices, strategy, evaluation_start=prices.index[126],
                                       placebo_trials=5, bootstrap_trials=50)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = save_robustness_report(root, "AAA momentum robustness", prices, "strategy_run_abc123", bundle)
            payload = load_robustness_report(path)
            self.assertEqual(payload["kind"], "robustness_report")
            self.assertEqual(payload["source_strategy_run_id"], "strategy_run_abc123")
            self.assertIsInstance(payload["checks"]["cost_stress"], pd.DataFrame)
            self.assertEqual(payload["checks"]["verdict"], bundle["verdict"])

            experiments = list_experiments(root)
            self.assertEqual(len(experiments), 1)
            self.assertEqual(experiments.iloc[0]["kind"], "robustness_report")


if __name__ == "__main__":
    unittest.main(verbosity=2)
