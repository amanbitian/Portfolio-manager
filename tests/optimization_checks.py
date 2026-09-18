from __future__ import annotations

import json
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.append(str(ROOT / ".app-deps"))

import numpy as np
import pandas as pd

from app_logic_checks import price_fixture, large_price_fixture
from ml_portfolio.backtest import BacktestConfig, run_strategy_comparison, run_walk_forward_backtest
from ml_portfolio.decisions import _predict, _training, build_context
from ml_portfolio.features import feature_matrix_at
from ml_portfolio.models import RidgeWindow, ridge_daily_statistics, ridge_forecast
from ml_portfolio.optimizers import METHODS, AllocationConfig, AllocationError, allocate_portfolio, risk_checks
from ml_portfolio.portfolio import _json_value, build_proposal, save_proposal, saved_proposals
from ml_portfolio.experiments import research_overview, save_research
from ml_portfolio.storage import cached_frame, load_artifact, price_key, prune_cache, read_table, write_artifact


class RidgeChecks(unittest.TestCase):
    def test_window_statistics_match_original_fits(self):
        prices = price_fixture()
        context = build_context(prices, 21, cache_dir=None)
        for origin, window in [(210, 166), (400, 252), (504, 504), (620, 252)]:
            with self.subTest(origin=origin, window=window):
                x, y = _training(context, origin, window)
                original = ridge_forecast(x, y, feature_matrix_at(context.features, prices.index[origin]), list(prices.columns))
                fast, baseline, days = _predict(context, origin, window)
                np.testing.assert_allclose(fast, original, atol=1e-11, rtol=1e-9)
                self.assertAlmostEqual(baseline, x.join(y).dropna()[y.name].mean(), places=12)
                self.assertEqual(days, x.join(y).dropna().index.get_level_values("date").nunique())

    def test_missing_and_constant_features(self):
        rng = np.random.default_rng(4)
        dates = pd.bdate_range("2020-01-01", periods=100)
        index = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["date", "symbol"])
        x = pd.DataFrame({"constant": 1., "random": rng.normal(size=200)}, index=index)
        y = pd.Series(rng.normal(size=200), index=index, name="target")
        x.iloc[40:48, 1] = np.nan
        y.iloc[60:70] = np.nan
        model = RidgeWindow(ridge_daily_statistics(x, y), x.columns)
        current = x.xs(dates[-1])
        predicted, _, _ = model.predict(dates[20], dates[80], current, ["A", "B"])
        mask = (index.get_level_values("date") >= dates[20]) & (index.get_level_values("date") <= dates[80])
        expected = ridge_forecast(x.loc[mask], y.loc[mask], current, ["A", "B"])
        np.testing.assert_allclose(predicted, expected, atol=1e-10)


class CacheChecks(unittest.TestCase):
    def test_reuse_and_horizon_invalidation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prices = price_fixture(300)
            first = build_context(prices, 21, cache_dir=root)
            with patch("ml_portfolio.decisions.build_feature_panel", side_effect=AssertionError("recomputed")), \
                 patch("ml_portfolio.decisions.ridge_daily_statistics", side_effect=AssertionError("refit")):
                second = build_context(prices.copy(), 21, cache_dir=root)
            pd.testing.assert_frame_equal(first.features, second.features)
            np.testing.assert_array_equal(first.ridge.prefix, second.ridge.prefix)
            other = build_context(prices, 5, cache_dir=root)
            self.assertFalse(first.targets.equals(other.targets))
            self.assertEqual(len(list((root / "features").glob("*.parquet"))), 1)
            changed = prices.copy()
            changed.iloc[-1, 0] *= 2
            self.assertNotEqual(price_key(prices), price_key(changed))
            self.assertNotEqual(price_key(prices), price_key(prices.rename(columns={"A": "NEW"})))
            shifted = prices.copy()
            shifted.index += pd.Timedelta(days=1)
            self.assertNotEqual(price_key(prices), price_key(shifted))

    def test_corruption_and_write_failure_recompute(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            frame = pd.DataFrame({"value": [1., 2.]})
            cached_frame(root, "test", "key", lambda: frame)
            (root / "test" / "key.parquet").write_bytes(b"invalid parquet")
            pd.testing.assert_frame_equal(cached_frame(root, "test", "key", lambda: frame), frame)
            with patch("ml_portfolio.storage.atomic_parquet", side_effect=PermissionError("readonly")):
                with self.assertWarns(RuntimeWarning):
                    pd.testing.assert_frame_equal(cached_frame(root, "test", "new", lambda: frame), frame)

    def test_concurrent_writers_and_budget(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            frame = pd.DataFrame({"value": np.arange(1000)})
            with ThreadPoolExecutor(max_workers=3) as pool:
                results = list(pool.map(lambda _: cached_frame(root, "test", "same", lambda: frame), range(3)))
            for result in results:
                pd.testing.assert_frame_equal(result, frame)
            pd.testing.assert_frame_equal(pd.read_parquet(root / "test" / "same.parquet"), frame)
            prune_cache(root, budget=0)
            self.assertFalse(list(root.glob("*/*.parquet")))


class ComparisonChecks(unittest.TestCase):
    def test_parallel_matches_independent_strategies(self):
        prices = price_fixture(300)
        config = AllocationConfig(max_weight=.6, solver_backend="convex", max_volatility=.10)
        backtest = BacktestConfig(train_days=200)
        with patch("ml_portfolio.backtest.build_context", wraps=build_context) as build:
            parallel = run_strategy_comparison(prices, config, backtest, cache_dir=None)
            self.assertEqual(build.call_count, 1)
        # Each `rebalances` frame carries wall-clock `solver_build_seconds`/
        # `solver_solve_seconds` columns. Comparing those for exact equality
        # between a 4-way-concurrent run and an independent sequential run was
        # never a meaningful check - real wall-clock timing under thread
        # contention vs. uncontended is not expected to match, regardless of
        # solver internals. Every other column - including the actual decided
        # weights, costs, and solver convergence/iteration diagnostics -
        # stays in the comparison and must match exactly.
        timing_columns = ["solver_build_seconds", "solver_solve_seconds"]
        for method in METHODS:
            reference = run_walk_forward_backtest(prices, replace(config, method=method), backtest, cache_dir=None)
            for name, value in reference.items():
                if isinstance(value, pd.DataFrame):
                    left, right = parallel[method][name], value
                    drop = [c for c in timing_columns if c in right.columns]
                    if drop:
                        left, right = left.drop(columns=drop), right.drop(columns=drop)
                    pd.testing.assert_frame_equal(left, right)
                elif isinstance(value, pd.Series):
                    pd.testing.assert_series_equal(parallel[method][name], value)

    def test_vectorized_drift_matches_daily_accounting(self):
        prices = price_fixture(300)
        result = run_walk_forward_backtest(prices, AllocationConfig(method="Equal weight"),
                                           BacktestConfig(train_days=200, transaction_cost_bps=35), cache_dir=None)
        current = pd.Series([0., 0., 0., 1.], index=["A", "B", "C", "CASH"])
        equity = gross = 1.
        for idx in range(200, len(prices)-1):
            date = prices.index[idx]
            cost = 0.
            if date in result["weights"].index:
                current = result["weights"].loc[date].copy()
                cost = result["rebalances"].set_index("date").loc[date, "cost"]
            ret = prices.iloc[idx+1] / prices.iloc[idx] - 1
            growth = 1 + current.reindex(prices.columns).dot(ret)
            gross *= growth
            equity *= (1-cost) * growth
            current.loc[prices.columns] *= 1 + ret
            current /= current.sum()
            observed = result["equity_curve"].loc[prices.index[idx+1]]
            self.assertAlmostEqual(observed.equity, equity, places=12)
            self.assertAlmostEqual(observed.gross_equity, gross, places=12)
            np.testing.assert_allclose(result["holdings"].loc[prices.index[idx+1]], current, atol=1e-12)


class ConvexChecks(unittest.TestCase):
    def test_matches_small_exact_allocations(self):
        symbols = ["A", "B", "C"]
        cov = pd.DataFrame([[.04, .005, .002], [.005, .06, .001], [.002, .001, .025]], index=symbols, columns=symbols)
        mu = pd.Series([.02, .01, .015], index=symbols)
        trailing = price_fixture(130).pct_change().dropna()
        for method in METHODS:
            config = AllocationConfig(method=method, max_weight=.6, max_turnover=.8)
            a = allocate_portfolio(mu, cov, trailing, replace(config, solver_backend="slsqp"), transaction_cost_bps=2)
            b = allocate_portfolio(mu, cov, trailing, replace(config, solver_backend="convex"), transaction_cost_bps=2)
            np.testing.assert_allclose(a, b, atol=2e-5, err_msg=method)

    def test_large_sector_risk_and_required_sales(self):
        symbols = [f"S{i}" for i in range(1000)]
        cov = pd.DataFrame(np.diag(np.full(1000, .04)), index=symbols, columns=symbols)
        mu = pd.Series(.01, index=symbols)
        trailing = pd.DataFrame(np.zeros((126, 1000)), columns=symbols)
        old = pd.Series(0., index=symbols + ["CASH"])
        old.iloc[0] = .08
        old.CASH = .92
        config = AllocationConfig(method="Equal weight", max_weight=.02, max_turnover=.10,
                                  max_volatility=.006, sectors={s: "Tech" for s in symbols},
                                  sector_caps={"Tech": .12})
        weights = allocate_portfolio(mu, cov, trailing, config, old)
        self.assertTrue(risk_checks(weights, cov, config, old).passed.all())
        self.assertEqual(weights.attrs["solver"]["backend"], "CLARABEL")
        with self.assertRaises(AllocationError):
            allocate_portfolio(mu, cov, trailing, replace(config, max_turnover=0), old)


class StorageChecks(unittest.TestCase):
    def test_proposal_roundtrip_dedup_and_legacy(self):
        prices = price_fixture(300)
        config = AllocationConfig(method="Equal weight", max_weight=.6)
        proposal = build_proposal(prices, pd.Series(0, index=prices.columns), 100000., config, 200, 10)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = save_proposal(root, proposal, prices, config, 200, 10)
            payload = load_artifact(path)
            pd.testing.assert_frame_equal(payload["prices"], prices, check_freq=False)
            pd.testing.assert_frame_equal(payload["proposal"]["decision"]["covariance"], proposal["decision"]["covariance"])
            pd.testing.assert_series_equal(payload["proposal"]["current_weights"], proposal["current_weights"])
            save_proposal(root, proposal, prices, config, 200, 10)
            self.assertEqual(len(list((root / "prices").glob("*.parquet"))), 1)
            legacy = {"schema_version": 1, "id": "legacy", "saved_at": "2020-01-01", "proposal": proposal}
            (root / "legacy.json").write_text(json.dumps(_json_value(legacy)), encoding="utf-8")
            history = saved_proposals(root, prices.iloc[-1], prices.index[-1], prices)
            self.assertEqual(len(history), 3)
            np.testing.assert_allclose(history.marked_value, proposal["capital_after"])
            self.assertLess(path.stat().st_size, 20000)

    def test_research_roundtrip(self):
        prices = price_fixture(270)
        config, backtest = AllocationConfig(), BacktestConfig(train_days=200)
        results = run_strategy_comparison(prices, config, backtest, cache_dir=None)
        with tempfile.TemporaryDirectory() as temp:
            path = save_research(Path(temp), prices, config, backtest, results)
            loaded = load_artifact(path)
            self.assertEqual(loaded["backtest_config"], asdict(backtest))
            for method in METHODS:
                pd.testing.assert_frame_equal(loaded["results"][method]["weights"], results[method]["weights"], check_freq=False)
            with patch("ml_portfolio.experiments.read_table", wraps=read_table) as read:
                overview = research_overview(path)
                self.assertEqual(read.call_count, 4)
                self.assertTrue(all(call.kwargs["columns"] == ["equity"] for call in read.call_args_list))
            for method in METHODS:
                pd.testing.assert_frame_equal(overview[method]["equity_curve"], results[method]["equity_curve"][["equity"]])
            self.assertTrue(saved_proposals(Path(temp), prices.iloc[-1], prices.index[-1]).empty)

    def test_partial_write_not_published(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(pd.DataFrame, "to_parquet", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    write_artifact(Path(temp), "failed", {"table": pd.DataFrame({"a": [1]})})
            self.assertFalse(list(Path(temp).iterdir()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
