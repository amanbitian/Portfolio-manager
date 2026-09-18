from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.append(str(ROOT / ".app-deps"))

import numpy as np
import pandas as pd

from ml_portfolio.convex import allocate_convex
from ml_portfolio.models import _factor_covariance, shrinkage_covariance
from ml_portfolio.optimizers import AllocationConfig, allocate_portfolio


def correlated_returns(n_assets=50, periods=252, base_corr=0.4, seed=3):
    rng = np.random.default_rng(seed)
    corr = np.full((n_assets, n_assets), base_corr)
    np.fill_diagonal(corr, 1.0)
    vol = np.full(n_assets, 0.015)
    cov = corr * np.outer(vol, vol)
    returns = rng.multivariate_normal(np.zeros(n_assets), cov, size=periods)
    symbols = [f"S{i}" for i in range(n_assets)]
    return pd.DataFrame(returns, columns=symbols), base_corr, symbols


class FactorModelChecks(unittest.TestCase):
    def test_recovers_known_correlation_better_than_diagonal_null(self):
        returns, true_corr, symbols = correlated_returns()
        frame = _factor_covariance(returns, symbols, n_factors=5)
        cov = frame.to_numpy()
        vol = np.sqrt(np.diag(cov))
        corr = cov / np.outer(vol, vol)
        off_diagonal = corr[~np.eye(len(symbols), dtype=bool)]
        recovered = float(off_diagonal.mean())
        # The old diagonal approximation implies zero off-diagonal correlation;
        # the factor model must land materially closer to the true value than that.
        self.assertGreater(recovered, true_corr * 0.5)
        self.assertLess(recovered, true_corr * 1.5)

    def test_positive_semidefinite_across_shapes(self):
        for n, t, k in [(50, 252, 5), (1000, 252, 10), (1000, 60, 3), (30, 25, 10)]:
            returns, _, symbols = correlated_returns(n_assets=n, periods=t)
            frame = _factor_covariance(returns, symbols, n_factors=k)
            eigenvalues = np.linalg.eigvalsh(frame.to_numpy())
            self.assertGreaterEqual(eigenvalues.min(), -1e-8, msg=f"n={n}, t={t}, k={k}")

    def test_shrinkage_covariance_reports_factor_above_the_limit(self):
        returns, _, symbols = correlated_returns(n_assets=500, periods=252)
        frame = shrinkage_covariance(returns, symbols, full_covariance_asset_limit=400)
        self.assertEqual(frame.attrs["approximation"], "factor")
        off_diagonal = frame.to_numpy()[~np.eye(len(symbols), dtype=bool)]
        self.assertTrue((off_diagonal != 0).any())

    def test_shrinkage_covariance_below_limit_unchanged(self):
        returns, _, symbols = correlated_returns(n_assets=50, periods=252)
        frame = shrinkage_covariance(returns, symbols, full_covariance_asset_limit=400)
        self.assertNotIn("approximation", frame.attrs)

    def test_insufficient_history_fallback_unchanged(self):
        symbols = [f"S{i}" for i in range(10)]
        returns = pd.DataFrame(np.zeros((5, 10)), columns=symbols)
        frame = shrinkage_covariance(returns, symbols)
        expected = np.eye(10) * 0.04
        np.testing.assert_allclose(frame.to_numpy(), expected)

    def test_end_to_end_risk_model_reporting(self):
        returns, _, symbols = correlated_returns(n_assets=500, periods=252)
        covariance = shrinkage_covariance(returns, symbols, full_covariance_asset_limit=400)
        mu = pd.Series(0.0, index=symbols)
        trailing = returns
        config = AllocationConfig(method="Minimum variance", max_weight=0.02)
        weights = allocate_portfolio(mu, covariance, trailing, config)
        self.assertEqual(weights.attrs["solver"]["risk_model"], "factor")
        self.assertIn("build_seconds", weights.attrs["solver"])
        self.assertIn("solve_seconds", weights.attrs["solver"])
        self.assertGreaterEqual(weights.attrs["solver"]["build_seconds"], 0.0)
        self.assertGreaterEqual(weights.attrs["solver"]["solve_seconds"], 0.0)


class LowRankSolverPathChecks(unittest.TestCase):
    """The low-rank formulation in convex.py must be mathematically equivalent
    to reconstructing the dense covariance - it exists purely to keep the
    conic program's per-iteration cost down, not to change the answer."""

    def _setup(self, n_assets, n_factors=5, max_volatility=0.15):
        returns, _, symbols = correlated_returns(n_assets=n_assets, periods=252)
        frame = _factor_covariance(returns, symbols, n_factors=n_factors)
        cov = frame.to_numpy()
        mu = pd.Series(0.0, index=symbols)
        old = pd.Series(0.0, index=symbols + ["CASH"])
        old.loc["CASH"] = 1.0
        config = AllocationConfig(method="Minimum variance", max_weight=5.0 / n_assets,
                                  max_volatility=max_volatility)
        factor_structure = {"loadings": frame.attrs["loadings"], "factor_variance": frame.attrs["factor_variance"],
                            "specific_variance": frame.attrs["specific_variance"]}
        return mu, cov, returns, config, symbols, old, factor_structure

    def test_fast_path_matches_dense_reconstruction(self):
        mu, cov, trailing, config, symbols, old, factor_structure = self._setup(n_assets=60)
        fast = allocate_convex(mu, cov, trailing, config, symbols, old, 10.0, factor_structure=factor_structure)
        dense = allocate_convex(mu, cov, trailing, config, symbols, old, 10.0, factor_structure=None)
        pd.testing.assert_series_equal(fast.sort_index(), dense.sort_index(), check_exact=False, atol=1e-3)
        # Both must respect the same volatility ceiling on the SAME underlying covariance.
        covariance = pd.DataFrame(cov, index=symbols, columns=symbols)
        fast_vol = float(np.sqrt(fast.drop("CASH").to_numpy() @ covariance.to_numpy() @ fast.drop("CASH").to_numpy()))
        self.assertLessEqual(fast_vol, config.max_volatility + 1e-3)

    def test_fast_path_is_materially_faster_at_moderate_scale(self):
        # n=300 is enough for the dense Cholesky path's O(n^3) cost to show up
        # clearly without slowing the test suite the way n=1000 would.
        mu, cov, trailing, config, symbols, old, factor_structure = self._setup(n_assets=300, max_volatility=0.2)
        fast = allocate_convex(mu, cov, trailing, config, symbols, old, 10.0, factor_structure=factor_structure)
        dense = allocate_convex(mu, cov, trailing, config, symbols, old, 10.0, factor_structure=None)
        fast_time = fast.attrs["solver"]["build_seconds"] + fast.attrs["solver"]["solve_seconds"]
        dense_time = dense.attrs["solver"]["build_seconds"] + dense.attrs["solver"]["solve_seconds"]
        self.assertLess(fast_time, dense_time)


if __name__ == "__main__":
    unittest.main(verbosity=2)
