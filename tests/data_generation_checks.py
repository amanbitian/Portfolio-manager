from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.append(str(ROOT / ".app-deps"))

import numpy as np
import pandas as pd

from ml_portfolio.data import make_synthetic_prices


class SyntheticPriceChecks(unittest.TestCase):
    def test_shape_and_positivity(self):
        symbols = ["A", "B", "C", "D"]
        prices = make_synthetic_prices(symbols=symbols, periods=400, seed=1)
        self.assertEqual(len(prices), 400)
        self.assertEqual(list(prices.columns), symbols)
        self.assertTrue((prices > 0).all().all())

    def test_exact_row_count_across_a_weekend_end_date(self):
        # 2024-01-06 is a Saturday; the BDay rollback guard keeps the row count exact.
        prices = make_synthetic_prices(periods=250, seed=7, end_date="2024-01-06")
        self.assertEqual(len(prices), 250)
        self.assertTrue(prices.index.is_monotonic_increasing)

    def test_deterministic_given_seed(self):
        first = make_synthetic_prices(periods=300, seed=99)
        second = make_synthetic_prices(periods=300, seed=99)
        pd.testing.assert_frame_equal(first, second)

    def test_different_seed_gives_different_output(self):
        first = make_synthetic_prices(periods=300, seed=1)
        second = make_synthetic_prices(periods=300, seed=2)
        self.assertFalse(first.equals(second))

    def test_bull_regime_drifts_above_bear_regime(self):
        # A single seed's regime-segment mean is noisy relative to the drift signal
        # (daily vol ~1.2-2.2% against a ~0.0011 drift gap, only partly averaged out
        # by 8 assets correlated at 0.32) - roughly an 80% per-seed win rate, verified
        # empirically. Check a majority across fixed seeds instead of one seed.
        periods = 900
        wins = 0
        seeds = range(20)
        for seed in seeds:
            prices = make_synthetic_prices(periods=periods, seed=seed)
            returns = np.log(prices).diff()
            bear = returns.iloc[periods // 3 : periods // 2].to_numpy().mean()
            bull = returns.iloc[periods // 2 : periods * 2 // 3].to_numpy().mean()
            wins += bull > bear
        self.assertGreaterEqual(wins, 14, f"Only {wins}/{len(seeds)} seeds showed bull > bear drift.")

    def test_return_correlation_matches_design(self):
        # A stable (non-high-vol) regime segment; a transpose bug in the batched
        # Cholesky draw would silently produce the wrong correlation structure
        # while still passing the checks above.
        periods = 900
        prices = make_synthetic_prices(
            symbols=[f"S{i}" for i in range(10)], periods=periods, seed=42
        )
        returns = np.log(prices).diff().iloc[1 : periods // 3]
        corr = returns.corr().to_numpy()
        off_diagonal = corr[~np.eye(corr.shape[0], dtype=bool)]
        self.assertTrue(np.all(off_diagonal > 0.1))
        self.assertTrue(np.all(off_diagonal < 0.6))


if __name__ == "__main__":
    unittest.main(verbosity=2)
