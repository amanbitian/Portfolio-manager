from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.append(str(ROOT / ".app-deps"))

import numpy as np
import pandas as pd

from ml_portfolio.strategy_copilot import (
    StrategyParseError,
    _rsi,
    _score_frame,
    compile_strategy_prompt,
)


def _wilder_rsi_reference(values: list[float], period: int) -> list[float]:
    """Independent, literal day-by-day implementation of Wilder's RSI recursion."""
    n = len(values)
    out = [np.nan] * n
    gains = [max(values[i] - values[i - 1], 0.0) for i in range(1, n)]
    losses = [max(values[i - 1] - values[i], 0.0) for i in range(1, n)]
    if len(gains) < period:
        return out
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    out[period] = 100.0 if avg_loss == 0 and avg_gain > 0 else (
        50.0 if avg_loss == 0 else 100 - 100 / (1 + avg_gain / avg_loss))
    for k in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[k]) / period
        avg_loss = (avg_loss * (period - 1) + losses[k]) / period
        idx = k + 1
        if avg_loss == 0 and avg_gain > 0:
            out[idx] = 100.0
        elif avg_loss == 0:
            out[idx] = 50.0
        else:
            out[idx] = 100 - 100 / (1 + avg_gain / avg_loss)
    return out


class RsiChecks(unittest.TestCase):
    def test_matches_independent_wilder_reference(self):
        rng = np.random.default_rng(3)
        n, period = 40, 5
        values = 100 + np.cumsum(rng.normal(0, 1, n))
        values[10:15] = values[9]  # a flat (no-movement) stretch
        prices = pd.DataFrame({"A": values})

        actual = _rsi(prices, period)["A"].to_numpy()
        expected = _wilder_rsi_reference(list(values), period)

        for i, (got, want) in enumerate(zip(actual, expected)):
            if want != want:  # NaN
                self.assertTrue(np.isnan(got), f"row {i}: expected NaN, got {got}")
            else:
                self.assertAlmostEqual(got, want, places=9, msg=f"row {i}")

    def test_flat_window_is_fifty_and_pure_uptrend_is_hundred(self):
        flat = pd.DataFrame({"A": [100.0] * 20})
        rsi_flat = _rsi(flat, 5)["A"]
        self.assertTrue((rsi_flat.dropna() == 50.0).all())

        uptrend = pd.DataFrame({"A": np.arange(100.0, 120.0)})
        rsi_up = _rsi(uptrend, 5)["A"]
        self.assertTrue((rsi_up.dropna() == 100.0).all())


class SmaCrossoverChecks(unittest.TestCase):
    def test_cross_mode_fires_only_near_the_crossing_day_trend_mode_persists(self):
        # fast=2, slow=3: a short deterministic dip-then-rise series with one
        # clear upward crossing.
        values = [100, 99, 98, 97, 100, 103, 106, 109, 112, 115, 118, 121]
        prices = pd.DataFrame({"A": values, "B": [100.0] * len(values)})
        strategy = {"entry": {"template": "sma_cross", "fast_window": 2, "slow_window": 3,
                              "signal_mode": "cross"}, "rebalance_days": 1}
        cross_scores = _score_frame(prices, strategy)["A"]
        strategy_trend = {"entry": {"template": "sma_cross", "fast_window": 2, "slow_window": 3,
                                    "signal_mode": "trend"}, "rebalance_days": 1}
        trend_scores = _score_frame(prices, strategy_trend)["A"]

        fast = prices["A"].rolling(2).mean()
        slow = prices["A"].rolling(3).mean()
        crossing_days = ((fast - slow).shift(1) <= 0) & ((fast - slow) > 0)
        self.assertTrue(crossing_days.any(), "fixture must contain a crossing day")
        first_cross = crossing_days[crossing_days].index[0]

        self.assertFalse(np.isnan(cross_scores.loc[first_cross]))
        self.assertTrue(trend_scores.loc[first_cross:].notna().all())


class ParserPrecedenceChecks(unittest.TestCase):
    def test_position_count_does_not_leak_into_sma_windows(self):
        strategy = compile_strategy_prompt("SMA crossover, hold 10 positions, 50/200")
        self.assertEqual(strategy["entry"]["fast_window"], 50)
        self.assertEqual(strategy["entry"]["slow_window"], 200)
        self.assertEqual(strategy["portfolio"]["max_positions"], 10)

    def test_bare_rsi_number_is_ambiguous(self):
        with self.assertRaises(StrategyParseError):
            compile_strategy_prompt("RSI 30")

    def test_ambiguous_sma_numbers_without_attribution_raise(self):
        with self.assertRaises(StrategyParseError):
            compile_strategy_prompt("SMA cross using 20, 50 and 100 somehow")


if __name__ == "__main__":
    unittest.main(verbosity=2)
