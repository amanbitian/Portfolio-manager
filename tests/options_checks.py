from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ml_portfolio.options import (OptionAssumptions, black_scholes, expiry_payoff, scenario_matrix,
                                  strategy_legs, strategy_summary)


class OptionsEngineChecks(unittest.TestCase):
    def setUp(self):
        self.assumptions = OptionAssumptions(100.0, 30, 0.20, 0.05, lot_size=1)

    def test_black_scholes_put_call_parity_and_greeks(self):
        call = black_scholes(100, 100, 1, 0.2, 0.05, option_type="Call")
        put = black_scholes(100, 100, 1, 0.2, 0.05, option_type="Put")
        self.assertAlmostEqual(call["price"] - put["price"], 100 - 100 * np.exp(-0.05), places=8)
        self.assertGreater(call["delta"], 0)
        self.assertLess(put["delta"], 0)
        self.assertGreater(call["gamma"], 0)
        self.assertGreater(call["vega"], 0)

    def test_long_straddle_is_symmetric_at_expiry(self):
        legs = strategy_legs("Long Straddle", 100, 10)
        payoff = expiry_payoff(legs, self.assumptions, np.array([80.0, 100.0, 120.0]))
        self.assertAlmostEqual(payoff.pnl.iloc[0], payoff.pnl.iloc[2], places=8)
        self.assertLess(payoff.pnl.iloc[1], 0)
        summary = strategy_summary(legs, self.assumptions)
        self.assertIsNone(summary["max_profit"])
        self.assertEqual(len(summary["breakevens"]), 2)

    def test_iron_condor_has_credit_and_defined_risk(self):
        legs = strategy_legs("Iron Condor", 100, 5)
        summary = strategy_summary(legs, self.assumptions)
        self.assertGreater(summary["net_premium"], 0)
        self.assertIsNotNone(summary["max_profit"])
        self.assertIsNotNone(summary["max_loss"])
        self.assertGreater(summary["max_profit"], 0)
        self.assertLess(summary["max_loss"], 0)
        self.assertEqual(len(summary["breakevens"]), 2)

    def test_custom_short_put_includes_zero_price_tail_risk(self):
        legs = strategy_legs("Custom", 100, 5)
        legs.loc[0, ["side", "option_type"]] = ["Sell", "Put"]
        summary = strategy_summary(legs, self.assumptions)
        premium = summary["net_premium"]
        self.assertAlmostEqual(summary["max_loss"], -(100.0 - premium), places=8)
        self.assertIsNotNone(summary["max_profit"])

    def test_scenario_matrix_shape_and_current_entry_anchor(self):
        legs = strategy_legs("Call Butterfly", 100, 5)
        scenarios = scenario_matrix(legs, self.assumptions, spot_moves=(0.0,), vol_moves=(0.0,),
                                    days_remaining=30)
        self.assertEqual(scenarios.shape, (1, 1))
        self.assertAlmostEqual(scenarios.iloc[0, 0], 0.0, places=8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
