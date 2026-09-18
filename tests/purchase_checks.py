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

from ml_portfolio.holdings import (PURCHASE_COLUMNS, load_purchase_portfolio, purchase_portfolios,
                                  save_purchase_portfolio, value_purchases)
from ml_portfolio.optimizers import AllocationConfig
from ml_portfolio.portfolio import build_proposal, save_proposal, saved_proposals
from ml_portfolio.storage import load_artifact


def lots_fixture():
    return pd.DataFrame([
        ["STOCK", "Stock", "2020-01-01", 10., 80.],
        ["STOCK", "Stock", "2020-02-01", 20., 110.],
        ["FUND", "Mutual fund", "2020-03-01", 12.345678, 20.],
    ], columns=PURCHASE_COLUMNS)


class PurchaseChecks(unittest.TestCase):
    def setUp(self):
        self.quotes = pd.Series({"STOCK": 120., "FUND": 25.})
        self.as_of = pd.Timestamp("2020-05-01")

    def test_repeated_lots_and_weighted_cost(self):
        book = value_purchases(lots_fixture(), self.quotes, self.as_of)
        position = book["positions"].loc["STOCK"]
        self.assertEqual(position.quantity, 30)
        self.assertEqual(position.average_purchase_price, 100)
        self.assertEqual(position.unrealized_gain, 600)
        self.assertEqual(position.lots, 2)
        self.assertEqual(book["lots"].iloc[0].holding_days, 121)
        self.assertAlmostEqual(book["invested_cost"], 3000 + 12.345678 * 20)
        self.assertAlmostEqual(book["current_value"], 3600 + 12.345678 * 25)
        self.assertAlmostEqual(book["return_pct"], book["unrealized_gain"] / book["invested_cost"])

    def test_bad_lots_rejected(self):
        for column, value in [("quantity", -1), ("quantity", np.nan), ("purchase_price", np.inf),
                              ("purchase_price", 0), ("purchase_date", "2021-01-01"),
                              ("purchase_date", None), ("purchase_date", "bad-date"),
                              ("purchase_date", 123),
                              ("instrument_type", "Unknown"), ("symbol", "MISSING")]:
            with self.subTest(column=column, value=value):
                lots = lots_fixture()
                lots[column] = lots[column].astype(object)
                lots.loc[0, column] = value
                with self.assertRaises(ValueError):
                    value_purchases(lots, self.quotes, self.as_of)
        lots = lots_fixture()
        lots.loc[0, "quantity"] = .5
        with self.assertRaises(ValueError):
            value_purchases(lots, self.quotes, self.as_of)
        lots = lots_fixture()
        lots.loc[1, "instrument_type"] = "Mutual fund"
        with self.assertRaises(ValueError):
            value_purchases(lots, self.quotes, self.as_of)

    def test_save_reload_and_revalue(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = save_purchase_portfolio(root, "My funds", lots_fixture(), 500, self.quotes, self.as_of)
            payload = load_purchase_portfolio(path)
            self.assertEqual(payload["name"], "My funds")
            self.assertIn("My funds", purchase_portfolios(root)[path])
            self.assertEqual(payload["cash"], 500)
            book = value_purchases(payload["book"]["lots"], self.quotes * 2, self.as_of)
            self.assertAlmostEqual(book["current_value"], payload["book"]["current_value"] * 2)
            self.assertEqual(book["invested_cost"], payload["book"]["invested_cost"])
            self.assertTrue(saved_proposals(root, self.quotes, self.as_of).empty)

    def test_fractional_fund_proposal_and_snapshot(self):
        prices = pd.DataFrame({"STOCK": 120., "FUND": 25.}, index=pd.bdate_range("2020-01-01", periods=300))
        lots = lots_fixture()
        quantities = lots.groupby("symbol").quantity.sum()
        types = {"STOCK": "Stock", "FUND": "Mutual fund"}
        config = AllocationConfig(method="Equal weight", max_weight=.8)
        proposal = build_proposal(prices, quantities, 10000, config, 200, 10,
                                  instrument_types=types, purchases=lots)
        trades = proposal["trades"].set_index("symbol")
        self.assertEqual(trades.loc["FUND", "current_quantity"], 12.345678)
        self.assertNotEqual(trades.loc["FUND", "trade_quantity"], int(trades.loc["FUND", "trade_quantity"]))
        self.assertAlmostEqual(trades.loc["STOCK", "target_quantity"] % 1, 0)
        self.assertAlmostEqual(trades.loc["FUND", "target_quantity"] * 1e6,
                               round(trades.loc["FUND", "target_quantity"] * 1e6), places=5)
        self.assertAlmostEqual(proposal["cash_after"], 10000 - trades.trade_value.sum() - trades.estimated_fee.sum())
        self.assertTrue(proposal["accepted"])
        with tempfile.TemporaryDirectory() as temp:
            path = save_proposal(Path(temp), proposal, prices, config, 200, 10)
            restored = load_artifact(path)["proposal"]
            pd.testing.assert_frame_equal(restored["purchase_book"]["lots"], proposal["purchase_book"]["lots"])
            pd.testing.assert_frame_equal(restored["trades"], proposal["trades"])

    def test_lots_must_reconcile_and_cost_does_not_drive_allocation(self):
        prices = pd.DataFrame({"STOCK": 120., "FUND": 25.}, index=pd.bdate_range("2020-01-01", periods=300))
        lots = lots_fixture()
        quantities = lots.groupby("symbol").quantity.sum()
        types = {"STOCK": "Stock", "FUND": "Mutual fund"}
        config = AllocationConfig(method="Equal weight", max_weight=.8)
        with self.assertRaises(ValueError):
            build_proposal(prices, quantities * 2, 10000, config, 200, 10, instrument_types=types, purchases=lots)
        a = build_proposal(prices, quantities, 10000, config, 200, 10, instrument_types=types, purchases=lots)
        b = build_proposal(prices, quantities, 10000, config, 200, 10, instrument_types=types,
                           purchases=lots.assign(purchase_price=lots.purchase_price * 2))
        pd.testing.assert_series_equal(a["decision"]["weights"], b["decision"]["weights"])
        self.assertNotEqual(a["purchase_book"]["unrealized_gain"], b["purchase_book"]["unrealized_gain"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
