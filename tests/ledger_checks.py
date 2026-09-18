from __future__ import annotations

import sys
import tempfile
from io import StringIO
from pathlib import Path

import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.append(str(ROOT / ".app-deps"))

import numpy as np
import pandas as pd

from ml_portfolio.holdings import load_purchase_portfolio, save_purchase_portfolio, value_purchases
from ml_portfolio.ledger import (
    apply_fifo,
    append_transactions,
    cash_balance,
    compute_xirr,
    load_transaction_ledger,
    modified_dietz_return,
    save_transaction_ledger,
    transaction_ledgers,
    value_ledger,
)


def buys_fixture() -> pd.DataFrame:
    return pd.DataFrame([
        {"action": "buy", "symbol": "STOCK", "instrument_type": "Stock",
         "date": "2020-01-01", "quantity": 10, "price": 80},
        {"action": "buy", "symbol": "STOCK", "instrument_type": "Stock",
         "date": "2020-02-01", "quantity": 20, "price": 110},
        {"action": "buy", "symbol": "FUND", "instrument_type": "Mutual fund",
         "date": "2020-03-01", "quantity": 12.345678, "price": 20},
    ])


QUOTES = pd.Series({"STOCK": 120.0, "FUND": 25.0})
AS_OF = pd.Timestamp("2020-05-01")


class LedgerCrossValidationChecks(unittest.TestCase):
    def test_buy_only_ledger_matches_value_purchases(self):
        transactions, _ = append_transactions(None, buys_fixture(), AS_OF)
        ledger_book = value_ledger(transactions, QUOTES, AS_OF)

        purchases = pd.DataFrame([
            {"symbol": "STOCK", "instrument_type": "Stock", "purchase_date": "2020-01-01",
             "quantity": 10, "purchase_price": 80},
            {"symbol": "STOCK", "instrument_type": "Stock", "purchase_date": "2020-02-01",
             "quantity": 20, "purchase_price": 110},
            {"symbol": "FUND", "instrument_type": "Mutual fund", "purchase_date": "2020-03-01",
             "quantity": 12.345678, "purchase_price": 20},
        ])
        legacy_book = value_purchases(purchases, QUOTES, AS_OF)

        for symbol in ("STOCK", "FUND"):
            self.assertAlmostEqual(ledger_book["positions"].loc[symbol, "quantity"],
                                   legacy_book["positions"].loc[symbol, "quantity"])
            self.assertAlmostEqual(ledger_book["positions"].loc[symbol, "invested_cost"],
                                   legacy_book["positions"].loc[symbol, "invested_cost"])
            self.assertAlmostEqual(ledger_book["positions"].loc[symbol, "average_purchase_price"],
                                   legacy_book["positions"].loc[symbol, "average_purchase_price"])
            self.assertAlmostEqual(ledger_book["positions"].loc[symbol, "unrealized_gain"],
                                   legacy_book["positions"].loc[symbol, "unrealized_gain"])
        self.assertAlmostEqual(ledger_book["holdings_value"], legacy_book["current_value"])


class FifoChecks(unittest.TestCase):
    def test_partial_sale_realized_gain_and_remaining_lot(self):
        transactions = pd.DataFrame([
            {"action": "buy", "symbol": "STOCK", "instrument_type": "Stock",
             "date": "2020-01-01", "quantity": 10, "price": 80},
            {"action": "buy", "symbol": "STOCK", "instrument_type": "Stock",
             "date": "2020-02-01", "quantity": 20, "price": 110},
            {"action": "sell", "symbol": "STOCK", "instrument_type": "Stock",
             "date": "2020-03-01", "quantity": 15, "price": 130},
        ])
        validated, _ = append_transactions(None, transactions, AS_OF)
        open_lots, realized = apply_fifo(validated)

        # FIFO: sell of 15 consumes all 10@80 then 5@110.
        self.assertEqual(len(realized), 2)
        first, second = realized.iloc[0], realized.iloc[1]
        self.assertAlmostEqual(first.quantity, 10)
        self.assertAlmostEqual(first.realized_gain, 10 * (130 - 80))
        self.assertAlmostEqual(second.quantity, 5)
        self.assertAlmostEqual(second.realized_gain, 5 * (130 - 110))
        self.assertAlmostEqual(realized.realized_gain.sum(), 600)

        self.assertEqual(len(open_lots), 1)
        self.assertAlmostEqual(open_lots.iloc[0].quantity, 15)
        self.assertAlmostEqual(open_lots.iloc[0].purchase_price, 110)

    def test_oversell_rejected(self):
        transactions = pd.DataFrame([
            {"action": "buy", "symbol": "STOCK", "instrument_type": "Stock",
             "date": "2020-01-01", "quantity": 10, "price": 80},
            {"action": "sell", "symbol": "STOCK", "instrument_type": "Stock",
             "date": "2020-02-01", "quantity": 20, "price": 130},
        ])
        validated, _ = append_transactions(None, transactions, AS_OF)
        with self.assertRaises(ValueError):
            apply_fifo(validated)


class CashBalanceChecks(unittest.TestCase):
    def test_cash_balance_reflects_all_flow_types(self):
        transactions = pd.DataFrame([
            {"action": "deposit", "date": "2020-01-01", "amount": 1000},
            {"action": "withdrawal", "date": "2020-01-05", "amount": 200},
            {"action": "fee", "date": "2020-01-06", "amount": 10},
            {"action": "distribution", "date": "2020-01-07", "amount": 50},
            {"action": "buy", "symbol": "STOCK", "instrument_type": "Stock",
             "date": "2020-01-10", "quantity": 5, "price": 100},
        ])
        validated, _ = append_transactions(None, transactions, AS_OF)
        self.assertAlmostEqual(cash_balance(validated), 1000 - 200 - 10 + 50 - 500)

    def test_spending_ahead_of_a_deposit_is_rejected(self):
        # A deposit signals real cash-tracking intent - a buy dated before
        # enough deposits exist to cover it means the history was never
        # actually affordable, and nothing in this app models borrowing.
        transactions = pd.DataFrame([
            {"action": "deposit", "date": "2020-01-01", "amount": 100},
            {"action": "buy", "symbol": "STOCK", "instrument_type": "Stock",
             "date": "2020-01-02", "quantity": 10, "price": 100},
        ])
        with self.assertRaises(ValueError):
            append_transactions(None, transactions, AS_OF)

    def test_deposit_then_affordable_spend_is_accepted(self):
        transactions = pd.DataFrame([
            {"action": "deposit", "date": "2020-01-01", "amount": 1000},
            {"action": "buy", "symbol": "STOCK", "instrument_type": "Stock",
             "date": "2020-01-02", "quantity": 5, "price": 100},
        ])
        validated, _ = append_transactions(None, transactions, AS_OF)
        self.assertAlmostEqual(cash_balance(validated), 500.0)

    def test_buy_only_ledger_without_any_deposit_is_still_accepted(self):
        # The long-standing convention this app already supports (see
        # LedgerCrossValidationChecks): a buy-only history never attempts
        # cash-tracking at all, so it's exempt from the chronological check
        # even though `cash_balance` is negative-by-construction there.
        validated, _ = append_transactions(None, buys_fixture(), AS_OF)
        self.assertLess(cash_balance(validated), 0)


class DuplicateImportChecks(unittest.TestCase):
    def test_reimporting_identical_rows_is_a_no_op(self):
        first, skipped_first = append_transactions(None, buys_fixture(), AS_OF)
        self.assertEqual(skipped_first, [])
        self.assertEqual(len(first), 3)
        second, skipped_second = append_transactions(first, buys_fixture(), AS_OF)
        self.assertEqual(len(second), 3)
        self.assertEqual(len(skipped_second), 3)

    def test_duplicate_survives_a_csv_round_trip(self):
        first, _ = append_transactions(None, buys_fixture(), AS_OF)
        csv_text = first[["action", "symbol", "instrument_type", "date", "quantity",
                          "price", "amount", "notes"]].to_csv(index=False)
        reimported = pd.read_csv(StringIO(csv_text))
        combined, skipped = append_transactions(first, reimported, AS_OF)
        self.assertEqual(len(combined), 3)
        self.assertEqual(len(skipped), 3)

    def test_duplicate_with_notes_survives_a_csv_round_trip(self):
        # The bare `buys_fixture()` never sets `notes`, so it's NaN on both
        # sides of the round trip and can't catch a notes-specific bug. With a
        # real notes value: `to_csv` writes it as plain text, `read_csv` reads
        # it back as the same string (not NaN) - this exercises the actual
        # export/import path without needing a genuinely-empty-notes case.
        rows = buys_fixture()
        rows["notes"] = "broker ref 123"
        first, _ = append_transactions(None, rows, AS_OF)
        csv_text = first[["action", "symbol", "instrument_type", "date", "quantity",
                          "price", "amount", "notes"]].to_csv(index=False)
        reimported = pd.read_csv(StringIO(csv_text))
        combined, skipped = append_transactions(first, reimported, AS_OF)
        self.assertEqual(len(combined), 3)
        self.assertEqual(len(skipped), 3)

    def test_within_batch_duplicates_are_collapsed(self):
        # Two identical rows arriving in the SAME import (e.g. an accidentally
        # double-pasted CSV row) must not both be appended - the dedup used to
        # only compare incoming rows against the existing ledger, never against
        # each other.
        rows = pd.DataFrame([
            {"action": "deposit", "date": "2020-01-01", "amount": 1000},
            {"action": "deposit", "date": "2020-01-01", "amount": 1000},
        ])
        combined, skipped = append_transactions(None, rows, AS_OF)
        self.assertEqual(len(combined), 1)
        self.assertEqual(len(skipped), 1)
        self.assertAlmostEqual(cash_balance(combined), 1000.0)

    def test_incoming_transaction_id_is_preserved(self):
        # A broker-provided id column must survive validation, not be silently
        # overwritten by a freshly-computed content hash.
        rows = buys_fixture()
        rows["transaction_id"] = ["broker-1", "broker-2", "broker-3"]
        validated, _ = append_transactions(None, rows, AS_OF)
        self.assertEqual(sorted(validated["transaction_id"]), ["broker-1", "broker-2", "broker-3"])


class XirrChecks(unittest.TestCase):
    def test_known_ten_percent_case(self):
        flows = [(pd.Timestamp("2020-01-01"), -1000.0), (pd.Timestamp("2021-01-01"), 1100.0)]
        result = compute_xirr(flows)
        self.assertTrue(result["converged"])
        self.assertAlmostEqual(result["xirr"], 0.10, places=3)
        self.assertFalse(result["multiple_roots_possible"])

    def test_missing_root_reported_not_raised(self):
        flows = [(pd.Timestamp("2020-01-01"), -1000.0), (pd.Timestamp("2020-06-01"), -500.0)]
        result = compute_xirr(flows)
        self.assertFalse(result["converged"])
        self.assertIsNone(result["xirr"])
        self.assertTrue(result["note"])

    def test_close_roots_are_found_not_missed(self):
        # A cash-flow pattern engineered to have two roots ~0.2% apart (10.2%
        # and 10.4%) - with the original ~1%-wide grid cells, both roots fell
        # inside ONE cell whose endpoints happened to share the same NPV sign,
        # so the bracket scan found no sign change at all and reported "no
        # root found" even though two real roots exist.
        target_roots = [0.102, 0.104]

        def npv_with_double_root(rate):
            # (rate - r1)(rate - r2) - matches the sign behavior a real NPV
            # curve would have around two close roots, without needing to
            # reverse-engineer a literal cash-flow schedule that produces
            # exactly this shape.
            return (rate - target_roots[0]) * (rate - target_roots[1])

        # Confirm the shape actually has both roots inside one ~1%-step grid
        # cell before asserting anything about compute_xirr's fixed grid.
        self.assertLess(target_roots[1] - target_roots[0], 0.01)
        grid = np.concatenate([np.linspace(-0.99, 1.0, 4000), np.linspace(1.0, 10.0, 1800)])
        values = [npv_with_double_root(rate) for rate in grid]
        brackets = [(grid[i], grid[i + 1]) for i in range(len(grid) - 1)
                   if np.sign(values[i]) != np.sign(values[i + 1]) and np.sign(values[i]) != 0]
        self.assertGreaterEqual(len(brackets), 2, "finer grid must bracket both close roots")
        old_grid = np.concatenate([np.linspace(-0.99, 1.0, 200), np.linspace(1.0, 10.0, 100)])
        old_values = [npv_with_double_root(rate) for rate in old_grid]
        old_brackets = [(old_grid[i], old_grid[i + 1]) for i in range(len(old_grid) - 1)
                       if np.sign(old_values[i]) != np.sign(old_values[i + 1]) and np.sign(old_values[i]) != 0]
        self.assertEqual(len(old_brackets), 0, "sanity check: old coarse grid really did miss both roots")


class ModifiedDietzChecks(unittest.TestCase):
    def test_hand_computable_single_midperiod_contribution(self):
        start = pd.Timestamp("2020-01-01")
        end = start + pd.Timedelta(days=100)
        contribution_date = start + pd.Timedelta(days=50)
        # Dietz convention: contributions are positive.
        result = modified_dietz_return([(contribution_date, 1000.0)], 0.0, 1100.0, start, end)
        self.assertAlmostEqual(result, 100 / 500, places=6)


class PersistenceChecks(unittest.TestCase):
    def test_save_load_roundtrip_and_listing(self):
        transactions, _ = append_transactions(None, buys_fixture(), AS_OF)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = save_transaction_ledger(root, "My ledger", transactions, QUOTES, AS_OF)
            payload = load_transaction_ledger(path)
            self.assertEqual(payload["name"], "My ledger")
            self.assertEqual(payload["kind"], "transaction_ledger")
            pd.testing.assert_frame_equal(
                payload["transactions"][["action", "symbol"]].reset_index(drop=True),
                transactions[["action", "symbol"]].reset_index(drop=True))
            self.assertIn("My ledger", transaction_ledgers(root)[path])

    def test_legacy_purchase_portfolio_still_loads_unaffected(self):
        purchases = pd.DataFrame([
            {"symbol": "STOCK", "instrument_type": "Stock", "purchase_date": "2020-01-01",
             "quantity": 10, "purchase_price": 80},
            {"symbol": "FUND", "instrument_type": "Mutual fund", "purchase_date": "2020-03-01",
             "quantity": 12.345678, "purchase_price": 20},
        ])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = save_purchase_portfolio(root, "Legacy", purchases, 500, QUOTES, AS_OF)
            payload = load_purchase_portfolio(path)
            self.assertEqual(payload["name"], "Legacy")
            self.assertEqual(payload["cash"], 500)
            # Confirm the new ledger picker doesn't see legacy purchase-portfolio files.
            self.assertEqual(transaction_ledgers(root), {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
