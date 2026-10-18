from __future__ import annotations

import json
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
from ml_portfolio.ledger import append_transactions, save_paper_account, save_transaction_ledger
from ml_portfolio.optimizers import AllocationConfig
from ml_portfolio.portfolio import build_proposal, save_proposal
from ml_portfolio.research_agent import build_price_research_report, save_research_report
from ml_portfolio.research_registry import (
    VALID_STATUSES,
    default_research_meta,
    get_research_meta,
    list_experiments,
    registry_summary,
    set_research_meta,
)
from ml_portfolio.strategy_copilot import run_strategy_dsl_backtest, save_strategy_run, strategy_from_controls


def price_fixture(rows=320):
    dates = pd.bdate_range("2020-01-01", periods=rows)
    base = np.linspace(100, 180, rows)
    return pd.DataFrame({
        "AAA": base,
        "BBB": np.linspace(100, 90, rows),
        "CCC": 100 + 8 * np.sin(np.linspace(0, 10, rows)),
    }, index=dates)


class ResearchRegistryChecks(unittest.TestCase):
    def test_fresh_artifact_has_default_unlabeled_metadata(self):
        prices = price_fixture()
        report = build_price_research_report(prices, "AAA", ["BBB"])
        with tempfile.TemporaryDirectory() as temp:
            path = save_research_report(Path(temp), "AAA brief", prices, report)
            self.assertEqual(get_research_meta(path), default_research_meta())

    def test_set_research_meta_only_updates_given_fields(self):
        prices = price_fixture()
        report = build_price_research_report(prices, "AAA", ["BBB"])
        with tempfile.TemporaryDirectory() as temp:
            path = save_research_report(Path(temp), "AAA brief", prices, report)
            set_research_meta(path, research_question="Does momentum persist?", hypothesis="Yes, 6M momentum.")
            meta = get_research_meta(path)
            self.assertEqual(meta["research_question"], "Does momentum persist?")
            self.assertEqual(meta["status"], "unlabeled")  # untouched, still the default

            set_research_meta(path, status="falsified", conclusion="Collapses after realistic costs.")
            meta = get_research_meta(path)
            # Fields from the first call survive a second call that only touches others.
            self.assertEqual(meta["research_question"], "Does momentum persist?")
            self.assertEqual(meta["hypothesis"], "Yes, 6M momentum.")
            self.assertEqual(meta["status"], "falsified")
            self.assertEqual(meta["conclusion"], "Collapses after realistic costs.")

    def test_invalid_status_rejected(self):
        prices = price_fixture()
        report = build_price_research_report(prices, "AAA", ["BBB"])
        with tempfile.TemporaryDirectory() as temp:
            path = save_research_report(Path(temp), "AAA brief", prices, report)
            with self.assertRaises(ValueError):
                set_research_meta(path, status="proven")
            # Rejected update must not have partially applied.
            self.assertEqual(get_research_meta(path)["status"], "unlabeled")

    def test_parent_id_links_a_strategy_run_to_its_source_report(self):
        prices = price_fixture()
        report = build_price_research_report(prices, "AAA", ["BBB"])
        strategy = strategy_from_controls("momentum_top", momentum_lookback=63, max_positions=1,
                                          rebalance_days=21, transaction_cost_bps=0)
        result = run_strategy_dsl_backtest(prices, strategy, evaluation_start=prices.index[126])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            report_path = save_research_report(root, "AAA brief", prices, report)
            report_id = json.loads(report_path.read_text())["id"]
            run_path = save_strategy_run(root, "Linked run", prices, result,
                                         source_research_report_id=report_id)
            set_research_meta(run_path, parent_id=report_id)
            self.assertEqual(get_research_meta(run_path)["parent_id"], report_id)
            self.assertIsNone(get_research_meta(report_path)["parent_id"])

    def test_list_experiments_excludes_operational_kinds(self):
        prices = price_fixture()
        strategy = strategy_from_controls("momentum_top", momentum_lookback=63, max_positions=1,
                                          rebalance_days=21, transaction_cost_bps=0)
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
            save_paper_account(root, transactions)

            experiments = list_experiments(root)
            self.assertEqual(len(experiments), 1)
            self.assertEqual(experiments.iloc[0]["kind"], "strategy_run")

    def test_saved_proposal_is_listed_as_an_experiment(self):
        prices = price_fixture(300)
        qty = pd.Series({"AAA": 100, "BBB": 0, "CCC": 0})
        config = AllocationConfig(method="Equal weight", max_weight=.6)
        proposal = build_proposal(prices, qty, 100000, config, 200, 10)
        self.assertTrue(proposal["accepted"])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            save_proposal(root, proposal, prices, config, 200, 10)
            experiments = list_experiments(root)
            self.assertEqual(len(experiments), 1)
            self.assertEqual(experiments.iloc[0]["kind"], "proposal")

    def test_list_experiments_defaults_missing_kind_to_proposal(self):
        # Mirrors portfolio.py's saved_proposals(): manifests saved before the
        # "kind" field existed have no kind at all and must still be treated
        # as proposals, not silently dropped from the registry.
        prices = price_fixture(300)
        qty = pd.Series({"AAA": 100, "BBB": 0, "CCC": 0})
        config = AllocationConfig(method="Equal weight", max_weight=.6)
        proposal = build_proposal(prices, qty, 100000, config, 200, 10)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = save_proposal(root, proposal, prices, config, 200, 10)
            payload = json.loads(path.read_text())
            del payload["kind"]
            path.write_text(json.dumps(payload))
            experiments = list_experiments(root)
            self.assertEqual(len(experiments), 1)
            self.assertEqual(experiments.iloc[0]["kind"], "proposal")

    def test_registry_summary_counts_by_status_and_kind(self):
        prices = price_fixture()
        report = build_price_research_report(prices, "AAA", ["BBB"])
        strategy = strategy_from_controls("momentum_top", momentum_lookback=63, max_positions=1,
                                          rebalance_days=21, transaction_cost_bps=0)
        result = run_strategy_dsl_backtest(prices, strategy, evaluation_start=prices.index[126])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            report_path = save_research_report(root, "AAA brief", prices, report)
            set_research_meta(report_path, status="supported")
            run_path = save_strategy_run(root, "Run", prices, result)
            set_research_meta(run_path, status="falsified")

            summary = registry_summary(root)
            self.assertEqual(summary["total"], 2)
            self.assertEqual(summary["by_status"]["supported"], 1)
            self.assertEqual(summary["by_status"]["falsified"], 1)
            self.assertEqual(summary["by_status"]["unlabeled"], 0)
            self.assertEqual(summary["by_kind"]["research_report"], 1)
            self.assertEqual(summary["by_kind"]["strategy_run"], 1)

    def test_registry_summary_empty_root(self):
        with tempfile.TemporaryDirectory() as temp:
            summary = registry_summary(Path(temp))
            self.assertEqual(summary["total"], 0)
            self.assertEqual(summary["by_status"], {status: 0 for status in VALID_STATUSES})


if __name__ == "__main__":
    unittest.main(verbosity=2)
