from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / ".app-deps"))

from streamlit.testing.v1 import AppTest
import pandas as pd


class WorkflowChecks(unittest.TestCase):
    def test_every_workspace_hop_renders_without_submitting_anything(self):
        # Regression for known_issues #11: hopping Workspace -> Research
        # (without running a comparison) -> Strategy Copilot once raised a
        # KeyError inside AppTest's widget-state reconciliation. It no longer
        # reproduces on the current Streamlit; every ordered pair of pages is
        # exercised from a fresh app so a recurrence is caught, not left to a
        # navigation sequence no test happens to follow.
        import itertools
        pages = ["Portfolio", "Research", "Research Agent", "Strategy Copilot", "Saved Decisions"]
        for first, second in itertools.permutations(pages, 2):
            with self.subTest(hop=f"{first} -> {second}"):
                app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
                next(r for r in app.radio if r.label == "Workspace").set_value(first).run()
                self.assertFalse(app.exception, first)
                next(r for r in app.radio if r.label == "Workspace").set_value(second).run()
                self.assertFalse(app.exception, f"{first} -> {second}")

    def test_load_named_purchase_portfolio(self):
        lots = pd.DataFrame({"symbol": ["RELIANCE"], "instrument_type": ["Stock"],
                             "purchase_date": pd.to_datetime(["2020-01-01"]),
                             "quantity": [10.], "purchase_price": [80.]})
        payload = {"name": "Long-term holdings", "cash": 450., "book": {"lots": lots}}
        path = Path("saved-portfolio.json")
        with patch("ml_portfolio.holdings.purchase_portfolios", return_value={path: "Long-term holdings"}), \
             patch("ml_portfolio.holdings.load_purchase_portfolio", return_value=payload):
            app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
            next(r for r in app.radio if r.label == "Holdings entry").set_value("Purchases").run()
            next(b for b in app.button if b.label == "Load portfolio").click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertEqual(next(t for t in app.text_input if t.label == "Portfolio name").value, "Long-term holdings")
            self.assertEqual(next(n for n in app.number_input if n.label == "Available cash").value, 450.)
            self.assertTrue(any(m.label == "Invested cost" and m.value == "800.00" for m in app.metric))

    def test_purchase_entry_save_and_proposal_invalidation(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
        purchases = pd.DataFrame({"symbol": ["RELIANCE", "HDFCBANK"],
                                  "instrument_type": ["Stock", "Mutual fund"],
                                  "purchase_date": pd.to_datetime(["2020-01-01", "2020-02-01"]),
                                  "quantity": [10., 12.345678], "purchase_price": [80., 25.]})
        app.session_state["purchase_draft"] = purchases
        app.session_state["purchase_cash"] = 100000.
        next(r for r in app.radio if r.label == "Holdings entry").set_value("Purchases").run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertTrue(any(m.label == "Invested cost" for m in app.metric))
        with patch("ml_portfolio.holdings.save_purchase_portfolio", return_value=Path("test-portfolio.json")) as save, \
             patch("ml_portfolio.portfolio.build_proposal", side_effect=AssertionError("Saving must not optimize")):
            next(b for b in app.button if b.label == "Save portfolio").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(save.call_count, 1)
        next(b for b in app.button if b.label == "Calculate proposal").click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertTrue(any(m.label == "Portfolio value" for m in app.metric))
        app.session_state["purchase_draft"] = purchases.assign(purchase_price=[100., 25.])
        app.session_state["purchase_last_valid"] = purchases.assign(purchase_price=[100., 25.])
        app.session_state["purchase_revision"] = 1
        app.run()
        self.assertTrue(any("Recalculate" in i.value for i in app.info))
        app.sidebar.radio[0].set_value("Research").run()
        app.sidebar.radio[0].set_value("Portfolio").run()
        next(r for r in app.radio if r.label == "Holdings entry").set_value("Purchases").run()
        self.assertFalse(app.exception)
        self.assertTrue(any(m.label == "Invested cost" for m in app.metric))

    def test_large_table_pagination_and_filter(self):
        script = """
import pandas as pd
from app import table_preview
frame = pd.DataFrame({'symbol': [f'S{i:04d}' for i in range(6000)], 'value': range(6000)})
table_preview(frame, 'test_table')
"""
        app = AppTest.from_string(script, default_timeout=60).run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.dataframe[0].value), 100)
        app.number_input[0].set_value(60).run()
        self.assertEqual(app.dataframe[0].value.iloc[-1].symbol, "S5999")
        app.text_input[0].set_value("S5999").run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.dataframe[0].value), 1)
        app.text_input[0].set_value("NO MATCH").run()
        self.assertFalse(app.exception)
        self.assertTrue(app.dataframe[0].value.empty)

    def test_portfolio_save_and_invalidation(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
        next(b for b in app.button if b.label == "Calculate proposal").click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertTrue(any(m.label == "Checks" and m.value == "Passed" for m in app.metric))
        with patch("ml_portfolio.portfolio.save_proposal", return_value=Path("mock-decision.json")) as save:
            next(b for b in app.button if b.label == "Save decision").click().run()
            self.assertEqual(save.call_count, 1)
            self.assertTrue(any("Saved decision" in s.value for s in app.success))
        next(n for n in app.sidebar.number_input if n.label == "Trading cost (bps)").set_value(20)
        next(b for b in app.button if b.label == "Apply settings").click().run()
        self.assertTrue(any("Recalculate" in i.value for i in app.info))
        self.assertFalse(any(b.label == "Save decision" for b in app.button))

    def test_research_and_navigation(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
        app.sidebar.radio[0].set_value("Research").run()
        next(b for b in app.button if b.label == "Run comparison").click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertTrue(any(m.label == "CAGR" for m in app.metric))
        for page in ["Research Agent", "Strategy Copilot", "Saved Decisions", "Portfolio", "Research"]:
            app.sidebar.radio[0].set_value(page).run()
            self.assertFalse(app.exception)

    def test_research_agent_and_strategy_copilot_pages(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
        app.sidebar.radio[0].set_value("Research Agent").run()
        next(b for b in app.button if b.label == "Generate research brief").click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertTrue(any(m.label == "Overall" for m in app.metric))
        next(b for b in app.button if b.label == "Use hypothesis in Strategy Copilot").click().run()
        app.sidebar.radio[0].set_value("Strategy Copilot").run()
        next(b for b in app.button if b.label == "Run strategy").click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertTrue(any(m.label == "CAGR" for m in app.metric))
        self.assertTrue(app.code)


if __name__ == "__main__":
    unittest.main(verbosity=2)
