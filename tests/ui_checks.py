from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.append(str(ROOT / ".app-deps"))

from streamlit.testing.v1 import AppTest


class WorkflowChecks(unittest.TestCase):
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
        for page in ["Saved Decisions", "Portfolio", "Research"]:
            app.sidebar.radio[0].set_value(page).run()
            self.assertFalse(app.exception)


if __name__ == "__main__":
    unittest.main(verbosity=2)
