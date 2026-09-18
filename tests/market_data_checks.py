from __future__ import annotations

import json
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.append(str(ROOT))              # for `import data_pipeline.config`, same as app.py's own path
sys.path.append(str(ROOT / ".app-deps"))

import numpy as np
import pandas as pd

from ml_portfolio.data import load_price_csv
from ml_portfolio.market_data import (
    MarketDataError,
    dataset_fingerprint,
    load_corporate_actions,
    load_market_prices,
    load_universe,
)

REAL_DATA_ROOT = Path("F:/quants project/stock Data")


def _write_symbol_parquet(root: Path, symbol: str, dates: pd.DatetimeIndex, closes: np.ndarray,
                          extra_duplicate_row: bool = False) -> None:
    ts_ist = pd.DatetimeIndex(dates).tz_localize("Asia/Kolkata")
    frame = pd.DataFrame({
        "open": closes, "high": closes, "low": closes, "close": closes,
        "volume": pd.array(np.full(len(dates), 1000), dtype="Int64"),
        "open_interest": pd.array(np.zeros(len(dates)), dtype="Int64"),
        "ts_ist": ts_ist,
        "ts_utc": ts_ist.tz_convert("UTC"),
        "symbol": symbol,
    })
    if extra_duplicate_row:
        # Mirrors the real pipeline-format-boundary artifact: a same-calendar-day
        # midnight-timestamp row alongside the normal intraday-anchored row.
        midnight_row = frame.iloc[[-1]].copy()
        midnight_row["ts_ist"] = midnight_row["ts_ist"].dt.normalize()
        midnight_row["ts_utc"] = midnight_row["ts_ist"].dt.tz_convert("UTC")
        frame = pd.concat([frame.iloc[:-1], midnight_row, frame.iloc[[-1]]], ignore_index=True)
    path = root / "market" / "ohlcv_1day" / f"symbol={symbol}" / "part.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)


def _write_universe(root: Path, symbols: list[str]) -> None:
    universe = pd.DataFrame({
        "symbol": symbols, "isin": [f"INE{i:09d}" for i in range(len(symbols))],
        "name": [f"{s} Ltd" for s in symbols], "industry": "Test",
        "instrument_key": [f"NSE_EQ|INE{i:09d}" for i in range(len(symbols))],
        "yf_ticker": [f"{s}.NS" for s in symbols], "upstox_tradingsymbol": symbols,
        "source": "manual_csv", "ingested_at": "2026-01-01T00:00:00Z",
    })
    path = root / "reference" / "universe.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    universe.to_parquet(path)


def _write_manifest(root: Path) -> None:
    path = root / "_manifests" / "market_1day.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"AAA|full": {"status": "done", "rows": 300, "ts": "2026-01-01T00:00:00Z"}}))


class MarketDataFixtureChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.dates = pd.bdate_range("2024-01-01", periods=300)
        _write_symbol_parquet(self.root, "AAA", self.dates, np.linspace(100, 130, 300))
        _write_symbol_parquet(self.root, "BBB", self.dates, np.linspace(100, 90, 300))
        _write_symbol_parquet(self.root, "DUP", self.dates, np.linspace(100, 110, 300),
                              extra_duplicate_row=True)
        # STALE has no data for the final 30 days - a real gap the ffill(limit=5)
        # policy cannot cover, so it must be excluded, not forward-filled flat.
        _write_symbol_parquet(self.root, "STALE", self.dates[:-30], np.linspace(100, 105, 270))
        # NEWLY listed 100 sessions into the window - a leading gap.
        _write_symbol_parquet(self.root, "NEWLY", self.dates[100:], np.linspace(50, 80, 200))
        # BRIEF listed late AND stopped early - both edges, no interior gap.
        _write_symbol_parquet(self.root, "BRIEF", self.dates[40:-40], np.linspace(20, 30, 220))
        _write_universe(self.root, ["AAA", "BBB", "DUP", "STALE", "NEWLY", "BRIEF"])
        _write_manifest(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_missing_root_raises_clear_error(self):
        with self.assertRaises(MarketDataError):
            load_market_prices(self.root / "does-not-exist", ["AAA", "BBB"])

    def test_missing_universe_file_raises_clear_error(self):
        with self.assertRaises(MarketDataError):
            load_universe(self.root / "empty-root")

    def test_universe_lists_only_priceable_symbols(self):
        universe = load_universe(self.root)
        self.assertEqual(set(universe.symbol), {"AAA", "BBB", "DUP", "STALE", "NEWLY", "BRIEF"})

    def test_missing_symbol_file_is_excluded_with_reason(self):
        prices, report = load_market_prices(self.root, ["AAA", "BBB", "MISSING"])
        self.assertNotIn("MISSING", prices.columns)
        self.assertIn("MISSING", report["excluded"])
        self.assertEqual(report["loaded_symbols"], 2)
        self.assertEqual(report["requested_symbols"], 3)

    def test_duplicated_calendar_date_is_deduplicated_not_raised(self):
        prices, report = load_market_prices(self.root, ["AAA", "DUP"])
        self.assertIn("DUP", prices.columns)
        self.assertFalse(prices.index.duplicated().any())
        self.assertEqual(len(prices), 300)

    def test_stale_symbol_retained_with_recorded_cutoff(self):
        # Regression test: STALE has 270 real days then an unbroken gap to the
        # end of the window (a purely trailing gap, e.g. a delisting near the
        # end of the loaded range) - previously dropped from the WHOLE
        # window, which meant an early rebalance decision was silently
        # affected by data availability at the very end of the range (real
        # lookahead bias in universe construction). It must now be retained,
        # with real data intact before the cutoff and a recorded cutoff date
        # so callers stop treating it as investable after that point.
        prices, report = load_market_prices(self.root, ["AAA", "BBB", "STALE"])
        self.assertIn("STALE", prices.columns)
        self.assertNotIn("STALE", report["excluded"])
        # The cutoff lands 5 rows (the short-gap ffill(limit=5) reach) after
        # the last genuinely observed date (index 269) - a short-gap-filled
        # price is already treated as usable everywhere else in this
        # codebase, so the trailing-gap cutoff consistently extends through
        # that same boundary rather than stopping at the raw last observation.
        self.assertEqual(report["truncated"]["STALE"], str(self.dates[274]))
        pd.testing.assert_series_equal(
            prices["STALE"].iloc[:270], pd.Series(np.linspace(100, 105, 270), index=self.dates[:270], name="STALE"))
        # Past the cutoff, the price is a flat-filled placeholder, not real data.
        self.assertTrue((prices["STALE"].iloc[269:] == prices["STALE"].iloc[269]).all())

    def test_eligibility_cutoff_survives_parquet_and_pickle_round_trips(self):
        # `.attrs["eligible_through"]` is stored JSON-natively (an ISO string)
        # so it survives pandas' attrs-in-parquet-metadata path - a Timestamp
        # there made `to_parquet` raise, and any saved artifact or cached
        # frame reloaded from disk would otherwise have lost the cutoff and
        # silently reintroduced the lookahead bias on re-run.
        import pickle
        from ml_portfolio.data import eligibility_cutoffs
        from ml_portfolio.storage import atomic_parquet
        prices, report = load_market_prices(self.root, ["AAA", "STALE"])
        self.assertEqual(json.loads(json.dumps(prices.attrs)), prices.attrs)
        expected = {"STALE": self.dates[274]}
        self.assertEqual(eligibility_cutoffs(prices), expected)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "prices.parquet"
            atomic_parquet(prices, path)
            self.assertEqual(eligibility_cutoffs(pd.read_parquet(path)), expected)
        self.assertEqual(eligibility_cutoffs(pickle.loads(pickle.dumps(prices))), expected)
        # A caller-built frame may still hand over a Timestamp directly.
        prices.attrs["eligible_through"] = {"STALE": self.dates[274]}
        self.assertEqual(eligibility_cutoffs(prices), expected)

    def test_parquet_write_drops_only_non_serializable_attrs_without_mutating_the_frame(self):
        # The factor risk model hangs numpy arrays on a covariance frame's
        # attrs; those cannot be stored and are dropped for the write only,
        # while the live object keeps them.
        from ml_portfolio.storage import atomic_parquet
        frame = pd.DataFrame({"x": [1.0, 2.0]}, index=self.dates[:2])
        frame.attrs = {"loadings": np.ones((2, 1)), "note": "kept in memory"}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "cov.parquet"
            atomic_parquet(frame, path)
            self.assertEqual(pd.read_parquet(path).attrs, {})
        self.assertEqual(set(frame.attrs), {"loadings", "note"})

    def test_mid_window_listing_retained_with_recorded_start(self):
        # A symbol that lists inside the window used to be dropped for the
        # whole window (266 of 750 real names over 2018-2026). It is now kept,
        # back-filled flat at its first real price so every consumer stays
        # finite, with the first real date recorded and disclosed.
        from ml_portfolio.data import eligibility_starts, listed_only
        prices, report = load_market_prices(self.root, ["AAA", "NEWLY"])
        self.assertIn("NEWLY", prices.columns)
        self.assertEqual(report["excluded"], {})
        self.assertEqual(eligibility_starts(prices), {"NEWLY": self.dates[100]})
        self.assertEqual(report["listed_mid_window"], {"NEWLY": str(self.dates[100])})
        self.assertEqual(report["truncated"], {})
        self.assertTrue((prices["NEWLY"].iloc[:100] == 50.0).all())   # flat placeholder, no jump at listing
        self.assertAlmostEqual(prices["NEWLY"].iloc[100], 50.0)
        self.assertFalse(prices.isna().any().any())
        real = listed_only(prices)
        self.assertEqual(int(real["NEWLY"].isna().sum()), 100)
        self.assertEqual(int(real["AAA"].isna().sum()), 0)
        self.assertEqual(json.loads(json.dumps(prices.attrs)), prices.attrs)  # still JSON-native

    def test_symbol_with_both_edge_gaps_records_start_and_cutoff(self):
        from ml_portfolio.data import eligibility_cutoffs, eligibility_starts
        prices, report = load_market_prices(self.root, ["AAA", "BRIEF"])
        self.assertIn("BRIEF", prices.columns)
        self.assertEqual(eligibility_starts(prices), {"BRIEF": self.dates[40]})
        # Trailing cutoff extends through the 5-row forward fill, as for STALE.
        self.assertEqual(eligibility_cutoffs(prices), {"BRIEF": self.dates[-36]})
        self.assertIn("BRIEF", report["listed_mid_window"])
        self.assertIn("BRIEF", report["truncated"])

    def test_interior_gap_still_excluded_not_silently_kept(self):
        # An INTERIOR gap (real data again AFTER a gap) is a materially
        # different, more ambiguous situation than a trailing one - keeps
        # today's behavior exactly: dropped, reported in `excluded`.
        root = Path(tempfile.mkdtemp())
        dates = pd.bdate_range("2024-01-01", periods=300)
        closes = np.linspace(100, 130, 300)
        closes[100:120] = np.nan  # a 20-day interior gap, well past the ffill(limit=5) reach
        _write_symbol_parquet(root, "AAA", dates, np.linspace(100, 130, 300))
        _write_symbol_parquet(root, "BBB", dates, np.linspace(100, 90, 300))
        _write_symbol_parquet(root, "GAPPY", dates, closes)
        _write_universe(root, ["AAA", "BBB", "GAPPY"])
        _write_manifest(root)
        prices, report = load_market_prices(root, ["AAA", "BBB", "GAPPY"])
        self.assertNotIn("GAPPY", prices.columns)
        self.assertIn("GAPPY", report["excluded"])
        self.assertNotIn("GAPPY", report["truncated"])

    def test_report_carries_price_basis_and_universe_caveat(self):
        _, report = load_market_prices(self.root, ["AAA", "BBB"])
        self.assertIn("adjusted", report["price_basis"])
        self.assertIn("survivorship", report["universe_caveat"])
        self.assertTrue(report["dataset_version"])

    def test_date_range_filters_before_gap_policy(self):
        prices, report = load_market_prices(self.root, ["AAA", "BBB"],
                                            start=self.dates[100], end=self.dates[200])
        self.assertEqual(len(prices), 101)

    def test_fingerprint_changes_with_selection_and_manifest(self):
        first = dataset_fingerprint(self.root, ["AAA", "BBB"])
        same = dataset_fingerprint(self.root, ["AAA", "BBB"])
        different_symbols = dataset_fingerprint(self.root, ["AAA"])
        self.assertEqual(first, same)
        self.assertNotEqual(first, different_symbols)

    def test_matches_equivalent_csv_load(self):
        parquet_prices, _ = load_market_prices(self.root, ["AAA", "BBB"])
        csv_text = "date,AAA,BBB\n" + "\n".join(
            f"{d.date()},{a},{b}" for d, a, b in zip(self.dates, np.linspace(100, 130, 300),
                                                     np.linspace(100, 90, 300)))
        csv_prices = load_price_csv(StringIO(csv_text))
        pd.testing.assert_frame_equal(
            parquet_prices.sort_index(axis=1), csv_prices.sort_index(axis=1),
            check_freq=False, check_names=False)

    def test_no_eligible_symbols_raises(self):
        with self.assertRaises(MarketDataError):
            load_market_prices(self.root, ["MISSING_ONE", "MISSING_TWO"])

    def test_corporate_actions_missing_files_return_empty_frame(self):
        actions = load_corporate_actions(self.root, ["AAA", "BBB"])
        self.assertTrue(actions.empty)
        self.assertIn("action_type", actions.columns)


@unittest.skipUnless(REAL_DATA_ROOT.exists(), "real data lake not present on this machine")
class RealDataSmokeCheck(unittest.TestCase):
    def test_loads_a_few_real_symbols(self):
        prices, report = load_market_prices(REAL_DATA_ROOT, ["RELIANCE", "TCS", "INFY"],
                                            start="2024-01-01")
        self.assertGreaterEqual(prices.shape[1], 2)
        self.assertTrue((prices > 0).all().all())
        self.assertIn("price_basis", report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
