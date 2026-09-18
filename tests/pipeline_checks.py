from __future__ import annotations

import sys
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.append(str(ROOT))              # for `import data_pipeline...`, same as app.py's own path
sys.path.append(str(ROOT / ".app-deps"))

import numpy as np
import pandas as pd

from data_pipeline.checkpoint import Manifest
from data_pipeline.io import write_parquet


class ManifestChecks(unittest.TestCase):
    def test_is_done_ignores_status_without_max_age(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest = Manifest(Path(temp) / "m.json")
            manifest.mark_done("A", rows=10)
            self.assertTrue(manifest.is_done("A"))
            self.assertFalse(manifest.is_done("B"))

    def test_is_done_respects_max_age_days(self):
        # Regression test: mfapi always returns full history to "today" with
        # no bounded-range query, so a scheme marked done once used to never
        # refresh regardless of how stale it got. `max_age_days` lets a
        # caller opt into treating an old fetch as no-longer-done.
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "m.json"
            manifest = Manifest(path)
            manifest.mark_done("A", rows=10)
            self.assertTrue(manifest.is_done("A", max_age_days=1))
            # Backdate the recorded timestamp to simulate an old fetch.
            stale_ts = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat(timespec="seconds")
            manifest._data["A"]["ts"] = stale_ts
            manifest._flush()
            reloaded = Manifest(path)
            self.assertTrue(reloaded.is_done("A"))  # still done with no age limit
            self.assertFalse(reloaded.is_done("A", max_age_days=1))  # too stale under a 1-day limit

    def test_concurrent_writers_do_not_lose_updates(self):
        # Regression test: two Manifest instances (simulating two separate
        # ingest processes) pointed at the same file used to each load a
        # construction-time snapshot and overwrite each other's entries on
        # flush - the second writer's atomic replace would silently discard
        # the first writer's key. Simulated here with threads sharing one
        # file path (the cross-process file lock doesn't care whether the
        # contention is between threads or processes).
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "m.json"
            n = 30
            errors = []

            def worker(i):
                try:
                    Manifest(path).mark_done(f"key-{i}", rows=i)
                except Exception as exc:  # noqa: BLE001
                    errors.append(exc)

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            self.assertEqual(errors, [])
            final = Manifest(path)
            for i in range(n):
                self.assertTrue(final.is_done(f"key-{i}"), f"key-{i} was lost to a concurrent-write race")


class WriteParquetConcurrencyChecks(unittest.TestCase):
    def test_concurrent_merges_do_not_lose_rows(self):
        # Regression test: write_parquet's read-existing -> concat -> dedupe
        # -> atomic-replace sequence used to have no lock around the whole
        # span - two concurrent writers could each read the same starting
        # file, merge their own new row in memory, and the second replace
        # would discard the first writer's row entirely (atomic replace only
        # guarantees no reader sees a half-written file, not that concurrent
        # writers merge correctly).
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "data.parquet"
            write_parquet(pd.DataFrame({"id": [0], "value": [0.0]}), path, primary_key=["id"])
            n = 20
            errors = []

            def worker(i):
                try:
                    write_parquet(pd.DataFrame({"id": [i + 1], "value": [float(i)]}), path, primary_key=["id"])
                except Exception as exc:  # noqa: BLE001
                    errors.append(exc)

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            self.assertEqual(errors, [])
            final = pd.read_parquet(path)
            self.assertEqual(len(final), n + 1, "some concurrent writers' rows were lost")
            self.assertEqual(set(final["id"]), set(range(n + 1)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
