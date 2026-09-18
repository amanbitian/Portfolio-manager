"""Resumable per-dataset progress manifests.

Each long-running ingest keeps a JSON file under ``_manifests/<dataset>.json`` mapping
a work-unit key (e.g. ``"RELIANCE|2022-03"``) to its status. Re-running an ingest skips
keys already marked ``done``; ``failed`` keys are retried.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from .io import cross_process_lock

Status = Literal["done", "failed"]


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Manifest:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._lock_path = path.with_name(path.name + ".lock")
        self._data: dict[str, dict] = {}
        self._reload()

    def _reload(self) -> None:
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text("utf-8"))
            except (json.JSONDecodeError, OSError):
                pass  # keep whatever's already in memory rather than losing it to a transient read glitch

    # ---- queries -----------------------------------------------------
    def is_done(self, key: str, max_age_days: float | None = None) -> bool:
        entry = self._data.get(key, {})
        if entry.get("status") != "done":
            return False
        if max_age_days is None:
            return True
        ts = entry.get("ts")
        try:
            fetched_at = datetime.fromisoformat(ts) if ts else None
        except ValueError:
            fetched_at = None
        if fetched_at is None:
            return False
        age_days = (datetime.now(timezone.utc) - fetched_at).total_seconds() / 86400
        return age_days <= max_age_days

    def summary(self) -> dict[str, int]:
        done = sum(1 for v in self._data.values() if v.get("status") == "done")
        failed = sum(1 for v in self._data.values() if v.get("status") == "failed")
        rows = sum(int(v.get("rows", 0)) for v in self._data.values())
        return {"done": done, "failed": failed, "rows": rows}

    def failed_keys(self) -> list[str]:
        return [k for k, v in self._data.items() if v.get("status") == "failed"]

    # ---- mutations --------------------------------------------------
    def mark_done(self, key: str, rows: int = 0) -> None:
        self._update(key, {"status": "done", "rows": int(rows), "ts": _utcnow()})

    def mark_failed(self, key: str, error: str) -> None:
        self._update(key, {"status": "failed", "error": str(error)[:500], "ts": _utcnow()})

    def _update(self, key: str, entry: dict) -> None:
        # `threading.Lock` serializes threads within THIS process;
        # `cross_process_lock` serializes against every OTHER process (or
        # Manifest instance) pointed at the same file. Reloading from disk
        # INSIDE both locks, right before merging, means this write is based
        # on the current on-disk state, not a construction-time snapshot that
        # another writer may have already moved past - closing the
        # read-stale-then-overwrite race a prior fix attempt would still have.
        with self._lock, cross_process_lock(self._lock_path):
            self._reload()
            self._data[key] = entry
            self._flush()

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(self._data, fh, indent=0, sort_keys=True)
            for i in range(6):
                try:
                    os.replace(tmp, self.path)
                    break
                except PermissionError:
                    if i == 5:
                        raise
                    time.sleep(0.5 * (i + 1))
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
