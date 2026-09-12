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

Status = Literal["done", "failed"]


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Manifest:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._data: dict[str, dict] = {}
        if path.exists():
            try:
                self._data = json.loads(path.read_text("utf-8"))
            except (json.JSONDecodeError, OSError):
                self._data = {}

    # ---- queries -----------------------------------------------------
    def is_done(self, key: str) -> bool:
        return self._data.get(key, {}).get("status") == "done"

    def summary(self) -> dict[str, int]:
        done = sum(1 for v in self._data.values() if v.get("status") == "done")
        failed = sum(1 for v in self._data.values() if v.get("status") == "failed")
        rows = sum(int(v.get("rows", 0)) for v in self._data.values())
        return {"done": done, "failed": failed, "rows": rows}

    def failed_keys(self) -> list[str]:
        return [k for k, v in self._data.items() if v.get("status") == "failed"]

    # ---- mutations --------------------------------------------------
    def mark_done(self, key: str, rows: int = 0) -> None:
        with self._lock:
            self._data[key] = {"status": "done", "rows": int(rows), "ts": _utcnow()}
            self._flush()

    def mark_failed(self, key: str, error: str) -> None:
        with self._lock:
            self._data[key] = {
                "status": "failed",
                "error": str(error)[:500],
                "ts": _utcnow(),
            }
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
