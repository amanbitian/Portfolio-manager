"""Parquet write/read helpers with atomic replace and primary-key de-duplication."""

from __future__ import annotations

import contextlib
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Sequence

import pandas as pd

LOG = logging.getLogger("data_pipeline.io")


def atomic_replace(src: str, dst: str, *, attempts: int = 6) -> None:
    """os.replace with retry - Windows denies rename while AV/indexer/a reader holds a lock."""
    for i in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(0.5 * (i + 1))


@contextlib.contextmanager
def cross_process_lock(lock_path: Path, timeout: float = 60.0, poll: float = 0.05):
    """A portable mutex using exclusive file creation (`O_CREAT|O_EXCL` is
    atomic on both POSIX and Windows) - unlike an in-process `threading.Lock`,
    this also serializes separate OS processes (or unrelated objects in the
    same process) writing to the same path. `write_parquet`'s merge below and
    `checkpoint.Manifest` both need it: a read-modify-write critical section
    is not safe just because the final write step is atomic - two writers can
    each read the same starting state, merge their own update in memory, and
    have the second writer's atomic replace silently discard the first's.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except (FileExistsError, PermissionError):
            # On Windows, one thread/process deleting the lock file (the
            # `finally: os.unlink()` below) at the exact moment another tries
            # to create it can surface as PermissionError (WinError 5)
            # instead of FileExistsError - a transient filesystem race during
            # concurrent create/delete of the same name, not a real
            # permissions problem. Retried the same as a normal contention case.
            if time.monotonic() > deadline:
                raise TimeoutError(f"Timed out waiting for lock: {lock_path}")
            time.sleep(poll)
    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        yield
    finally:
        try:
            os.unlink(lock_path)
        except OSError:
            pass


def write_parquet(
    df: pd.DataFrame,
    path: Path,
    *,
    primary_key: Sequence[str] | None = None,
    sort_by: Sequence[str] | None = None,
    merge: bool = True,
) -> int:
    """Write ``df`` to ``path`` (snappy Parquet), atomically.

    If ``merge`` and the file already exists, the existing rows are read back and
    concatenated, then de-duplicated on ``primary_key`` (keeping the newest write).
    Returns the row count actually persisted.
    """
    if df is None or df.empty:
        return 0

    path.parent.mkdir(parents=True, exist_ok=True)

    # The read (existing rows) -> merge (concat/dedupe) -> write (atomic
    # replace) sequence below is only safe end-to-end under a lock: the final
    # `atomic_replace` alone just guarantees no reader ever sees a
    # half-written file, it does nothing to stop two concurrent writers from
    # both reading the same pre-write state and one silently clobbering the
    # other's merged rows.
    with cross_process_lock(path.with_name(path.name + ".lock")):
        out = df.copy()
        if merge and path.exists():
            try:
                existing = pd.read_parquet(path)
                out = pd.concat([existing, out], ignore_index=True)
            except (OSError, ValueError) as exc:  # corrupt / unreadable -> overwrite
                LOG.warning("could not merge into %s (%s); overwriting", path, exc)

        if primary_key:
            out = out.drop_duplicates(subset=list(primary_key), keep="last")
        if sort_by:
            out = out.sort_values(list(sort_by)).reset_index(drop=True)

        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".parquet.tmp")
        os.close(fd)
        try:
            out.to_parquet(tmp, engine="pyarrow", compression="snappy", index=False)
            atomic_replace(tmp, str(path))
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
        return len(out)


def read_parquet_dir(root: Path, **filters: object) -> pd.DataFrame:
    """Read every ``*.parquet`` under ``root`` into one frame (empty if none)."""
    files = sorted(root.rglob("*.parquet"))
    if not files:
        return pd.DataFrame()
    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    for col, val in filters.items():
        if col in df.columns:
            df = df[df[col] == val]
    return df.reset_index(drop=True)


def dir_size_mb(root: Path) -> float:
    if not root.exists():
        return 0.0
    total = sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
    return round(total / (1024 * 1024), 1)
