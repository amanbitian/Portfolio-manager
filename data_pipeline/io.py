"""Parquet write/read helpers with atomic replace and primary-key de-duplication."""

from __future__ import annotations

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
