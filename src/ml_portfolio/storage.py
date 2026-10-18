from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import warnings
from datetime import datetime
from pathlib import Path
from threading import Lock
from uuid import uuid4

import numpy as np
import pandas as pd

CACHE_VERSION = "features-ridge-v2"  # v2: features/targets mask edge placeholders (data.listed_only)
DEFAULT_CACHE = Path(__file__).resolve().parents[2] / ".portfolio-cache"
CACHE_BUDGET_BYTES = 2 * 1024**3
_CACHE_LOCKS = tuple(Lock() for _ in range(64))


def price_key(prices: pd.DataFrame, *settings) -> str:
    digest = hashlib.sha256(pd.util.hash_pandas_object(prices, index=True).values.tobytes())
    digest.update(repr((list(prices.columns), list(map(str, prices.dtypes)), settings)).encode())
    return digest.hexdigest()


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    """`to_parquet` with `.attrs` kept only when pandas can actually store it.

    pandas JSON-encodes a non-empty `.attrs` into the parquet schema metadata
    with no type guard, so any non-JSON value (a Timestamp, a numpy array -
    the factor risk model hangs `loadings`/`factor_variance`/`specific_variance`
    on a covariance frame, see models.py) raises TypeError from deep inside the
    writer. JSON-native attrs (data.py's `eligible_through` ISO strings and
    `fill_counts`) are written and come back verbatim from `pd.read_parquet`;
    anything else is dropped for the write only, saved and restored around
    the call so the caller's live in-memory object is never mutated.
    """
    original_attrs = frame.attrs
    try:
        json.dumps(original_attrs, allow_nan=False)
    except (TypeError, ValueError):
        frame.attrs = {}
    try:
        frame.to_parquet(path, compression="zstd")
    finally:
        frame.attrs = original_attrs


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        _write_parquet(frame, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def prune_cache(root: Path, budget: int = CACHE_BUDGET_BYTES) -> None:
    files = []
    for path in root.glob("*/*.parquet"):
        try:
            stat = path.stat()
            files.append((stat.st_mtime, stat.st_size, path))
        except FileNotFoundError:
            continue
    size = sum(item[1] for item in files)
    for _, length, path in sorted(files):
        if size <= budget:
            break
        path.unlink(missing_ok=True)
        size -= length


def cached_frame(root: Path | None, namespace: str, key: str, build) -> pd.DataFrame:
    if root is None:
        return build()
    # Coalesce concurrent session misses, including Windows readers holding the file open.
    lock = _CACHE_LOCKS[hash((str(Path(root).resolve()), namespace, key)) % len(_CACHE_LOCKS)]
    with lock:
        return _cached_frame(Path(root), namespace, key, build)


def _cached_frame(root: Path, namespace: str, key: str, build) -> pd.DataFrame:
    path = Path(root) / namespace / f"{key}.parquet"
    try:
        frame = pd.read_parquet(path)
        os.utime(path, None)
        return frame
    except (OSError, ValueError):
        pass
    frame = build()
    try:
        atomic_parquet(frame, path)
        prune_cache(Path(root))
    except (OSError, ValueError) as exc:
        warnings.warn(f"Persistent cache unavailable; using memory: {exc}", RuntimeWarning)
    return frame


def write_artifact(root: Path, key: str, payload: dict) -> Path:
    """Publish the manifest last so partially written tables are never visible."""
    root.mkdir(parents=True, exist_ok=True)
    folder = root / key
    staging = Path(tempfile.mkdtemp(prefix=".pending-", dir=root))
    tables = 0

    def encode(value):
        nonlocal tables
        if isinstance(value, (pd.Series, pd.DataFrame)):
            name = f"table-{tables}.parquet"
            tables += 1
            is_series = isinstance(value, pd.Series)
            frame = value.to_frame("value") if is_series else value
            # `frame IS value` for a DataFrame (not a copy), so the attrs
            # guard inside `_write_parquet` is what keeps the caller's live
            # object unmutated by saving it.
            _write_parquet(frame, staging / name)
            return {"$table": f"{key}/{name}", "series": is_series,
                    "name": encode(value.name) if is_series else None}
        if isinstance(value, dict):
            return {str(k): encode(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [encode(v) for v in value]
        if isinstance(value, (datetime, pd.Timestamp)):
            return {"$timestamp": value.isoformat()}
        if isinstance(value, np.generic):
            return encode(value.item())
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    temporary = root / f".{key}.tmp"
    backup = folder.with_name(f".{folder.name}.{uuid4().hex}.bak")
    replaced_existing = False
    try:
        encoded = encode(payload)
        temporary.write_text(json.dumps(encoded, allow_nan=False), encoding="utf-8")
        # `os.replace` can't atomically overwrite an existing non-empty
        # directory on Windows (raises "Access is denied") - only a problem
        # for a caller that reuses the same `key` across saves (e.g. a
        # single persistent account updated in place, unlike the
        # timestamp+uuid keys every other caller in this codebase mints
        # fresh each time). Move the old folder aside first so both
        # `os.replace` calls below are plain renames onto non-existing
        # paths, which is atomic on every platform.
        if folder.exists():
            os.replace(folder, backup)
            replaced_existing = True
        os.replace(staging, folder)
        os.replace(temporary, root / f"{key}.json")
    finally:
        if staging.exists() and staging.resolve().is_relative_to(root.resolve()):
            shutil.rmtree(staging)
        temporary.unlink(missing_ok=True)
        if replaced_existing and backup.exists():
            shutil.rmtree(backup)
    return root / f"{key}.json"


def read_table(root: Path, reference: dict, columns=None):
    path = (root / reference["$table"]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Artifact table is outside the experiment directory.")
    frame = pd.read_parquet(path, columns=columns)
    return frame.iloc[:, 0].rename(reference.get("name")) if reference.get("series") else frame


def load_artifact(path: Path) -> dict:
    def decode(value):
        if isinstance(value, dict):
            if "$table" in value:
                return read_table(path.parent, value)
            if "$timestamp" in value:
                return pd.Timestamp(value["$timestamp"])
            return {k: decode(v) for k, v in value.items()}
        if isinstance(value, list):
            return [decode(v) for v in value]
        return value

    return decode(json.loads(path.read_text(encoding="utf-8")))
