"""Ingest screener.in Excel exports from the Screener data folder.

Drop per-company `.xlsx` files (Screener's "Export to Excel") into
``F:\\quants project\\Screener data`` (override via ``SCREENER_DATA_ROOT``). Name each
file by its NSE symbol (``RELIANCE.xlsx``) or add a mapping row to
``<folder>\\_symbol_map.csv`` with columns ``file,symbol``.

Writes long-form Parquet under ``fundamentals/screener/<section>/<SYM>.parquet`` with
columns ``symbol, period_end, freq, statement, line_item, value, source, retrieved_at``
- the same shape as the yfinance fundamentals, so the sanity checker can compare them.

Work unit (manifest key): ``<filename>|<mtime>`` so edited files re-ingest automatically.
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime, timezone

import pandas as pd

from ..checkpoint import Manifest
from ..config import PipelineConfig
from ..io import write_parquet
from ..sources.screener import parse_workbook, symbol_from_path

LOG = logging.getLogger("data_pipeline.ingest.screener")

# section -> (statement label, frequency)
_SECTION_META = {
    "pnl": ("income_stmt", "A"),
    "quarters": ("income_stmt", "Q"),
    "balance_sheet": ("balance_sheet", "A"),
    "cash_flow": ("cash_flow", "A"),
    "price": ("price", "M"),
    "derived": ("derived", "A"),
}


def _symbol_map(cfg: PipelineConfig) -> dict[str, str]:
    path = cfg.screener_root / "_symbol_map.csv"
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("file") and row.get("symbol"):
                out[row["file"].strip().lower()] = row["symbol"].strip().upper()
    return out


def run(
    cfg: PipelineConfig,
    *,
    symbols: list[str] | None = None,
    limit: int | None = None,
) -> None:
    cfg.ensure_dirs()
    if not cfg.screener_root.exists():
        LOG.warning("screener: folder %s does not exist", cfg.screener_root)
        return

    files = sorted(
        p for p in cfg.screener_root.glob("*.xlsx") if not p.name.startswith("~$")
    )
    if not files:
        LOG.warning("screener: no .xlsx files in %s", cfg.screener_root)
        return

    smap = _symbol_map(cfg)
    manifest = Manifest(cfg.manifest_dir / "screener.json")
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    want = {s.upper() for s in symbols} if symbols else None
    processed = 0
    for path in files:
        sym = smap.get(path.name.lower()) or symbol_from_path(path)
        if want and sym not in want:
            continue
        if limit and processed >= limit:
            break
        key = f"{path.name}|{int(path.stat().st_mtime)}"
        if manifest.is_done(key):
            continue

        try:
            sections = parse_workbook(path)
            total = 0
            for section, df in sections.items():
                statement, freq = _SECTION_META.get(section, (section, "A"))
                out = df.copy()
                out["symbol"] = sym
                out["freq"] = freq
                out["statement"] = statement
                out["source"] = "screener"
                out["retrieved_at"] = now
                out = out[
                    ["symbol", "period_end", "freq", "statement", "line_item",
                     "value", "source", "retrieved_at"]
                ]
                total += write_parquet(
                    out,
                    cfg.screener_parquet_dir / section / f"{sym}.parquet",
                    primary_key=["symbol", "period_end", "freq", "statement", "line_item"],
                    sort_by=["statement", "period_end", "line_item"],
                    merge=False,
                )
            manifest.mark_done(key, total)
            processed += 1
            LOG.info("screener: %s <- %s (%d rows, sections: %s)",
                     sym, path.name, total, ", ".join(sorted(sections)))
        except Exception as exc:  # noqa: BLE001
            LOG.warning("screener: %s failed: %s", path.name, exc)
            manifest.mark_failed(key, str(exc))

    s = manifest.summary()
    LOG.info("screener done: %d files ok, %d failed", s["done"], s["failed"])
