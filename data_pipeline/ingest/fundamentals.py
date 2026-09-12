"""Fundamentals ingest (yfinance).

Writes:
    fundamentals/income_stmt/<SYM>.parquet
    fundamentals/balance_sheet/<SYM>.parquet
    fundamentals/cash_flow/<SYM>.parquet
    fundamentals/key_stats/<SYM>.parquet
"""

from __future__ import annotations

import logging

from tqdm import tqdm

from ..checkpoint import Manifest
from ..config import PipelineConfig
from ..io import write_parquet
from ..sources.fundamentals import fetch_statements
from ..universe import load_universe

LOG = logging.getLogger("data_pipeline.ingest.fundamentals")


def run(
    cfg: PipelineConfig,
    *,
    symbols: list[str] | None = None,
    limit: int | None = None,
) -> None:
    cfg.ensure_dirs()
    uni = load_universe(cfg)
    if symbols:
        uni = uni[uni["symbol"].isin({s.upper() for s in symbols})]
    if limit:
        uni = uni.head(limit)

    manifest = Manifest(cfg.manifest_dir / "fundamentals.json")
    todo = [r for _, r in uni.iterrows() if not manifest.is_done(r["symbol"])]
    LOG.info("fundamentals: %d symbols pending", len(todo))

    for row in tqdm(todo, desc="fundamentals", unit="sym"):
        sym = row["symbol"]
        try:
            frames = fetch_statements(row["yf_ticker"], sym)
            total = 0
            for name, df in frames.items():
                pk = (
                    ["symbol", "key"]
                    if name == "key_stats"
                    else ["symbol", "period_end", "freq", "statement", "line_item"]
                )
                total += write_parquet(
                    df,
                    cfg.fundamentals_dir / name / f"{sym}.parquet",
                    primary_key=pk,
                    merge=False,
                )
            manifest.mark_done(sym, total)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("fundamentals %s failed: %s", sym, exc)
            manifest.mark_failed(sym, str(exc))

    s = manifest.summary()
    LOG.info("fundamentals done: %d ok, %d failed", s["done"], s["failed"])
