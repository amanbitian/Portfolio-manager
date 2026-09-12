"""Market OHLCV ingest via Upstox historical candles.

Layout written:
    market/ohlcv_1day/symbol=<SYM>/part.parquet
    market/ohlcv_1min/symbol=<SYM>/year=YYYY/month=MM/part.parquet

Work unit (manifest key): "<SYM>|<chunk.key>". Re-runs skip completed chunks, so the
multi-hour 1-minute backfill is safe to stop and resume.
"""

from __future__ import annotations

import logging

import pandas as pd
from tqdm import tqdm

from ..checkpoint import Manifest
from ..config import PipelineConfig
from ..http import HttpClient
from ..io import write_parquet
from ..sources.upstox import UpstoxClient, chunks_for
from ..universe import load_universe

LOG = logging.getLogger("data_pipeline.ingest.market")

_SPECS = {
    "1day": {"unit": "days", "interval": 1},
    "1min": {"unit": "minutes", "interval": 1},
}


def _partition_path(cfg: PipelineConfig, interval: str, symbol: str, row: pd.Series):
    base = cfg.ohlcv_dir(interval) / f"symbol={symbol}"
    if interval == "1min":
        ts = row["ts_ist"]
        return base / f"year={ts.year:04d}" / f"month={ts.month:02d}" / "part.parquet"
    return base / "part.parquet"


def _write_frame(cfg: PipelineConfig, interval: str, symbol: str, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    df = df.assign(symbol=symbol)
    written = 0
    if interval == "1min":
        df["_y"] = df["ts_ist"].dt.year
        df["_m"] = df["ts_ist"].dt.month
        for (y, m), g in df.groupby(["_y", "_m"], sort=True):
            path = (
                cfg.ohlcv_dir(interval)
                / f"symbol={symbol}"
                / f"year={int(y):04d}"
                / f"month={int(m):02d}"
                / "part.parquet"
            )
            written += write_parquet(
                g.drop(columns=["_y", "_m"]),
                path,
                primary_key=["ts_utc"],
                sort_by=["ts_utc"],
            )
    else:
        path = cfg.ohlcv_dir(interval) / f"symbol={symbol}" / "part.parquet"
        written = write_parquet(
            df, path, primary_key=["ts_utc"], sort_by=["ts_utc"]
        )
    return written


def run(
    cfg: PipelineConfig,
    interval: str,
    *,
    symbols: list[str] | None = None,
    limit: int | None = None,
) -> None:
    if interval not in _SPECS:
        raise ValueError(f"interval must be one of {list(_SPECS)}")
    spec = _SPECS[interval]
    cfg.ensure_dirs()

    uni = load_universe(cfg, only_priceable=True)
    if symbols:
        want = {s.upper() for s in symbols}
        uni = uni[uni["symbol"].isin(want)]
    if limit:
        uni = uni.head(limit)
    if uni.empty:
        LOG.warning("market: no priceable symbols selected")
        return

    start = cfg.daily_start if interval == "1day" else cfg.minute_start
    manifest = Manifest(cfg.manifest_dir / f"market_{interval}.json")
    client = UpstoxClient(HttpClient(cfg))

    pending = []
    for _, row in uni.iterrows():
        for ch in chunks_for(spec["unit"], spec["interval"], start, cfg.end):
            key = f"{row['symbol']}|{ch.key}"
            if not manifest.is_done(key):
                pending.append((row, ch, key))

    LOG.info(
        "market %s: %d symbols, %d chunks pending", interval, len(uni), len(pending)
    )
    for row, ch, key in tqdm(pending, desc=f"ohlcv {interval}", unit="chunk"):
        try:
            df = client.fetch_chunk(
                row["instrument_key"], spec["unit"], spec["interval"], ch
            )
            rows = _write_frame(cfg, interval, row["symbol"], df)
            manifest.mark_done(key, rows)
        except Exception as exc:  # noqa: BLE001 - keep going, record the failure
            LOG.warning("market %s: %s failed: %s", interval, key, exc)
            manifest.mark_failed(key, str(exc))

    s = manifest.summary()
    LOG.info(
        "market %s done: %d chunks ok, %d failed, %d rows",
        interval,
        s["done"],
        s["failed"],
        s["rows"],
    )
