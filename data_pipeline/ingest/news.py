"""News ingest via GDELT DOC 2.0.

Writes news/gdelt/symbol=<SYM>/<YYYY-MM>.parquet. Work unit: "<SYM>|<YYYY-MM>".

Default window is the last ``cfg.news_backfill_years`` years (GDELT coverage starts
~2017). At ~1 request / 5s this is a long job for the full universe - it is fully
resumable, so run it with ``--limit`` first, then let the rest run in the background.
"""

from __future__ import annotations

import logging
from datetime import date

from tqdm import tqdm

from ..checkpoint import Manifest
from ..config import PipelineConfig
from ..http import HttpClient
from ..io import write_parquet
from ..sources.gdelt import GDELT_MIN_DATE, fetch_month
from ..universe import load_universe


LOG = logging.getLogger("data_pipeline.ingest.news")


def _months(start: date, end: date) -> list[tuple[int, int]]:
    out = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        out.append((y, m))
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return out


def run(
    cfg: PipelineConfig,
    *,
    symbols: list[str] | None = None,
    limit: int | None = None,
    years: int | None = None,
) -> None:
    cfg.ensure_dirs()
    uni = load_universe(cfg)
    if symbols:
        uni = uni[uni["symbol"].isin({s.upper() for s in symbols})]
    if limit:
        uni = uni.head(limit)

    back_years = years or cfg.news_backfill_years
    start = max(GDELT_MIN_DATE, date(cfg.end.year - back_years, cfg.end.month, 1))
    months = _months(start, cfg.end)

    manifest = Manifest(cfg.manifest_dir / "news_gdelt.json")
    client = HttpClient(cfg)

    pending = [
        (row, y, m)
        for _, row in uni.iterrows()
        for (y, m) in months
        if not manifest.is_done(f"{row['symbol']}|{y:04d}-{m:02d}")
    ]
    LOG.info(
        "news: %d symbols x %d months, %d chunks pending (~%.1f h at 5s/req)",
        len(uni),
        len(months),
        len(pending),
        len(pending) * 5 / 3600,
    )

    for row, y, m in tqdm(pending, desc="gdelt news", unit="chunk"):
        sym = row["symbol"]
        key = f"{sym}|{y:04d}-{m:02d}"
        try:
            company = row["name"] if isinstance(row["name"], str) and row["name"] else sym
            df = fetch_month(client, company, y, m)
            if not df.empty:
                df["symbol"] = sym
            rows = write_parquet(
                df,
                cfg.news_dir / "gdelt" / f"symbol={sym}" / f"{y:04d}-{m:02d}.parquet",
                primary_key=["symbol", "url"],
                sort_by=["seendate"],
                merge=False,
            )
            manifest.mark_done(key, rows)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("news %s failed: %s", key, exc)
            manifest.mark_failed(key, str(exc))

    s = manifest.summary()
    LOG.info("news done: %d ok, %d failed, %d articles", s["done"], s["failed"], s["rows"])
