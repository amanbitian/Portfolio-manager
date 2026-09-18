"""Ingest NSE's own shareholding-pattern API across the universe.

New data type (promoter/public/employee-trust %, quarterly) - not present in the
Screener Excel export. No auth needed.

Worker threads here only hide per-request latency; the actual outbound request
rate is still capped by the shared "nse" rate limit in config.py (1 req/s),
enforced once per HttpClient regardless of thread count - the same pattern
already used for Common Crawl (workers <= 6) and screener.in (~1 req/4.5s)
elsewhere in this pipeline, because NSE's site blocks aggressively if you
actually hammer it concurrently.

    py -3 -m data_pipeline.run shareholding
    py -3 -m data_pipeline.run shareholding --workers 12 --limit 20
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from ..checkpoint import Manifest
from ..config import PipelineConfig
from ..http import HttpClient
from ..io import write_parquet
from ..sources import nse
from ..universe import load_universe

LOG = logging.getLogger("data_pipeline.ingest.shareholding")


def _fetch_one(client: HttpClient, symbol: str):
    return symbol, nse.fetch_shareholding(client, symbol)


def run(
    cfg: PipelineConfig,
    *,
    symbols: list[str] | None = None,
    limit: int | None = None,
    workers: int = 8,
) -> None:
    cfg.ensure_dirs()
    uni = load_universe(cfg)
    if symbols:
        uni = uni[uni["symbol"].isin({s.upper() for s in symbols})]
    if limit:
        uni = uni.head(limit)

    manifest = Manifest(cfg.manifest_dir / "shareholding.json")
    todo = [s for s in uni["symbol"].tolist() if not manifest.is_done(s)]
    LOG.info("shareholding: %d/%d symbols pending (%d workers)", len(todo), len(uni), workers)
    if not todo:
        return

    client = HttpClient(cfg)
    nse.prime(client)

    out_dir = cfg.shareholding_dir
    n_ok = n_failed = total_rows = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch_one, client, sym): sym for sym in todo}
        with tqdm(total=len(futures), desc="shareholding", unit="sym") as bar:
            for fut in as_completed(futures):
                sym = futures[fut]
                bar.update(1)
                try:
                    sym, df = fut.result()
                except Exception as exc:  # noqa: BLE001
                    LOG.warning("shareholding: %s failed: %s", sym, exc)
                    manifest.mark_failed(sym, str(exc))
                    n_failed += 1
                    continue
                if not df.empty:
                    df = df.assign(symbol=sym)
                    write_parquet(
                        df,
                        out_dir / f"{sym}.parquet",
                        primary_key=["symbol", "date"],
                        sort_by=["date"],
                    )
                    total_rows += len(df)
                manifest.mark_done(sym, len(df))
                n_ok += 1

    LOG.info(
        "shareholding done: %d ok, %d failed, %d rows -> %s",
        n_ok, n_failed, total_rows, out_dir,
    )
