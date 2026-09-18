"""Ingest AMFI mutual fund NAV history via mfapi.in across the full scheme universe.

New data type - daily NAV history for ~18k Indian mutual fund scheme/plan
combinations. No auth needed. mfapi.in publishes no numeric rate limit (just
asks callers to cache, which this ingest does - one-time historical pull per
scheme, written to Parquet, resumable). Worker threads give latency-hiding
parallelism; actual outbound request rate is still capped by the shared
"mfapi" rate limit in config.py so a burst of workers can't turn into a
request storm against a small community-run free service.

    py -3 -m data_pipeline.run mutual-funds
    py -3 -m data_pipeline.run mutual-funds --workers 16 --limit 50
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from ..checkpoint import Manifest
from ..config import PipelineConfig
from ..http import HttpClient
from ..io import write_parquet
from ..sources import amfi

LOG = logging.getLogger("data_pipeline.ingest.mutual_funds")


def _fetch_one(client: HttpClient, scheme_code: int):
    return scheme_code, amfi.fetch_nav_history(client, scheme_code)


def run(
    cfg: PipelineConfig,
    *,
    limit: int | None = None,
    workers: int = 10,
    refresh_after_days: float | None = None,
) -> None:
    cfg.ensure_dirs()
    client = HttpClient(cfg)

    schemes = amfi.fetch_scheme_list(client)
    write_parquet(
        schemes,
        cfg.reference_dir / "mutual_fund_schemes.parquet",
        primary_key=["scheme_code"],
    )
    LOG.info("mutual-funds: %d schemes in AMFI/mfapi universe", len(schemes))

    codes = schemes["scheme_code"].tolist()
    if limit:
        codes = codes[:limit]

    manifest = Manifest(cfg.manifest_dir / "mutual_funds.json")
    # mfapi has no bounded-range query - it always returns full history to
    # today, so a scheme marked "done" once never picked up newly published
    # NAVs on a later run regardless of how stale it got. `refresh_after_days`
    # (None by default, preserving the original behavior exactly) lets a
    # caller opt into re-fetching schemes whose last successful pull is older
    # than that many days.
    todo = [c for c in codes if not manifest.is_done(str(c), max_age_days=refresh_after_days)]
    LOG.info("mutual-funds: %d/%d schemes pending (%d workers)", len(todo), len(codes), workers)
    if not todo:
        return

    out_dir = cfg.fundamentals_dir / "mutual_funds" / "nav"
    n_ok = n_failed = total_rows = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch_one, client, code): code for code in todo}
        with tqdm(total=len(futures), desc="mutual-funds", unit="scheme") as bar:
            for fut in as_completed(futures):
                code = futures[fut]
                bar.update(1)
                try:
                    code, df = fut.result()
                except Exception as exc:  # noqa: BLE001
                    LOG.warning("mutual-funds: %s failed: %s", code, exc)
                    manifest.mark_failed(str(code), str(exc))
                    n_failed += 1
                    continue
                if not df.empty:
                    write_parquet(
                        df,
                        out_dir / f"{code}.parquet",
                        primary_key=["scheme_code", "date"],
                        sort_by=["date"],
                    )
                    total_rows += len(df)
                manifest.mark_done(str(code), len(df))
                n_ok += 1

    LOG.info(
        "mutual-funds done: %d ok, %d failed, %d rows -> %s",
        n_ok, n_failed, total_rows, out_dir,
    )
