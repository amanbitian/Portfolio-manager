"""Macro ingest: World Bank (always) + FRED (if key) + RBI repo rate (bundled).

Writes macro/<source>/<series_id>.parquet  (series_id ':' and '/' -> '_' for filenames).
"""

from __future__ import annotations

import logging

from ..checkpoint import Manifest
from ..config import PipelineConfig
from ..http import HttpClient
from ..io import write_parquet
from ..sources import fred, worldbank
from ..sources.rbi import repo_rate_history

LOG = logging.getLogger("data_pipeline.ingest.macro")


def _safe_name(series_id: str) -> str:
    return series_id.replace(":", "_").replace("/", "_")


def run(cfg: PipelineConfig, *, force: bool = False, **_: object) -> None:
    cfg.ensure_dirs()
    client = HttpClient(cfg)
    manifest = Manifest(cfg.manifest_dir / "macro.json")

    # --- World Bank ---
    LOG.info("macro: World Bank (%d series)", len(worldbank.SERIES))
    for country, code, label in worldbank.SERIES:
        key = f"worldbank:{country}:{code}"
        if not force and manifest.is_done(key):
            continue
        try:
            df = worldbank.fetch_series(client, country, code, label)
            rows = write_parquet(
                df,
                cfg.macro_dir / "worldbank" / f"{country}_{_safe_name(code)}.parquet",
                primary_key=["series_id", "date"],
                sort_by=["date"],
                merge=False,
            )
            manifest.mark_done(key, rows)
            LOG.info("macro %s: %d rows", key, rows)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("macro %s failed: %s", key, exc)
            manifest.mark_failed(key, str(exc))

    # --- FRED ---
    if cfg.fred_api_key:
        for series_id, label in fred.SERIES.items():
            key = f"fred:{series_id}"
            if not force and manifest.is_done(key):
                continue
            try:
                df = fred.fetch_series(client, cfg.fred_api_key, series_id, label)
                rows = write_parquet(
                    df,
                    cfg.macro_dir / "fred" / f"{series_id}.parquet",
                    primary_key=["series_id", "date"],
                    sort_by=["date"],
                    merge=False,
                )
                manifest.mark_done(key, rows)
            except Exception as exc:  # noqa: BLE001
                LOG.warning("macro %s failed: %s", key, exc)
                manifest.mark_failed(key, str(exc))
    else:
        LOG.warning("macro: FRED_API_KEY not set - skipping FRED (global rates/CPI/oil/VIX)")

    # --- RBI repo rate (bundled) ---
    try:
        df = repo_rate_history()
        rows = write_parquet(
            df,
            cfg.macro_dir / "rbi" / "REPO_RATE.parquet",
            primary_key=["series_id", "date"],
            sort_by=["date"],
            merge=False,
        )
        manifest.mark_done("rbi:REPO_RATE", rows)
    except Exception as exc:  # noqa: BLE001
        LOG.warning("macro rbi:REPO_RATE failed: %s", exc)
        manifest.mark_failed("rbi:REPO_RATE", str(exc))

    s = manifest.summary()
    LOG.info("macro done: %d ok, %d failed, %d rows", s["done"], s["failed"], s["rows"])
