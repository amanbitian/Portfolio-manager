"""Corporate-actions ingest (yfinance). Writes corporate_actions/<SYM>.parquet."""

from __future__ import annotations

import logging

from tqdm import tqdm

from ..checkpoint import Manifest
from ..config import PipelineConfig
from ..io import write_parquet
from ..sources.corporate_actions import fetch_actions
from ..universe import load_universe

LOG = logging.getLogger("data_pipeline.ingest.corp_actions")


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

    manifest = Manifest(cfg.manifest_dir / "corporate_actions.json")
    todo = [r for _, r in uni.iterrows() if not manifest.is_done(r["symbol"])]
    LOG.info("corp-actions: %d symbols pending", len(todo))

    for row in tqdm(todo, desc="corp actions", unit="sym"):
        sym = row["symbol"]
        try:
            df = fetch_actions(row["yf_ticker"], sym)
            rows = write_parquet(
                df,
                cfg.corp_actions_dir / f"{sym}.parquet",
                primary_key=["symbol", "ex_date", "action_type"],
                sort_by=["ex_date"],
                merge=False,
            )
            manifest.mark_done(sym, rows)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("corp-actions %s failed: %s", sym, exc)
            manifest.mark_failed(sym, str(exc))

    s = manifest.summary()
    LOG.info("corp-actions done: %d ok, %d failed", s["done"], s["failed"])
