"""Augment corporate_actions/<SYM>.parquet with bonus/split events Screener's balance
sheet reveals but yfinance missed. Run after both `corp-actions` and `screener`.

    py -3 -m data_pipeline.run corp-actions-screener
"""

from __future__ import annotations

import logging

import pandas as pd
from tqdm import tqdm

from ..checkpoint import Manifest
from ..config import PipelineConfig
from ..io import write_parquet
from ..sources.screener_actions import derive_actions, merge_actions
from ..universe import load_universe

LOG = logging.getLogger("data_pipeline.ingest.corp_actions_screener")


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

    manifest = Manifest(cfg.manifest_dir / "corp_actions_screener.json")
    todo = [r for _, r in uni.iterrows() if not manifest.is_done(r["symbol"])]
    LOG.info("corp-actions-screener: %d symbols pending", len(todo))

    added_total = 0
    for row in tqdm(todo, desc="corp actions (screener)", unit="sym"):
        sym = row["symbol"]
        bs_path = cfg.screener_parquet_dir / "balance_sheet" / f"{sym}.parquet"
        if not bs_path.exists():
            manifest.mark_done(sym, 0)
            continue
        try:
            derived = derive_actions(pd.read_parquet(bs_path), sym)
            ca_path = cfg.corp_actions_dir / f"{sym}.parquet"
            existing = pd.read_parquet(ca_path) if ca_path.exists() else pd.DataFrame()
            n_before = len(existing)
            combined = merge_actions(existing, derived)
            n_new = max(len(combined) - n_before, 0)
            if not combined.empty:
                write_parquet(
                    combined,
                    ca_path,
                    primary_key=["symbol", "ex_date", "action_type"],
                    sort_by=["ex_date"],
                    merge=False,
                )
            added_total += n_new
            manifest.mark_done(sym, n_new)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("corp-actions-screener %s failed: %s", sym, exc)
            manifest.mark_failed(sym, str(exc))

    s = manifest.summary()
    LOG.info(
        "corp-actions-screener done: %d ok, %d failed, %d new actions added",
        s["done"], s["failed"], added_total,
    )
