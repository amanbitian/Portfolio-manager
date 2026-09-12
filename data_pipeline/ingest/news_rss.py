"""Ingest Indian financial-media + BSE-notice RSS feeds.

Ongoing signal only (feeds are a rolling window). Run on a schedule; articles
accumulate. Writes:
    news/rss/date=<YYYY-MM-DD>/articles.parquet   - all feeds merged, de-duped on url
Columns: feed, category, published, title, summary, url, symbols (comma-joined),
n_symbols, ingested_at.

Re-running the same day is idempotent (merge + de-dup on url).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd

from ..config import PipelineConfig
from ..entities import load_matcher
from ..io import write_parquet
from ..sources import rss

LOG = logging.getLogger("data_pipeline.ingest.news_rss")


def run(cfg: PipelineConfig, **_: object) -> None:
    cfg.ensure_dirs()
    matcher = load_matcher(cfg)
    articles = rss.fetch_all()
    if not articles:
        LOG.warning("news-rss: no articles fetched")
        return

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    df = pd.DataFrame(articles)
    df["published"] = pd.to_datetime(df["published"], utc=True, errors="coerce")
    df["published"] = df["published"].fillna(pd.Timestamp.now(tz="UTC"))
    syms = [matcher.match(t, s) for t, s in zip(df["title"], df["summary"])]
    df["symbols"] = [",".join(x) for x in syms]
    df["n_symbols"] = [len(x) for x in syms]
    df["ingested_at"] = now
    df["day"] = df["published"].dt.strftime("%Y-%m-%d")

    written = 0
    for day, g in df.groupby("day"):
        # keep the first feed that carried each url
        g = g.drop(columns=["day"]).drop_duplicates("url", keep="first")
        written += write_parquet(
            g,
            cfg.news_dir / "rss" / f"date={day}" / "articles.parquet",
            primary_key=["url"],
            sort_by=["published"],
        )

    LOG.info(
        "news-rss: %d fetched, %d unique articles from %d feeds, %d matched to >=1 symbol (%d distinct)",
        len(df),
        df["url"].nunique(),
        df["feed"].nunique(),
        int((df.drop_duplicates("url")["n_symbols"] > 0).sum()),
        len({s for row in syms for s in row}),
    )
