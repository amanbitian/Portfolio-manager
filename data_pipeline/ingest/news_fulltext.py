"""Full-text news from publisher monthly sitemaps.

Reads a publisher's monthly sitemap for YYYY-MM, fetches each article page, extracts
the body, tags universe symbols, and writes:

    news/fulltext/<source>/year=YYYY/part-YYYY-MM.parquet

    py -3 -m data_pipeline.run news-fulltext --source et --month 2016-06 --limit 50

Resumable: articles already stored with completeness != "failed" are skipped; failed
rows are retried and replaced on the next run (primary key article_id).
"""

from __future__ import annotations

import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ..config import PipelineConfig
from ..entities import load_matcher
from ..http import HttpClient
from ..io import write_parquet
from ..sources import sitemap_news as sm

LOG = logging.getLogger("data_pipeline.ingest.news_fulltext")

SOURCE_NAMES = {"et": "Economic Times"}
FULL_MIN_CHARS = 300      # body at least this long -> "full"
CHUNK = 200               # write to disk every CHUNK articles so progress survives a crash


def _article_id(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]


def _out_path(cfg: PipelineConfig, source: str, year: int, month: int) -> Path:
    return cfg.news_dir / "fulltext" / source / f"year={year}" / f"part-{year}-{month:02d}.parquet"


def _done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    df = pd.read_parquet(path, columns=["article_id", "completeness"])
    return set(df.loc[df["completeness"] != "failed", "article_id"])


def _parse_date(value: str | None) -> pd.Timestamp | None:
    if not value:
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    return None if pd.isna(ts) else ts


def _row(source: str, url: str, lastmod: datetime | None, scraped_at: str, **fields) -> dict:
    """One output row, with the schema shared by every full-text source."""
    return {
        "article_id": _article_id(url),
        "url": url,
        "source_code": source,
        "source_name": SOURCE_NAMES.get(source, source),
        "published_at": fields.get("published_at"),
        "date_source": fields.get("date_source"),
        "headline": fields.get("headline"),
        "body": fields.get("body"),
        "author": fields.get("author", []),
        "language": fields.get("language"),
        "completeness": fields["completeness"],
        "failure_reason": fields.get("failure_reason"),
        "word_count": fields.get("word_count", 0),
        "symbols": fields.get("symbols", []),
        "scraped_at": scraped_at,
        "raw_html_path": None,
    }


def _authors(raw: str | None) -> list[str]:
    """Split a byline into names. Extractors often pick up headline or section text
    instead of a byline, so anything that doesn't look like a name is dropped."""
    if not raw:
        return []
    names = []
    for a in raw.replace(" and ", ",").split(","):
        a = a.strip()
        if not a or len(a) > 40 or len(a.split()) > 4:
            continue
        if a.upper().startswith("SECTIONS") or any(ch.isdigit() for ch in a):
            continue
        names.append(a)
    return names


def _process(item, source: str, client: HttpClient, matcher, host_key: str) -> dict:
    url, lastmod = item
    scraped_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    sitemap_date = pd.Timestamp(lastmod) if lastmod else None

    try:
        resp = client.get(url, host_key=host_key, expect_json=False)
    except Exception as exc:  # noqa: BLE001 - recorded as a failed row, not raised
        return _row(source, url, lastmod, scraped_at, published_at=sitemap_date,
                    date_source="sitemap_lastmod" if sitemap_date is not None else None,
                    completeness="failed", failure_reason=f"fetch: {type(exc).__name__}")
    if resp.status_code != 200:
        return _row(source, url, lastmod, scraped_at, published_at=sitemap_date,
                    date_source="sitemap_lastmod" if sitemap_date is not None else None,
                    completeness="failed", failure_reason=f"http {resp.status_code}")

    try:
        art = sm.extract(resp.text)
    except Exception as exc:  # noqa: BLE001
        art = None
        reason = f"extract: {type(exc).__name__}"
    else:
        reason = "no extractable text"
    if art is None:
        return _row(source, url, lastmod, scraped_at, published_at=sitemap_date,
                    date_source="sitemap_lastmod" if sitemap_date is not None else None,
                    completeness="failed", failure_reason=reason)

    body = art["body"]
    headline = art["title"] or None
    # Prefer the page's own publish date; fall back to the sitemap's lastmod.
    page_date = _parse_date(art["date"])
    if page_date is not None:
        published_at, date_source = page_date, "page_meta"
    else:
        published_at = sitemap_date
        date_source = "sitemap_lastmod" if sitemap_date is not None else None

    if len(body) >= FULL_MIN_CHARS:
        completeness = "full"
    elif body:
        completeness = "partial"
    elif headline:
        completeness = "headline_only"
    else:
        return _row(source, url, lastmod, scraped_at, published_at=published_at,
                    date_source=date_source, completeness="failed",
                    failure_reason="no headline or body")

    authors = _authors(art["author"])
    symbols = matcher.match(headline or "", body[:1500]) if body else matcher.match(headline or "")
    return _row(
        source, url, lastmod, scraped_at,
        published_at=published_at,
        date_source=date_source,
        headline=headline,
        body=body or None,
        author=authors,
        language=art["language"],
        completeness=completeness,
        word_count=len(body.split()) if body else 0,
        symbols=symbols,
    )


def run(
    cfg: PipelineConfig,
    *,
    source: str,
    month: str,
    limit: int | None = None,
    workers: int = 4,
) -> None:
    if source not in sm.SITEMAPS:
        raise ValueError(f"no sitemap configured for source {source!r}")
    year, mon = (int(x) for x in month.split("-"))
    cfg.ensure_dirs()
    path = _out_path(cfg, source, year, mon)
    host_key = f"{source}_news"

    client = HttpClient(cfg)
    idx = client.get(sm.SITEMAP_INDEXES[source], host_key=host_key, expect_json=False)
    if idx.status_code != 200:
        LOG.error("news-fulltext: sitemap index returned HTTP %s", idx.status_code)
        return
    parts = sm.month_part_urls(idx.text, source, year, mon)
    if not parts:
        LOG.warning("news-fulltext %s %s: no sitemap parts listed in the index", source, month)
        return

    entries: list[tuple[str, object]] = []
    seen: set[str] = set()
    for part in parts:
        resp = client.get(part, host_key=host_key, expect_json=False)
        if resp.status_code != 200:
            LOG.error("news-fulltext: %s returned HTTP %s", part, resp.status_code)
            return
        for loc, lastmod in sm.parse_urlset(resp.text):
            if "articleshow" in loc and loc not in seen:
                seen.add(loc)
                entries.append((loc, lastmod))
    LOG.info("news-fulltext %s %s: %d sitemap part(s)", source, month, len(parts))

    done = _done_ids(path)
    todo = [e for e in entries if _article_id(e[0]) not in done]
    already = len(entries) - len(todo)
    if limit is not None:
        todo = todo[:limit]
    LOG.info("news-fulltext %s %s: %d in sitemap, %d already stored, %d to fetch this run",
             source, month, len(entries), already, len(todo))
    if not todo:
        return

    matcher = load_matcher(cfg)
    counts = {"full": 0, "partial": 0, "headline_only": 0, "failed": 0}
    with ThreadPoolExecutor(max_workers=workers) as pool, tqdm(total=len(todo), desc=f"{source} {month}", unit="art") as bar:
        for start in range(0, len(todo), CHUNK):
            chunk = todo[start:start + CHUNK]
            rows = list(pool.map(lambda e: _process(e, source, client, matcher, host_key), chunk))
            for r in rows:
                counts[r["completeness"]] += 1
            write_parquet(pd.DataFrame(rows), path, primary_key=["article_id"])
            bar.update(len(chunk))

    LOG.info("news-fulltext %s %s: %s -> %s", source, month, counts, path)
