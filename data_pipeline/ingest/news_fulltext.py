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
import itertools
import logging
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ..config import (
    FULLTEXT_CHUNK,
    FULLTEXT_EXTRACT_PROCESSES,
    FULLTEXT_FULL_MIN_CHARS,
    FULLTEXT_PROGRESS_EVERY,
    FULLTEXT_SAVE_EVERY,
    FULLTEXT_STALL_SECONDS,
    FULLTEXT_THROTTLE_LIMIT,
    FULLTEXT_USER_AGENT,
    PipelineConfig,
    RateLimit,
)
from ..entities import load_matcher
from ..http import HttpClient
from ..io import write_parquet
from ..sources import sitemap_news as sm

LOG = logging.getLogger("data_pipeline.ingest.news_fulltext")

SOURCE_NAMES = {"et": "Economic Times"}


_HEADERS = {"User-Agent": FULLTEXT_USER_AGENT}


class ThrottleBrake(RuntimeError):
    """The publisher is pushing back (429 / repeated 5xx). The run stops after saving
    the batch it was on, so the rate can be lowered before anything else is requested."""


class BlockedBrake(ThrottleBrake):
    """HTTP 403: the publisher refused this connection. Rows are saved first; the caller
    pauses and retries (FULLTEXT_BLOCK_PAUSE_SECONDS) instead of stopping outright."""


class StallError(RuntimeError):
    """Nothing finished for FULLTEXT_STALL_SECONDS: a request or an extraction is stuck."""


# A retry cannot change these, so a failed row with one of them counts as done.
_PERMANENT_FAILURES = ("http 404", "http 410", "redirect loop")


def _stream(pool, fn, items, window: int, stall_seconds: float = FULLTEXT_STALL_SECONDS):
    """Yield fn(item) for each item as soon as it finishes, keeping at most `window` in flight.
    Unlike pool.map over a batch, one slow item never holds up the rest. Raises StallError if
    nothing at all finishes for `stall_seconds`, so a hang cannot go unnoticed."""
    from concurrent.futures import FIRST_COMPLETED, wait

    it = iter(items)
    inflight = set()
    for item in itertools.islice(it, window):
        inflight.add(pool.submit(fn, item))
    while inflight:
        done, inflight = wait(inflight, timeout=stall_seconds, return_when=FIRST_COMPLETED)
        if not done:
            raise StallError(f"no article finished in {stall_seconds:.0f} s with {len(inflight)} in flight")
        for fut in done:
            yield fut.result()
            nxt = next(it, None)
            if nxt is not None:
                inflight.add(pool.submit(fn, nxt))


def _is_throttled(reason: str | None) -> bool:
    return bool(reason) and (reason.startswith("throttled") or reason.startswith("http 429")
                             or reason.startswith("http 5") or reason.startswith("blocked"))


def _article_id(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]


def _out_path(cfg: PipelineConfig, source: str, year: int, month: int) -> Path:
    return cfg.news_dir / "fulltext" / source / f"year={year}" / f"part-{year}-{month:02d}.parquet"


def _done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    df = pd.read_parquet(path, columns=["article_id", "completeness", "failure_reason"])
    permanent = df["failure_reason"].fillna("").str.startswith(_PERMANENT_FAILURES)
    return set(df.loc[(df["completeness"] != "failed") | permanent, "article_id"])


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


def _process(item, source: str, client: HttpClient, matcher, host_key: str, extract_fn) -> dict:
    url, lastmod = item
    scraped_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    sitemap_date = pd.Timestamp(lastmod) if lastmod else None

    try:
        resp = client.get(url, host_key=host_key, expect_json=False, headers=_HEADERS)
    except Exception as exc:  # noqa: BLE001 - recorded as a failed row, not raised
        # RetryableHTTPError is raised for 429 / 5xx after the client's own retries
        reason = "throttled" if type(exc).__name__ == "RetryableHTTPError" else f"fetch: {type(exc).__name__}"
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status == 403:
            reason = "blocked 403"  # publisher refused this connection: the brake must trip
        elif status in (404, 410):
            reason = f"http {status}"   # page removed: permanent, not retried on later runs
        elif type(exc).__name__ == "TooManyRedirects":
            reason = "redirect loop"    # permanent for this page; says nothing about rate limits
        return _row(source, url, lastmod, scraped_at, published_at=sitemap_date,
                    date_source="sitemap_lastmod" if sitemap_date is not None else None,
                    completeness="failed", failure_reason=reason)
    if resp.status_code != 200:
        return _row(source, url, lastmod, scraped_at, published_at=sitemap_date,
                    date_source="sitemap_lastmod" if sitemap_date is not None else None,
                    completeness="failed", failure_reason=f"http {resp.status_code}")

    try:
        art = extract_fn(resp.text)
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

    if len(body) >= FULLTEXT_FULL_MIN_CHARS:
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
    rps: float | None = None,
) -> None:
    if source not in sm.SITEMAPS:
        raise ValueError(f"no sitemap configured for source {source!r}")
    year, mon = (int(x) for x in month.split("-"))
    cfg.ensure_dirs()
    path = _out_path(cfg, source, year, mon)
    host_key = f"{source}_news"
    if rps is not None:
        cfg.rate_limits[host_key] = RateLimit(rps=rps)
    LOG.info("news-fulltext %s %s: rate %.1f req/s, %d workers", source, month,
             cfg.rate_limits[host_key].rps, workers)

    client = HttpClient(cfg)
    idx = client.get(sm.SITEMAP_INDEXES[source], host_key=host_key, expect_json=False, headers=_HEADERS)
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
        resp = client.get(part, host_key=host_key, expect_json=False, headers=_HEADERS)
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
    # Extraction is CPU work; Python threads share one core for it (the GIL). Threads
    # wait on the process pool instead, so extraction runs on several cores at once.
    with ProcessPoolExecutor(max_workers=FULLTEXT_EXTRACT_PROCESSES) as extract_pool, \
            ThreadPoolExecutor(max_workers=workers) as pool, \
            tqdm(total=len(todo), desc=f"{source} {month}", unit="art") as bar:
        def extract_fn(html: str):
            return extract_pool.submit(sm.extract, html).result()

        pending: list[dict] = []

        def flush() -> None:
            # Every write merges into and rewrites the whole month file, so it is batched
            # (FULLTEXT_SAVE_EVERY) instead of done after every small fetch batch.
            if pending:
                write_parquet(pd.DataFrame(pending), path, primary_key=["article_id"])
                pending.clear()

        def check_brake(rows: list[dict]) -> None:
            # a 403 always stops the run; throttling stops it only on a large enough sample
            if any((r["failure_reason"] or "").startswith("blocked") for r in rows):
                raise BlockedBrake(f"403 at {month} (rate {cfg.rate_limits[host_key].rps} req/s). Saved so far.")
            throttled = sum(_is_throttled(r["failure_reason"]) for r in rows)
            if len(rows) >= 50 and throttled and (throttled / len(rows) > FULLTEXT_THROTTLE_LIMIT
                                                  or any(r["failure_reason"] == "http 429" for r in rows)):
                raise ThrottleBrake(
                    f"{throttled}/{len(rows)} articles throttled at {month} "
                    f"(rate {cfg.rate_limits[host_key].rps} req/s). Saved so far; "
                    f"re-run at a lower --rps to resume."
                )

        try:
            batch: list[dict] = []
            for row in _stream(pool, lambda e: _process(e, source, client, matcher, host_key, extract_fn),
                               todo, window=workers * 4):
                counts[row["completeness"]] += 1
                pending.append(row)
                batch.append(row)
                bar.update(1)
                if bar.n % FULLTEXT_PROGRESS_EVERY == 0:
                    LOG.info("news-fulltext %s %s: %d/%d fetched %s", source, month, bar.n, len(todo), counts)
                if len(pending) >= FULLTEXT_SAVE_EVERY:
                    flush()
                if len(batch) >= FULLTEXT_CHUNK:
                    check_brake(batch)
                    batch = []
            if batch:
                check_brake(batch)
        except StallError as exc:
            flush()
            LOG.error("news-fulltext %s %s: STALLED, %s. Saved %d rows; exiting with code 4 so the "
                      "driver retries the month.", source, month, exc, sum(counts.values()))
            logging.shutdown()
            os._exit(4)   # a stuck worker would block the normal shutdown, so leave immediately
        finally:
            flush()   # also runs when the brake (or an error) stops the run, so nothing fetched is lost

    LOG.info("news-fulltext %s %s: %s -> %s", source, month, counts, path)
