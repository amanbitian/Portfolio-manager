"""Historical news ingest from Common Crawl (2017 -> present).

Phase 1 (catalog): query every CC-MAIN index for our news-domain URL patterns and
save the captures to news/commoncrawl/_catalog/<index>.parquet.

Phase 2 (fetch): range-GET + extract each unique article, entity-match to universe
symbols, write news/commoncrawl/year=YYYY/month=MM/part-*.parquet.

    py -3 -m data_pipeline.run news-cc --phase catalog
    py -3 -m data_pipeline.run news-cc --phase fetch --limit 5000
    py -3 -m data_pipeline.run news-cc --phase both
    py -3 -m data_pipeline.run news-cc --since 2020 --domains et,bs,mint

Both phases are resumable: catalog via the manifest, fetch via the URLs already
present in the output (and _failed.parquet).
"""

from __future__ import annotations

import hashlib
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pandas as pd
from tqdm import tqdm

from ..checkpoint import Manifest
from ..config import PipelineConfig
from ..entities import load_matcher
from ..io import write_parquet
from ..sources import commoncrawl as cc

LOG = logging.getLogger("data_pipeline.ingest.news_cc")

# domain-key -> list of URL patterns to query (path-prefixed to avoid whole-site pulls)
DOMAIN_PATTERNS: dict[str, list[str]] = {
    "et": [
        "economictimes.indiatimes.com/markets/*",
        "economictimes.indiatimes.com/industry/*",
        "economictimes.indiatimes.com/news/company/*",
    ],
    "bs": [
        "business-standard.com/markets/*",
        "business-standard.com/companies/*",
        "business-standard.com/article/markets/*",
        "business-standard.com/article/companies/*",
    ],
    "mint": [
        "livemint.com/market/*",
        "livemint.com/companies/*",
        "livemint.com/industry/*",
    ],
    "hbl": [
        "thehindubusinessline.com/markets/*",
        "thehindubusinessline.com/companies/*",
        "thehindubusinessline.com/money-and-banking/*",
    ],
    "mc": [
        "moneycontrol.com/news/business/*",
    ],
    "fe": [
        "financialexpress.com/market/*",
        "financialexpress.com/industry/*",
        "financialexpress.com/business/*",
    ],
    "bt": [
        "businesstoday.in/markets/*",
        "businesstoday.in/industry/*",
    ],
}

_CATALOG = "_catalog"
_FAILED = "_failed.parquet"


def _hash(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]


def _cc_dir(cfg: PipelineConfig):
    return cfg.news_dir / "commoncrawl"


# ---------------------------------------------------------------------------
def catalog(cfg: PipelineConfig, since_year: int, domains: list[str]) -> None:
    s = cc.session()
    indexes = cc.list_indexes(s, since_year=since_year)
    patterns = [(d, p) for d in domains for p in DOMAIN_PATTERNS.get(d, [])]
    manifest = Manifest(cfg.manifest_dir / "news_cc_catalog.json")
    cat_dir = _cc_dir(cfg) / _CATALOG
    cat_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("news-cc catalog: %d indexes x %d patterns", len(indexes), len(patterns))
    consec_fail = 0
    for index in indexes:
        rows: list[dict] = []
        # (key, row count) pending for this index - only marked done in the manifest
        # once `rows` has actually been persisted, so a crash/failed write between
        # querying and writing can't leave a pattern marked done with no data on disk.
        pending_marks: list[tuple[str, int]] = []

        def _persist() -> None:
            if rows:
                write_parquet(
                    pd.DataFrame(rows).drop_duplicates("url"),
                    cat_dir / f"{index}.parquet", primary_key=["url"],
                )
            for key, n in pending_marks:
                manifest.mark_done(key, n)
            pending_marks.clear()

        for dkey, pat in patterns:
            key = f"{index}|{pat}"
            if manifest.is_done(key):
                continue
            try:
                got = list(cc.query_index(s, index, pat))
                consec_fail = 0
            except cc.CdxUnavailable as exc:
                manifest.mark_failed(key, str(exc))
                consec_fail += 1
                if consec_fail >= 20:
                    LOG.warning(
                        "news-cc catalog: CDX index server unreachable (20 in a row) - "
                        "stopping. Re-run later to resume."
                    )
                    _persist()
                    s.close()
                    return
                continue
            for r in got:
                r["domain"] = dkey
            rows.extend(got)
            pending_marks.append((key, len(got)))
            LOG.info("  %s  %-48s %d", index, pat, len(got))
        _persist()

    s.close()
    LOG.info("news-cc catalog done: %s", manifest.summary())


# ---------------------------------------------------------------------------
def _done_urls(cfg: PipelineConfig) -> set[str]:
    done: set[str] = set()
    base = _cc_dir(cfg)
    for p in base.rglob("part-*.parquet"):
        try:
            done.update(pd.read_parquet(p, columns=["url"])["url"].tolist())
        except Exception:  # noqa: BLE001
            pass
    fp = base / _FAILED
    if fp.exists():
        try:
            done.update(pd.read_parquet(fp)["url"].tolist())
        except Exception:  # noqa: BLE001
            pass
    return done


_throttle = None  # threading.Event, set when CC is rate-limiting


def _process_one(rec: dict, sess) -> dict | None:
    """Range-GET + extract one capture. Runs on a worker thread."""
    if _throttle is not None and _throttle.is_set():
        import time as _t

        _t.sleep(15)
    try:
        html = cc.fetch_html(sess, rec)
    except cc.RateLimited:
        if _throttle is not None:
            _throttle.set()
        return {"url": rec["url"], "_retry": True}
    except cc.FetchTimeout:
        # transient network/timeout failure, not a permanent miss - retry on a
        # later run instead of recording it in _failed.parquet.
        return {"url": rec["url"], "_retry": True}
    if _throttle is not None:
        _throttle.clear()
    art = cc.extract_article(html) if html else None
    if not art:
        return {"url": rec["url"], "_failed": True}
    pub = pd.to_datetime(art["date"] or None, errors="coerce")
    if pd.isna(pub):
        pub = pd.to_datetime(rec["timestamp"], format="%Y%m%d%H%M%S", errors="coerce")
    return {
        "url": rec["url"],
        "domain": rec.get("domain"),
        "published": pub,
        "crawl_ts": rec["timestamp"],
        "title": art["title"],
        "excerpt": art["excerpt"],
        "text": art["text"],
        "sitename": art["sitename"],
    }


def fetch(cfg: PipelineConfig, limit: int | None, workers: int) -> None:
    base = _cc_dir(cfg)
    cat_files = sorted((base / _CATALOG).glob("*.parquet"))
    if not cat_files:
        LOG.warning("news-cc fetch: no catalog - run --phase catalog first")
        return

    cat = pd.concat([pd.read_parquet(f) for f in cat_files], ignore_index=True)
    cat = cat.sort_values("timestamp").drop_duplicates("url", keep="first")
    done = _done_urls(cfg)
    todo = cat[~cat["url"].isin(done)]
    if limit:
        todo = todo.head(limit)
    LOG.info(
        "news-cc fetch: %d unique catalogued, %d done, %d to fetch (%d workers)",
        len(cat), len(done), len(todo), workers,
    )
    if todo.empty:
        return

    global _throttle
    import threading

    _throttle = threading.Event()

    matcher = load_matcher(cfg)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    stamp = now[:16].replace(":", "").replace("-", "")
    sessions = [cc.session() for _ in range(workers)]
    buf: dict[str, list[dict]] = {}
    failed: list[dict] = []
    n_ok = n_retry = consec_retry = 0

    def flush() -> None:
        for ym, rows in list(buf.items()):
            y, m = ym.split("-")
            write_parquet(
                pd.DataFrame(rows),
                base / f"year={y}" / f"month={m}" / f"part-{stamp}.parquet",
                primary_key=["url"], sort_by=["published"],
            )
        buf.clear()
        if failed:
            write_parquet(pd.DataFrame(failed), base / _FAILED, primary_key=["url"])
            failed.clear()

    records = todo.to_dict("records")

    def _handle(res: dict | None) -> bool:
        """Fold one worker result into the buffers. Returns False to abort the run."""
        nonlocal n_ok, n_retry, consec_retry
        if res is None:
            return True
        if res.get("_retry"):
            n_retry += 1
            consec_retry += 1
            if consec_retry >= 40:
                LOG.warning("news-cc fetch: data.commoncrawl.org throttling hard - stopping.")
                return False
            return True
        consec_retry = 0
        if res.get("_failed"):
            failed.append({"url": res["url"], "reason": "no_extract", "at": now})
            return True
        pub = res["published"]
        pub = pub if not pd.isna(pub) else pd.Timestamp(now)
        pub = pub.tz_localize(None) if getattr(pub, "tzinfo", None) else pub
        syms = matcher.match(res["title"], res["excerpt"], res["text"][:1500])
        res.update(published=pub, symbols=",".join(syms), n_symbols=len(syms), ingested_at=now)
        res.pop("_failed", None)
        buf.setdefault(f"{pub.year:04d}-{pub.month:02d}", []).append(res)
        n_ok += 1
        if n_ok % 300 == 0:
            flush()
        return True

    # as_completed with a bounded in-flight window: a single stuck request can't
    # freeze the whole run (the old pool.map returned results strictly in order).
    from concurrent.futures import FIRST_COMPLETED, wait

    it = iter(enumerate(records))
    inflight: set = set()
    aborted = False
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in range(workers * 4):
            try:
                i, rec = next(it)
            except StopIteration:
                break
            inflight.add(pool.submit(_safe_process, rec, sessions[i % workers]))

        with tqdm(total=len(records), desc="cc articles", unit="art") as bar:
            while inflight and not aborted:
                done_set, inflight = wait(inflight, timeout=300, return_when=FIRST_COMPLETED)
                if not done_set:
                    LOG.warning("news-cc fetch: 5 min with no worker result - a request is "
                                "wedged; stopping. Re-run to resume.")
                    break
                for fut in done_set:
                    bar.update(1)
                    try:
                        ok = _handle(fut.result())
                    except Exception:  # noqa: BLE001
                        ok = True
                    if not ok:
                        aborted = True
                        break
                    try:
                        i, rec = next(it)
                        inflight.add(pool.submit(_safe_process, rec, sessions[i % workers]))
                    except StopIteration:
                        pass
        for f in inflight:
            f.cancel()

    flush()
    for sess in sessions:
        sess.close()
    LOG.info(
        "news-cc fetch: +%d articles, %d unextractable, %d throttled (will retry)",
        n_ok, len(failed), n_retry,
    )


def _safe_process(rec: dict, sess) -> dict | None:
    try:
        return _process_one(rec, sess)
    except Exception as exc:  # noqa: BLE001
        LOG.debug("worker error on %s: %s", rec.get("url"), exc)
        return {"url": rec["url"], "_failed": True}


def run(
    cfg: PipelineConfig,
    *,
    phase: str = "both",
    since: int = 2017,
    domains: list[str] | None = None,
    limit: int | None = None,
    workers: int = 6,
) -> None:
    cfg.ensure_dirs()
    dkeys = domains or list(DOMAIN_PATTERNS)
    if phase in ("catalog", "both"):
        catalog(cfg, since, dkeys)
    if phase in ("fetch", "both"):
        fetch(cfg, limit, workers)
