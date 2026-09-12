"""Historical news from Common Crawl.

Common Crawl runs a broad web crawl ~monthly (``CC-MAIN-YYYY-NN``). Each crawl has a
CDX-style URL index we can query by domain/path. For every matching capture the index
gives ``(warc filename, offset, length)`` - we HTTP range-GET just that gzipped WARC
record (~40 KB) from ``data.commoncrawl.org`` (no S3 auth), decompress, and run
trafilatura to pull title / date / body text.

Two phases, both resumable:
  1. catalog  - query each (index, domain-pattern) -> list of captures
  2. fetch    - range-GET + extract each unique URL

News-useful coverage starts ~2017 (CC-NEWS proper starts 2016; the main crawl indexes
news sites reasonably from 2017).
"""

from __future__ import annotations

import gzip
import json
import logging
import re
import time
from typing import Iterator

import requests

LOG = logging.getLogger("data_pipeline.commoncrawl")

COLLINFO = "https://index.commoncrawl.org/collinfo.json"
CDX = "https://index.commoncrawl.org/{index}-index"
DATA = "https://data.commoncrawl.org/"

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
)


class CdxUnavailable(RuntimeError):
    """The CDX index server would not answer (transient - retry on a later run)."""


def session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": _UA})
    return s


COLLECTIONS_HTML = "https://data.commoncrawl.org/cc-index/collections/index.html"
_ID_RE = re.compile(r"CC-MAIN-(20\d\d)-\d\d")


def list_indexes(s: requests.Session, since_year: int = 2017) -> list[str]:
    ids: set[str] = set()
    # primary: collinfo.json on the (often flaky) index host
    for attempt in range(3):
        try:
            for row in s.get(COLLINFO, timeout=60).json():
                ids.add(row["id"])
            break
        except requests.exceptions.RequestException:
            time.sleep(3 * (attempt + 1))
    # fallback: scrape the crawl list off the (reliable) data host
    if not ids:
        try:
            html = s.get(COLLECTIONS_HTML, timeout=60).text
            ids.update(m.group(0) for m in _ID_RE.finditer(html))
        except requests.exceptions.RequestException:
            pass
    if not ids:
        raise CdxUnavailable("could not list CC-MAIN indexes")
    keep = [
        cid for cid in ids
        if (m := _ID_RE.match(cid)) and int(m.group(1)) >= since_year
    ]
    return sorted(keep)


def query_index(
    s: requests.Session, index: str, url_pattern: str, *, rps: float = 3.0, retries: int = 3
) -> Iterator[dict]:
    """Yield capture records for a URL pattern. Raises CdxUnavailable if the server
    never answered (so the caller can mark it failed, not empty)."""
    url = CDX.format(index=index)
    for attempt in range(retries):
        try:
            time.sleep(1.0 / rps)
            r = s.get(url, params={"url": url_pattern, "output": "json"}, timeout=30)
            if r.status_code == 404:
                return
            if r.status_code in (500, 502, 503, 504, 429):
                time.sleep(3 * (attempt + 1))
                continue
            r.raise_for_status()
            for line in r.text.splitlines():
                line = line.strip()
                if not line or line[0] != "{":
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("status") not in (None, "200"):
                    continue
                if "warc" not in (rec.get("mime") or "text/html") and rec.get("mime") not in (None, "text/html"):
                    continue
                yield {
                    "url": rec["url"],
                    "timestamp": rec["timestamp"],
                    "filename": rec["filename"],
                    "offset": int(rec["offset"]),
                    "length": int(rec["length"]),
                }
            return
        except requests.exceptions.RequestException as exc:
            LOG.debug("cdx %s %s attempt %d: %s", index, url_pattern, attempt, exc)
            time.sleep(2 * (attempt + 1))
    raise CdxUnavailable(f"{index} {url_pattern}")


class RateLimited(RuntimeError):
    """data.commoncrawl.org is throttling us (403/429/503) - back off hard."""


class FetchTimeout(RuntimeError):
    """Range-GET failed on network/timeout errors after all retries - transient,
    unlike a 404, so the caller must retry it later rather than treat it as a
    permanent miss."""


def fetch_html(s: requests.Session, rec: dict, *, retries: int = 4) -> str | None:
    """Range-GET one WARC record. Returns HTML, or None if the record is genuinely
    gone (404). Raises RateLimited or FetchTimeout - both retryable - so the caller
    never mistakes a transient failure for a permanent one."""
    off, ln = rec["offset"], rec["length"]
    hdr = {"Range": f"bytes={off}-{off + ln - 1}"}
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            r = s.get(DATA + rec["filename"], headers=hdr, timeout=(10, 45))
            if r.status_code in (403, 429, 503):
                if attempt == retries - 1:
                    raise RateLimited(str(r.status_code))
                time.sleep(6 * (attempt + 1))
                continue
            if r.status_code == 404:
                return None
            r.raise_for_status()
            raw = gzip.decompress(r.content)
            parts = raw.split(b"\r\n\r\n", 2)  # WARC hdr / HTTP hdr / body
            return parts[2].decode("utf-8", "replace") if len(parts) >= 3 else None
        except (requests.exceptions.RequestException, OSError, EOFError) as exc:
            LOG.debug("range-get %s attempt %d: %s", rec["url"], attempt, exc)
            last_exc = exc
            time.sleep(3 * (attempt + 1))
    raise FetchTimeout(f"{rec['url']}: {last_exc}")


def extract_article(html: str) -> dict | None:
    import trafilatura

    js = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=False,
        with_metadata=True,
        favor_recall=True,
        output_format="json",
    )
    if not js:
        return None
    d = json.loads(js)
    text = (d.get("text") or "").strip()
    if len(text) < 120:
        return None
    return {
        "title": (d.get("title") or "").strip(),
        "date": d.get("date"),
        "author": d.get("author"),
        "sitename": d.get("sitename"),
        "text": text[:8000],
        "excerpt": (d.get("excerpt") or "").strip()[:500],
    }
