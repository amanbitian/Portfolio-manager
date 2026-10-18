"""Publisher news sitemaps -> article URLs (with lastmod), and full-text extraction.

A monthly sitemap lists every article a publisher put out that month, so it gives a
complete URL list for history that RSS (rolling window) and Common Crawl (partial
crawl coverage) cannot. Each article page is then fetched and its body extracted.
"""

from __future__ import annotations

import calendar
from datetime import datetime, timezone
from xml.etree import ElementTree as ET

import trafilatura

# publisher key -> monthly sitemap URL template
SITEMAPS: dict[str, str] = {
    "et": "https://economictimes.indiatimes.com/etstatic/sitemaps/et/news/{year}-{month}-1.xml",
}

_NS = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}


# publisher key -> sitemap index listing every monthly part file
SITEMAP_INDEXES: dict[str, str] = {
    "et": "https://economictimes.indiatimes.com/etstatic/sitemaps/et/news/sitemap-index.xml",
}


def sitemap_url(source: str, year: int, month: int) -> str:
    """First part of a month. Busy months are split into -2, -3, ... files as well."""
    return SITEMAPS[source].format(year=year, month=calendar.month_name[month])


def month_part_urls(index_xml: str, source: str, year: int, month: int) -> list[str]:
    """Every part file for one month, read from the publisher's sitemap index.

    A monthly sitemap caps at 10,000 URLs, so busy months come as several files
    (e.g. 2026-September-1.xml and -2.xml). Reading only -1 silently drops the rest.
    """
    import re

    name = calendar.month_name[month]
    pattern = re.compile(rf"/{year}-{name}-\d+\.xml$")
    root = ET.fromstring(index_xml.encode("utf-8"))
    out = []
    for node in root.findall("s:sitemap", _NS):
        loc = (node.findtext("s:loc", default="", namespaces=_NS) or "").strip()
        if pattern.search(loc):
            out.append(loc)
    return sorted(out, key=lambda u: int(re.search(r"-(\d+)\.xml$", u).group(1)))


def parse_urlset(xml_text: str) -> list[tuple[str, datetime | None]]:
    """(loc, lastmod in UTC) for every <url> in a sitemap."""
    root = ET.fromstring(xml_text.encode("utf-8"))
    out: list[tuple[str, datetime | None]] = []
    for node in root.findall("s:url", _NS):
        loc = (node.findtext("s:loc", default="", namespaces=_NS) or "").strip()
        if not loc:
            continue
        lm = (node.findtext("s:lastmod", default="", namespaces=_NS) or "").strip()
        lastmod = None
        if lm:
            try:
                lastmod = datetime.fromisoformat(lm).astimezone(timezone.utc)
            except ValueError:
                lastmod = None
        out.append((loc, lastmod))
    return out


def extract(html: str) -> dict | None:
    """Full article body + metadata. None if the page has no extractable text.

    Unlike the Common Crawl extractor, the body is not truncated.
    """
    import json

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
    return {
        "title": (d.get("title") or "").strip(),
        "body": (d.get("text") or "").strip(),
        "author": d.get("author"),
        "date": d.get("date"),
        "language": d.get("language"),
    }
