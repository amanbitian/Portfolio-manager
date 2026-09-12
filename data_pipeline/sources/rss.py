"""RSS / Atom news feeds - Indian financial media + BSE exchange notices.

Feeds are a rolling window (latest ~30-60 items each), so this is an **ongoing** signal:
run it on a schedule and it accumulates. There is no historical backfill from RSS.

Each run yields article dicts:
    {feed, category, published (UTC), title, summary, url}
Entity matching to universe symbols happens in the ingest layer.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import feedparser

LOG = logging.getLogger("data_pipeline.rss")

# Query params that only carry tracking/attribution noise - safe to drop. Anything
# else (e.g. "id" on a BSE notice link) is kept: it can be the only thing that
# distinguishes two otherwise-identical article URLs.
_TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "utm_id",
    "utm_name", "utm_reader", "utm_viz_id", "utm_pubreferrer", "utm_swu",
    "fbclid", "gclid", "gclsrc", "msclkid", "mc_cid", "mc_eid", "cmpid",
    "_hsenc", "_hsmi",
}


def _clean_url(url: str) -> str:
    parts = urlsplit(url)
    kept = [
        (k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if k.lower() not in _TRACKING_PARAMS
    ]
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(kept), ""))

# (id, category, url). Moneycontrol RSS is frozen (stuck in 2024) - excluded.
FEEDS: list[tuple[str, str, str]] = [
    ("et_markets", "markets", "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"),
    ("et_stocks", "markets", "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms"),
    ("et_stockstowatch", "markets", "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146843.cms"),
    ("et_earnings", "results", "https://economictimes.indiatimes.com/markets/earnings/rssfeeds/1052732854.cms"),
    ("et_ipos", "markets", "https://economictimes.indiatimes.com/markets/ipos/fpos/rssfeeds/14655708.cms"),
    ("et_industry", "companies", "https://economictimes.indiatimes.com/industry/rssfeeds/13352306.cms"),
    ("bs_markets", "markets", "https://www.business-standard.com/rss/markets-106.rss"),
    ("bs_companies", "companies", "https://www.business-standard.com/rss/companies-101.rss"),
    ("bs_finance", "finance", "https://www.business-standard.com/rss/finance-103.rss"),
    ("bs_economy", "economy", "https://www.business-standard.com/rss/economy-102.rss"),
    ("mint_companies", "companies", "https://www.livemint.com/rss/companies"),
    ("mint_markets", "markets", "https://www.livemint.com/rss/markets"),
    ("mint_industry", "companies", "https://www.livemint.com/rss/industry"),
    ("hbl_markets", "markets", "https://www.thehindubusinessline.com/markets/feeder/default.rss"),
    ("hbl_companies", "companies", "https://www.thehindubusinessline.com/companies/feeder/default.rss"),
    ("hbl_money", "finance", "https://www.thehindubusinessline.com/money-and-banking/feeder/default.rss"),
    ("hbl_economy", "economy", "https://www.thehindubusinessline.com/economy/feeder/default.rss"),
    ("ndtvprofit", "markets", "https://feeds.feedburner.com/ndtvprofit-latest"),
    ("bse_notices", "exchange", "https://www.bseindia.com/data/xml/notices.xml"),
]

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
)


def _to_utc(entry) -> datetime | None:
    for key in ("published_parsed", "updated_parsed"):
        tm = entry.get(key)
        if tm:
            return datetime(*tm[:6], tzinfo=timezone.utc)
    return None


def _clean(text: str | None) -> str:
    if not text:
        return ""
    import re

    return re.sub(r"<[^>]+>", "", text).replace("&nbsp;", " ").strip()


def fetch_feed(feed_id: str, category: str, url: str) -> list[dict]:
    parsed = feedparser.parse(url, agent=_UA)
    if parsed.bozo and not parsed.entries:
        LOG.warning("rss: %s unreadable (%s)", feed_id, getattr(parsed, "bozo_exception", ""))
        return []
    out: list[dict] = []
    for e in parsed.entries:
        title = _clean(e.get("title"))
        link = e.get("link") or ""
        if not title or not link or title.lower() in {"economic times", "markets", "companies"}:
            continue
        out.append(
            {
                "feed": feed_id,
                "category": category,
                "published": _to_utc(e),
                "title": title,
                "summary": _clean(e.get("summary") or e.get("description")),
                "url": _clean_url(link),
            }
        )
    return out


def fetch_all(workers: int = 8) -> list[dict]:
    def _one(spec: tuple[str, str, str]) -> list[dict]:
        feed_id, category, url = spec
        try:
            got = fetch_feed(feed_id, category, url)
            LOG.info("rss: %s -> %d items", feed_id, len(got))
            return got
        except Exception as exc:  # noqa: BLE001
            LOG.warning("rss: %s failed: %s", feed_id, exc)
            return []

    articles: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for got in pool.map(_one, FEEDS):
            articles.extend(got)
    return articles
