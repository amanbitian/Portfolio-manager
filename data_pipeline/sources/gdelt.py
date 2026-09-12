"""GDELT DOC 2.0 article search (no auth).

https://api.gdeltproject.org/api/v2/doc/doc  - GDELT asks for <= 1 request / 5 seconds,
which the "gdelt" rate limit in config enforces. Coverage starts ~2017.

We query per company name, chunked by month, and keep article-level metadata. Turning
articles into sentiment features is a later (local-LLM) step.
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from ..http import HttpClient

LOG = logging.getLogger("data_pipeline.gdelt")

DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_MIN_DATE = date(2017, 1, 1)


def _clean_company(name: str) -> str:
    for suffix in (" Ltd.", " Ltd", " Limited", " Corporation", " Corp."):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name.strip()


def fetch_month(
    client: HttpClient, company: str, year: int, month: int, max_records: int = 250
) -> pd.DataFrame:
    start = date(year, month, 1)
    end = date(year + (month // 12), (month % 12) + 1, 1)
    query = f'"{_clean_company(company)}"'
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": max_records,
        "sort": "datedesc",
        "startdatetime": start.strftime("%Y%m%d000000"),
        "enddatetime": end.strftime("%Y%m%d000000"),
    }
    # GDELT returns text/plain for errors and a 429 notice when throttled - raise on
    # those so the manifest records a retryable failure instead of an empty "done".
    resp = client.raw_get(DOC_API, host_key="gdelt", params=params)
    body = resp.text.strip()
    if resp.status_code == 429 or "limit requests" in body[:120].lower():
        raise RuntimeError("gdelt rate-limited (429)")
    if resp.status_code != 200:
        raise RuntimeError(f"gdelt HTTP {resp.status_code}")
    if not body.startswith("{"):
        return pd.DataFrame()  # genuine empty result
    arts = resp.json().get("articles", []) or []
    if not arts:
        return pd.DataFrame()
    df = pd.DataFrame(arts)
    keep = [c for c in ["seendate", "title", "url", "domain", "language", "sourcecountry"] if c in df.columns]
    df = df[keep].copy()
    df["seendate"] = pd.to_datetime(df["seendate"], errors="coerce", format="%Y%m%dT%H%M%SZ")
    df["query_company"] = _clean_company(company)
    return df
