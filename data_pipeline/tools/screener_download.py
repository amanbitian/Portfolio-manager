"""Download screener.in "Export to Excel" workbooks for the whole universe.

screener.in has no public API and its ToS discourages automated access - this uses
*your own* logged-in session for personal research, rate-limited and resumable. Use it
at your own discretion.

Setup:
  1. Log in to https://www.screener.in in a browser.
  2. DevTools -> Application -> Cookies -> https://www.screener.in -> copy the value of
     the `sessionid` cookie.
  3. set SCREENER_SESSIONID=<that value>

Run:
  py -3 -m data_pipeline.run screener-download                 # whole universe
  py -3 -m data_pipeline.run screener-download --symbols RELIANCE,INFY
  py -3 -m data_pipeline.run screener-download --limit 25
  py -3 -m data_pipeline.run screener-download --standalone    # standalone, not consolidated
  py -3 -m data_pipeline.run screener-download --force          # re-download existing files

Saves ``<SYMBOL>.xlsx`` into the Screener data folder. Then run
``py -3 -m data_pipeline.run screener`` to ingest them.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from typing import NamedTuple

import pandas as pd
import requests

from ..config import PipelineConfig, load_config
from ..universe import load_universe

LOG = logging.getLogger("data_pipeline.tools.screener_download")

BASE = "https://www.screener.in"
SEARCH = BASE + "/api/company/search/"
_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
)
_XLSX_MAGIC = b"PK\x03\x04"
# The "Export to Excel" button is <button formaction="/user/company/export/<id>/"> inside
# a <form method="post"> that carries a csrfmiddlewaretoken + next hidden input.
_EXPORT_RE = re.compile(r'formaction="(/user/company/export/\d+/)"')
_CSRF_RE = re.compile(r'name="csrfmiddlewaretoken"\s+value="([^"]+)"')
_NEXT_RE = re.compile(r'name="next"\s+value="([^"]+)"')


class ExportForm(NamedTuple):
    action: str          # POST path, e.g. /user/company/export/6598251/
    csrf: str            # csrfmiddlewaretoken from the form
    next_path: str       # the form's `next` field
    referer: str         # company page URL (Referer for the POST)


class NotLoggedIn(RuntimeError):
    pass


def _session(cfg: PipelineConfig) -> requests.Session:
    if not cfg.screener_sessionid:
        sys.exit(
            "SCREENER_SESSIONID not set. Log in to screener.in, copy the `sessionid` "
            "cookie from your browser, and: set SCREENER_SESSIONID=<value>"
        )
    s = requests.Session()
    s.headers.update({"User-Agent": _UA, "Referer": BASE + "/"})
    s.cookies.set("sessionid", cfg.screener_sessionid, domain=".screener.in")
    return s


def _norm_name(text: str) -> str:
    text = re.sub(r"[^a-z0-9 ]", " ", str(text).lower())
    text = re.sub(r"\b(ltd|limited|the|india|indian|company|co|corp|corporation)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _search(s: requests.Session, q: str) -> list[dict]:
    r = s.get(SEARCH, params={"q": q}, timeout=30)
    r.raise_for_status()
    try:
        return r.json() or []
    except ValueError:
        return []


def _page_has_export(s: requests.Session, url: str) -> ExportForm | None:
    r = s.get(url, timeout=30, allow_redirects=True)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    if "id_username" in r.text or r.url.rstrip("/").endswith("/login"):
        raise NotLoggedIn(url)
    t = r.text
    m = _EXPORT_RE.search(t)
    if not m:
        return None
    action = m.group(1)
    # csrfmiddlewaretoken / next live in the same <form> - search a window before the button
    form = t[max(0, m.start() - 2000):m.start()]
    csrf = _CSRF_RE.search(form) or _CSRF_RE.search(t)
    nxt = _NEXT_RE.search(form)
    if not csrf:
        return None
    return ExportForm(
        action=action,
        csrf=csrf.group(1),
        next_path=nxt.group(1) if nxt else r.url.replace(BASE, ""),
        referer=r.url,
    )


def resolve_export(
    s: requests.Session, symbol: str, name: str | None, standalone: bool
) -> ExportForm | None:
    """Locate a company's Export-to-Excel form, or None if it can't be found."""
    suffixes = ["/", "/consolidated/"] if standalone else ["/consolidated/", "/"]

    # 1. direct company slug == NSE symbol (handles BAJAJ-AUTO, M&M, ...)
    for suf in suffixes:
        got = _page_has_export(s, f"{BASE}/company/{symbol}{suf}")
        if got:
            return got

    # 2. search by ticker, accept an exact slug or exact normalised-name hit
    want_name = _norm_name(name) if name else ""
    for hit in _search(s, symbol):
        slug = hit.get("url", "").strip("/").split("/")[1:2]
        if (slug and slug[0].upper() == symbol.upper()) or (
            want_name and _norm_name(hit.get("name", "")) == want_name
        ):
            got = _follow(s, hit["url"], suffixes)
            if got:
                return got

    # 3. search by company name, accept only an exact normalised-name hit
    if want_name and name:
        for hit in _search(s, name):
            if _norm_name(hit.get("name", "")) == want_name:
                got = _follow(s, hit["url"], suffixes)
                if got:
                    return got
    return None


def _follow(s: requests.Session, company_url: str, suffixes: list[str]) -> ExportForm | None:
    base = BASE + re.sub(r"/(consolidated/)?$", "/", company_url)
    for suf in suffixes:
        got = _page_has_export(s, base.rstrip("/") + suf)
        if got:
            return got
    return None


def _download(s: requests.Session, form: ExportForm) -> bytes:
    r = s.post(
        BASE + form.action,
        data={"csrfmiddlewaretoken": form.csrf, "next": form.next_path},
        headers={"Referer": form.referer},
        timeout=90,
        allow_redirects=False,
    )
    if r.status_code in (301, 302):
        raise NotLoggedIn("export redirected - sessionid expired or account can't export")
    r.raise_for_status()
    if not r.content.startswith(_XLSX_MAGIC):
        raise RuntimeError("response was not an .xlsx (login page / premium wall?)")
    return r.content


def run(
    cfg: PipelineConfig,
    *,
    symbols: list[str] | None = None,
    limit: int | None = None,
    standalone: bool = False,
    force: bool = False,
) -> None:
    cfg.screener_root.mkdir(parents=True, exist_ok=True)
    uni = load_universe(cfg)
    if symbols:
        uni = uni[uni["symbol"].isin({x.upper() for x in symbols})]
    if limit:
        uni = uni.head(limit)

    names = dict(zip(uni["symbol"], uni.get("name", pd.Series(dtype=str))))
    s = _session(cfg)
    gap = 1.0 / cfg.rate_limits["screener"].rps
    ok = skipped = failed = 0
    missing: list[str] = []

    for sym in uni["symbol"]:
        dest = cfg.screener_root / f"{sym}.xlsx"
        if dest.exists() and not force:
            skipped += 1
            continue
        try:
            time.sleep(gap)
            found = resolve_export(s, sym, names.get(sym), standalone)
            if not found:
                missing.append(sym)
                LOG.info("screener-dl: %s not found on screener.in", sym)
                continue
            time.sleep(gap)
            data = _download(s, found)
            dest.write_bytes(data)
            ok += 1
            LOG.info("screener-dl: %s -> %s (%d KB)", sym, dest.name, len(data) // 1024)
        except NotLoggedIn as exc:
            sys.exit(f"screener-dl: not logged in ({exc}). Refresh SCREENER_SESSIONID.")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            LOG.warning("screener-dl: %s failed: %s", sym, exc)

    if missing:
        (cfg.screener_root / "_not_found.txt").write_text("\n".join(missing), encoding="utf-8")
    LOG.info(
        "screener-dl done: %d downloaded, %d skipped (exist), %d failed, %d not found",
        ok, skipped, failed, len(missing),
    )
    if ok:
        LOG.info("next: py -3 -m data_pipeline.run screener")


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="data_pipeline.tools.screener_download")
    p.add_argument("--symbols")
    p.add_argument("--limit", type=int)
    p.add_argument("--standalone", action="store_true")
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(
        load_config(),
        symbols=[x.strip().upper() for x in a.symbols.split(",")] if a.symbols else None,
        limit=a.limit,
        standalone=a.standalone,
        force=a.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
