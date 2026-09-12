# data_pipeline

Raw-data ingestion for the ML portfolio allocator. Fetches market prices, corporate
actions, fundamentals, macro series and news for the current broad NSE universe and
writes them as partitioned **Parquet** under `STOCK_DATA_ROOT`
(default `F:\quants project\stock Data`).

## Documentation map

| Doc | What's in it |
|---|---|
| **This file** | Install, every command, output layout, per-source notes |
| [`DATA.md`](DATA.md) | **Start here for "what data do we have and what's missing"** - every source grouped complete / in-progress / not-started, plus known data-quality issues |
| [`DATAFLOW.md`](DATAFLOW.md) | The pipeline mechanics: a diagram, the full storage tree, and the recommended order of remaining work |
| [`NEWS.md`](NEWS.md) | Deep dive on the 3 news sources (Common Crawl, RSS, GDELT), why BSE's API is dead, and a running progress log of the Common Crawl backfill |
| [`postgres/README.md`](postgres/README.md) | Phase 2: loading the Parquet lake into PostgreSQL (not run yet) |
| [`../README.md`](../README.md) | The Streamlit allocator app this data feeds (not yet wired to it) |
| [`../architecture.md`](../architecture.md) | The allocator's target end-state architecture |

## Install

```powershell
py -3 -m pip install -r requirements-data.txt
```

## Commands

```powershell
py -3 -m data_pipeline.run universe                        # build reference/universe.parquet
py -3 -m data_pipeline.run market --interval 1day          # OHLCV 2000 -> today
py -3 -m data_pipeline.run market --interval 1min          # OHLCV 2022 -> today (hours; resumable)
py -3 -m data_pipeline.run corp-actions                    # dividends + splits (yfinance)
py -3 -m data_pipeline.run corp-actions-screener            # + bonuses/splits Screener's balance sheet reveals
py -3 -m data_pipeline.run fundamentals                    # income / balance / cashflow / key stats
py -3 -m data_pipeline.run macro                           # World Bank + FRED + RBI repo rate
py -3 -m data_pipeline.run news --source rss               # Indian media + BSE-notice RSS (ongoing)
py -3 -m data_pipeline.run news --source gdelt --limit 25  # GDELT article metadata (rate-limited)
py -3 -m data_pipeline.run screener                        # ingest screener.in Excel exports
py -3 -m data_pipeline.run sanity                          # cross-source data-quality report
py -3 -m data_pipeline.run status                          # progress + disk summary
py -3 -m data_pipeline.run all                             # everything except the full 1-min job
```

Useful flags: `--symbols RELIANCE,INFY`, `--limit N` (first N symbols),
`--years N` (news backfill window).

## Layout written

```
reference/  universe.parquet, upstox_instruments_nse.parquet
market/ohlcv_1day/symbol=<SYM>/part.parquet
market/ohlcv_1min/symbol=<SYM>/year=YYYY/month=MM/part.parquet
corporate_actions/<SYM>.parquet
fundamentals/{income_stmt,balance_sheet,cash_flow,key_stats}/<SYM>.parquet
macro/{worldbank,fred,rbi}/<series>.parquet
news/gdelt/symbol=<SYM>/<YYYY-MM>.parquet
news/rss/date=<YYYY-MM-DD>/articles.parquet   (feeds merged, symbols matched)
_manifests/<dataset>.json     resumable progress
_logs/<dataset>-<date>.log
```

## Resumability

Every long job writes a JSON manifest under `_manifests/`. Re-running a command skips
work units already marked `done`, so `market --interval 1min` (~57 monthly chunks x
~750 symbols) can be stopped with Ctrl-C and resumed, or run in the background across
sessions.

## Data-source notes

| Dataset            | Source            | Auth | Notes |
|--------------------|-------------------|------|-------|
| OHLCV 1d / 1m      | Upstox V3 historical-candle | none | **Prices are already split/bonus adjusted** by Upstox. Daily back to 2000, minute back to 2022. |
| Corporate actions  | yfinance `.actions` | none | Dividends + split ratios only. Bonus/rights need NSE's feed (bot-protected) - add later. |
| Fundamentals       | yfinance          | none | ~4-5 years only. Deep history needs NSE XBRL filings (future pipeline). |
| Fundamentals (deep)| screener.in Excel exports | manual download | ~10-12 yr standardized P&L / balance sheet / cash flow + monthly price. Drop `.xlsx` in the Screener folder, run `screener`. Values in **Rs Crore**. |
| Macro - global     | World Bank        | none | Annual, ~1960+. API is occasionally flaky (502s); re-run `macro` to backfill. |
| Macro - global     | FRED              | free key | Set `FRED_API_KEY` (30-second signup). Skipped with a warning if unset. Gives US rates, CPI, VIX, Brent/WTI, USD/INR. |
| Macro - India      | RBI repo rate     | bundled | Hand-maintained series in `sources/rbi.py`. DBIE / e-Sankhyiki are documented there for later. |
| News - historical  | GDELT DOC 2.0     | none | Heavily IP-rate-limited (429s under load). Coverage from ~2017. |
| News - ongoing     | RSS: 19 feeds - ET, BS, Mint, Hindu BusinessLine, NDTV Profit, BSE notices | none | Rolling window (feeds hold ~1-3 days), **no backfill**. ~770 articles/poll (feeds fetched in parallel), ~20% match to >=1 symbol via `entities.py` name aliases. Run every ~2 h so no window is missed - `scripts\install_rss_schedule.ps1` registers a Windows Scheduled Task (`QuantsRssNews`). Only runs while the PC is on. |

## Universe

The true **Nifty 1000** list is not published as a free NSE CSV and niftyindices.com is
bot-protected. `universe` therefore falls back to NSE's **Nifty Total Market (~750)**
archive CSV. To use the exact Nifty 1000: download it from niftyindices.com
(Nifty 1000 -> "Download (.csv)"), save it as `reference/universe_manual.csv` with at
least `symbol,isin` columns, and re-run `py -3 -m data_pipeline.run universe`.

## Screener.in cross-check

**Get the files.** Either export manually (company page -> "Export to Excel", save as
`<SYMBOL>.xlsx`) or bulk-download with your login session:

```powershell
# 1. log in at screener.in, copy the `sessionid` cookie (DevTools -> Application -> Cookies)
set SCREENER_SESSIONID=<value>
py -3 -m data_pipeline.run screener-download --limit 25     # try a few first
py -3 -m data_pipeline.run screener-download                # whole universe (~1 h, resumable)
```

`screener-download` resolves each symbol to its screener company page (direct slug ->
ticker search -> name search), submits the "Export to Excel" form (a POST with the page's
`csrfmiddlewaretoken`), and saves `<SYMBOL>.xlsx` into the Screener folder. Works on a
free account. Rate-limited to ~1 request / 4.5 s. Symbols it can't resolve (renames,
delistings, missing pages) are listed in `_not_found.txt` - grab those by hand.
`--standalone` fetches standalone instead of consolidated. Note: screener.in's ToS
discourages automated access; this uses your own account for personal research.

**Ingest them.** `screener` parses the `.xlsx` files (name `<SYMBOL>.xlsx` or map via
`<folder>\_symbol_map.csv` with `file,symbol`) into
`fundamentals/screener/{pnl,quarters,balance_sheet,cash_flow,price}/<SYM>.parquet`.

## Data sanity checker

`sanity` runs cross-source checks and writes `_reports/sanity_<ts>.{md,parquet}`:

- **price_1day**: non-positive prices, high/low not bounding open/close, duplicate
  timestamps, >50% daily moves with no nearby corporate action, multi-day history gaps,
  flat-line runs, stale/short history.
- **fundamentals_yf**: non-positive revenue, stale statements, missing coverage.
- **screener**: cash-flow reconciliation (CFO+CFI+CFF vs net), non-positive sales.
- **xcheck_fundamentals**: yfinance vs Screener revenue & net income per fiscal year
  (unit-normalised Crore -> Rs); >10% -> WARN, >30% -> ERROR.
- **xcheck_price**: Upstox vs Screener monthly close (detects split/adjustment basis
  mismatches).
- **xcheck_corp_actions**: Screener share-count / bonus changes with no matching
  split/bonus in our data.

Severities: ERROR (unusable), WARN (look at it), INFO (fyi).

## Next phases (not built)

- NSE XBRL corporate filings for deep fundamentals + filing dates.
- Local-LLM sentiment / event extraction over `news/gdelt/` + `news/rss/` articles.
- BSE company-announcement history: the old `api.bseindia.com/.../AnnGetData/w` endpoint
  now returns nothing; BSE's new Angular API needs reverse-engineering, or a paid feed.
- Point-in-time / feature-store layer feeding `src/ml_portfolio`.
