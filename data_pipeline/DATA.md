# Data Sources — status and pending work

Every data source this project pulls from, what it provides, and where it stands right
now. For the pipeline mechanics (diagram, storage layout, commands) see
[`DATAFLOW.md`](DATAFLOW.md); for news specifically see [`NEWS.md`](NEWS.md).

Run `py -3 -m data_pipeline.run status` anytime for live counts.

## ✅ Complete

| Data | Source | Auth | Coverage | Command |
|---|---|---|---|---|
| Universe | NSE archive CSV + Upstox instruments | none | 750 symbols (Nifty Total Market, current membership) | `run universe` |
| Daily OHLCV | Upstox V3 historical-candle | none | 750 symbols, 2000/2003→2026 (per-symbol start = listing date - newer IPOs like MEESHO/WAKEFIT/GROWW only have ~200 rows), **2,625,446 rows** (verified via parquet metadata), adjusted. Synced to 2026-09-18 | `run market --interval 1day` |
| Minute OHLCV | Upstox V3 historical-candle | none | 750 symbols, 2022→2026, **290.1M rows** (verified), 7.6GB. Synced to 2026-09-18 | `run market --interval 1min` |
| Corporate actions | yfinance `.actions` + Screener-derived bonus/split | none | 750 symbols, 13,201 yfinance rows + **118 bonus/splits** recovered from Screener. Full re-sync 2026-09-18 (see [[data-pipeline]] merge-safety fix - now additive, never overwrites) | `run corp-actions` then `run corp-actions-screener` |
| Fundamentals (deep) | Screener.in Excel export (P&L, balance sheet, cash flow, quarters, **year-end price**) | your login (semi-auto download) | **750/750 symbols**, ~10-12yr, ₹ Cr, 279,031 rows. Last regenerated 2026-09-12 (not re-synced 2026-09-18 - financials don't change daily, and re-downloading hits your personal login session, ~1hr) | `run screener-download` then `run screener` |
| Shareholding pattern | NSE's own `/api/corporate-share-holdings-master` (unofficial, no auth) | none | **750/750 symbols**, 14,157 quarterly rows, promoter+group % / public % / employee-trust % only — finer FII/DII/pledge detail lives in each quarter's linked XBRL filing, not fetched. Full re-sync 2026-09-18 | `run shareholding` |
| Mutual fund NAV (India) | AMFI's daily feed via `api.mfapi.in` (unofficial, no auth) | none | **37,895/37,895 schemes** (all plans/options, incl. closed legacy ones), **35.3M daily NAV rows**, full history per scheme, 0 failed. Refreshed to 2026-09-18 (`--refresh-after-days`, since mfapi always returns full history so a "done" scheme otherwise never re-checks for new NAVs) | `run mutual-funds --refresh-after-days N` |
| Macro | World Bank API + hand-maintained RBI repo rate | none | 15 series, annual, back to ~1960s (WB, through 2025) / 2000 (repo rate, current through 2025-12-05, no MPC change since). Checked 2026-09-18, nothing new to pull | `run macro` |
| News — ongoing | 19 RSS feeds (ET, Business Standard, Mint, Hindu BusinessLine, NDTV Profit, BSE notices) | none | **6,426 articles**, scheduled every 2h (Windows Task `QuantsRssNews`) - the only source that's genuinely live/current every day | `run news --source rss` |
| Data-quality report | cross-checks every source above against the others | — | last run 2026-09-12 (post-regeneration): 52 ERROR / 163 WARN / 390 INFO. Not re-run after the 2026-09-18 full re-sync - due for a fresh run | `run sanity` |

## ⚠️ Partial / in progress

| Data | Source | Status | Command to continue |
|---|---|---|---|
| Fundamentals (shallow) | yfinance | 750/750 symbols, ~4-5yr, 803,411 rows across balance sheet/cash flow/income stmt/key stats - kept only as a cross-check source (INFY-type currency bugs, shallower than Screener). Full re-sync 2026-09-18 (took ~3.5hr - single-threaded, 7 sequential yfinance calls/symbol, no incremental-refresh flag so a full manifest reset was needed to pick up anything new). Statements only change quarterly/annually - **not worth a daily sync**; `key_stats`' market-derived ratios (PE, market cap) are the only genuinely-daily part and are better computed locally from existing price data than re-fetched | `run fundamentals` |
| News — historical | Common Crawl (CDX index + WARC range-GET + trafilatura) | **2020-2026 backfill complete as of 2026-09-14**: all 1,278,848 URLs catalogued from 2020+ crawl snapshots are now fetched (0 remaining), up from a 437,912 pre-2020-backfill baseline. Separately, a **~100,029-URL pre-2020 backlog** (crawls before 2020 that were never fetched) still exists and was not in scope for this effort - use `fetch` without `--since` to work through that one. Catalog covers through CC-MAIN-2026-21 (2,516,838 captures, 34 patterns still stuck on an `index.commoncrawl.org` outage - stops cleanly, re-run to retry). Caveats found along the way: (1) `--since` filters by *crawl* date, not *publish* date - some fetched pages are old-but-still-live content re-discovered by a newer crawl, so this backfill's actual publish-date mix skews wider than just 2020-2026; (2) `data.commoncrawl.org` hard-throttles after sustained heavy use regardless of worker count, auto-stopping cleanly (not an IP ban) - seen at both 12 and 8 workers, just re-run `fetch` to resume; (3) one run hit an unrelated transient segfault (exit 139, no Python traceback) on startup - retried once and ran fine, a one-off native-library crash, not a code bug. **Worker-count note**: 12 workers is faster (12-20 art/s) but throttles sooner; 8 was the sustainable choice used to finish this backfill (vs. the original default of 6) | `run news-cc --phase fetch` (no `--since`, to start on the ~100K pre-2020 backlog) or `--phase catalog` (to clear the 34 stuck patterns and keep discovering new crawls) |
| News — historical (backup) | GDELT DOC 2.0 API | 1,429 rows, 489 failed - IP-rate-limited to the point of not being worth pursuing further | `run news --source gdelt` |

## ❌ Not started (pending) — with free sources identified 2026-09-12

| Data | Why we'd want it | What it needs |
|---|---|---|
| **Mutual fund portfolio holdings** | See what funds actually hold, not just NAV (NAV itself is done - see Complete table) - enables exposure/overlap/turnover signals (e.g. "which funds added HDFC Bank this month") | SEBI mandates monthly disclosure but there's no central API - one Excel workbook per AMC per month, on AMFI's portfolio-disclosure page and each AMC's own site. Bigger scrape+parse job across dozens of sites. Normalize to `date, amc, scheme_code, isin, security_name, sector, quantity, market_value, portfolio_weight` |
| **Mutual fund AUM / flows / folios** | Net flow, AUM growth, and market-share signals per scheme/category - more useful for quant research than NAV alone | AMFI publishes monthly scheme-level resource-mobilisation (inflow/outflow/net flow), AUM/AAUM, and folio-count reports - XLS/PDF archives, no API, needs a scrape+parse job across monthly releases. SEBI's stats page (`sebi.gov.in/statistics`) has the same aggregates as a cross-check/validation layer |
| **Mutual fund expense ratio / benchmark / riskometer / dividends** | TER for net-of-cost comparisons, benchmark for alpha/beta, riskometer, distribution events | Scattered across AMC/AMFI disclosures, no central historical API; availability varies by scheme and AMC |
| **FII/DII/pledge-level shareholding detail** | `shareholding` above only has the promoter/public summary split | Parse the per-quarter XBRL filing linked from each NSE shareholding record (`xbrl` field) - structured but a new XML-parsing build |
| **True Nifty 1000 + point-in-time membership** | We're on Nifty Total Market (750) with *today's* membership applied to all history → survivorship bias | `niftyindices.com/IndexConstituent/ind_nifty1000list.csv` exists but the site's bot-detection blocks us - try the same `curl_cffi` Chrome-impersonation trick `sources/_yf.py` already uses for yfinance. Point-in-time history: NSE's `archives.nseindia.com/IndexInclExcl.xls` only covers changes through Jul 2020 - no free complete source found; would need paid data or manual archaeology |
| **FRED global macro** (US rates, CPI, VIX, oil, USD/INR) | Our macro is India-only + World Bank annuals | Free API key (30-sec signup), then `run macro` picks it up automatically |
| **Deeper India macro** (CPI, GDP, IIP - not just the repo rate) | Current India macro is one series | `data.gov.in` has a free API key (same signup pattern as FRED) and MOSPI publishes GDP/CPI/IIP datasets there - codebase already has an unused `DATA_GOV_IN_KEY` slot in `rbi.py` waiting for this |
| BSE company announcements | Highest-signal event data (results, orders, rating actions, management changes) | Old API (`AnnGetData/w`) is dead; new Angular API not reverse-engineered. Free substitute found: NSE's own `/api/corporate-announcements` (unofficial, no auth, same session/cookie pattern as `shareholding`) - covers the same events for dual-listed companies, which is most of the universe |
| Screener's own EPS/ROE/ROCE (formula tabs) | Free extra ratios, already downloaded | Tabs are broken Excel formulas with no cached value - not worth a formula evaluator; compute these ourselves from the raw numbers instead (feature work, not a new data source) |
| News sentiment / event extraction | Turn the 437K+ raw articles into usable features | Local-LLM pass over `news/commoncrawl` + `news/rss` - not started |
| Point-in-time fundamentals layer | Any backtest using fundamentals right now has look-ahead bias | Lag each number to an estimated filing date (~60d annual, ~45d quarterly) |
| Computed ratio features (PE, PB, ROE, momentum, vol as daily series) | The actual inputs a factor model needs - we only have raw numbers today | Join prices × fundamentals × shares outstanding, build a `features/` module |
| Postgres deployment | Everything is Parquet-only right now | `postgres/schema.sql` + `postgres/load.py` are written, just never run |
| Wiring into the app | `src/ml_portfolio` still runs on synthetic/uploaded data - none of this dataset is used yet | Build a loader from `market/ohlcv_1day` into the app's price matrix |

## Known data-quality issues (from `run sanity`)

| Issue | Scope | Status |
|---|---|---|
| Screener cash-flow section mis-parsed | Was 746/747 symbols missing a proper "Net Cash Flow" line (mislabeled as generic "Cash_Flow") | **fixed and regenerated 2026-09-12** (exact-match section headers, was substring match) - verified 0/747 now contaminated |
| Phantom bonus events alongside real splits | Was at least 1 confirmed (CAMPUS, 2022-03-31: real 2x split double-counted as a 2x bonus) | **fixed and regenerated 2026-09-12** (require positive bonus evidence + reconcile against splits; ingest now also self-heals on every re-run) - verified CAMPUS now shows only the real split |
| Bad OHLC bars (zero/inconsistent) | ~15 symbols (VEDL, GNFC, ASHOKLEY, ...), concentrated 2003-2005 | identified, not cleaned |
| Price-basis mismatches (Upstox vs Screener) | 13 ERROR + 19 WARN - mostly real demergers/splits (ABFRL, NMDC) | identified, not resolved per-symbol |
| Fundamentals revenue/net-income mismatches (yfinance vs Screener) | 18+20 revenue, 7+8 net income - mostly gross-vs-net revenue or consolidated-vs-standalone | explainable, not a data bug |
| INFY fundamentals in USD (yfinance) | 1 symbol confirmed, possibly others with US ADRs | known yfinance bug, use Screener instead |
| Unexplained >50% single-day price jumps | 62 symbols, 1 instance each mostly | unverified - some real events, some possibly bad ticks |
| Illiquid/suspended flat-line runs | 39 symbols | expected for small caps, not a bug |
