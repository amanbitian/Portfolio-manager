# Data Sources — status and pending work

Every data source this project pulls from, what it provides, and where it stands right
now. For the pipeline mechanics (diagram, storage layout, commands) see
[`DATAFLOW.md`](DATAFLOW.md); for news specifically see [`NEWS.md`](NEWS.md).

Run `py -3 -m data_pipeline.run status` anytime for live counts.

## ✅ Complete

| Data | Source | Auth | Coverage | Command |
|---|---|---|---|---|
| Universe | NSE archive CSV + Upstox instruments | none | 750 symbols (Nifty Total Market, current membership) | `run universe` |
| Daily OHLCV | Upstox V3 historical-candle | none | 750 symbols, 2000/2003→2026, **4.69M rows**, adjusted | `run market --interval 1day` |
| Minute OHLCV | Upstox V3 historical-candle | none | 750 symbols, 2022→2026, **288M rows**, 7.6GB | `run market --interval 1min` |
| Corporate actions | yfinance `.actions` + Screener-derived bonus/split | none | 755 symbols, 13,090 yfinance rows + **119 bonus/splits** recovered from Screener | `run corp-actions` then `run corp-actions-screener` |
| Fundamentals (deep) | Screener.in Excel export (P&L, balance sheet, cash flow, quarters, **year-end price**) | your login (semi-auto download) | **750/750 symbols**, ~10-12yr, ₹ Cr, 279,031 rows | `run screener-download` then `run screener` |
| Macro | World Bank API + hand-maintained RBI repo rate | none | 15 series, annual, back to ~1960s (WB) / 2000 (repo rate) | `run macro` |
| News — ongoing | 19 RSS feeds (ET, Business Standard, Mint, Hindu BusinessLine, NDTV Profit, BSE notices) | none | **~3,000 articles**, scheduled every 2h (Windows Task `QuantsRssNews`) | `run news --source rss` |
| Data-quality report | cross-checks every source above against the others | — | last run: 52 ERROR / 163 WARN / 390 INFO | `run sanity` |

## ⚠️ Partial / in progress

| Data | Source | Status | Command to continue |
|---|---|---|---|
| Fundamentals (shallow) | yfinance | 755 symbols, ~4yr - kept only as a cross-check source (INFY-type currency bugs, shallower than Screener) | `run fundamentals` |
| News — historical | Common Crawl (CDX index + WARC range-GET + trafilatura) | **437,912 articles fetched (2007-2019)**. 2020→2023 catalogued (2.05M captures) but not yet fetched. 2024-2026 not yet catalogued. Blocked repeatedly by Common Crawl's flaky `index.commoncrawl.org`; stops cleanly and resumes. | `run news-cc --phase catalog` then `--phase fetch` |
| News — historical (backup) | GDELT DOC 2.0 API | 1,429 rows, 489 failed - IP-rate-limited to the point of not being worth pursuing further | `run news --source gdelt` |

## ❌ Not started (pending)

| Data | Why we'd want it | What it needs |
|---|---|---|
| **Shareholding pattern** (promoter / FII / DII / pledge %) | Real signal - promoter selling/pledging is a known factor | Screener company-page scrape (not in the Excel export) - new build |
| **True Nifty 1000 + point-in-time membership** | We're on Nifty Total Market (750) with *today's* membership applied to all history → survivorship bias | Manual CSV from niftyindices.com for the current list; point-in-time history needs a paid feed or a lot of manual archaeology |
| **FRED global macro** (US rates, CPI, VIX, oil, USD/INR) | Our macro is India-only + World Bank annuals | Free API key (30-sec signup), then `run macro` picks it up automatically |
| **Deeper India macro** (CPI, GDP, IIP - not just the repo rate) | Current India macro is one series | RBI DBIE / MOSPI - no clean API, needs scraping |
| BSE company announcements | Highest-signal event data (results, orders, rating actions, management changes) | Old API (`AnnGetData/w`) is dead; new Angular API not reverse-engineered |
| Screener's own EPS/ROE/ROCE (formula tabs) | Free extra ratios, already downloaded | Tabs are broken Excel formulas with no cached value - not worth a formula evaluator; compute these ourselves from the raw numbers instead (feature work, not a new data source) |
| News sentiment / event extraction | Turn the 437K+ raw articles into usable features | Local-LLM pass over `news/commoncrawl` + `news/rss` - not started |
| Point-in-time fundamentals layer | Any backtest using fundamentals right now has look-ahead bias | Lag each number to an estimated filing date (~60d annual, ~45d quarterly) |
| Computed ratio features (PE, PB, ROE, momentum, vol as daily series) | The actual inputs a factor model needs - we only have raw numbers today | Join prices × fundamentals × shares outstanding, build a `features/` module |
| Postgres deployment | Everything is Parquet-only right now | `postgres/schema.sql` + `postgres/load.py` are written, just never run |
| Wiring into the app | `src/ml_portfolio` still runs on synthetic/uploaded data - none of this dataset is used yet | Build a loader from `market/ohlcv_1day` into the app's price matrix |

## Known data-quality issues (from `run sanity`)

| Issue | Scope | Status |
|---|---|---|
| Bad OHLC bars (zero/inconsistent) | ~15 symbols (VEDL, GNFC, ASHOKLEY, ...), concentrated 2003-2005 | identified, not cleaned |
| Price-basis mismatches (Upstox vs Screener) | 13 ERROR + 19 WARN - mostly real demergers/splits (ABFRL, NMDC) | identified, not resolved per-symbol |
| Fundamentals revenue/net-income mismatches (yfinance vs Screener) | 18+20 revenue, 7+8 net income - mostly gross-vs-net revenue or consolidated-vs-standalone | explainable, not a data bug |
| INFY fundamentals in USD (yfinance) | 1 symbol confirmed, possibly others with US ADRs | known yfinance bug, use Screener instead |
| Unexplained >50% single-day price jumps | 62 symbols, 1 instance each mostly | unverified - some real events, some possibly bad ticks |
| Illiquid/suspended flat-line runs | 39 symbols | expected for small caps, not a bug |
