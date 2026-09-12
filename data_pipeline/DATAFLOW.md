# Data Flow — ML Portfolio Allocator

Map of the pipeline's mechanics: how data moves from source to disk. For **what data we
have and what's missing**, see [`DATA.md`](DATA.md). Other companion docs:
[`README.md`](README.md) (command reference), [`NEWS.md`](NEWS.md) (news sources in
depth), [`postgres/README.md`](postgres/README.md) (Phase 2 DB load),
[`../README.md`](../README.md) (the allocator app this feeds).

## The pipeline end to end

```text
                              SOURCES
   ┌───────────┬───────────┬───────────┬───────────┬───────────┬───────────┐
   │  Upstox   │ yfinance  │Screener.in│World Bank │    RSS    │ CommonCrawl│
   │  (no key) │(unofficial)│  (Excel) │ + RBI     │ 19 feeds  │  + GDELT   │
   └─────┬─────┴─────┬─────┴─────┬─────┴─────┬─────┴─────┬─────┴─────┬─────┘
         │           │           │           │           │           │
         v           v           v           v           v           v
┌─────────────────────────────────────────────────────────────────────────┐
│                    data_pipeline/  (this package)                       │
│  universe.py   sources/*.py   ingest/*.py   entities.py   sanity.py     │
│  - resolves the Nifty-Total-Market universe + Upstox instrument keys    │
│  - one ingest module per dataset, each resumable via checkpoint.py      │
│  - entities.py maps free text -> symbols (news matching)                │
│  - sanity.py cross-checks sources against each other                    │
└──────────────────────────────┬────────────────────────────────────────┬─┘
                                v                                        v
                     F:\quants project\stock Data\           _reports\sanity_*.md
                     Parquet data lake (below)                (data-quality findings)
                                │
                                v
                    ┌───────────────────────┐
                    │   NOT BUILT YET:       │
                    │  point-in-time layer   │
                    │  -> feature store      │
                    └───────────┬───────────┘
                                v
                    ┌───────────────────────┐
                    │   NOT WIRED YET:       │
                    │  src/ml_portfolio      │   <- still runs on synthetic/
                    │  (the Streamlit app)   │      uploaded data only
                    └───────────────────────┘
```

## Datasets — status and what's pending

Moved to **[`DATA.md`](DATA.md)** - the single source of truth for "what data do we have,
from where, and what's still missing," kept current as we go. This file stays focused on
pipeline mechanics (diagram, storage layout, command reference below).

## Storage layout

```text
F:\quants project\stock Data\
  reference\
    universe.parquet                 symbol, isin, name, industry, instrument_key, yf_ticker
    upstox_instruments_nse.parquet
  market\
    ohlcv_1day\symbol=<SYM>\part.parquet
    ohlcv_1min\symbol=<SYM>\year=YYYY\month=MM\part.parquet
  corporate_actions\<SYM>.parquet
  fundamentals\
    income_stmt\ balance_sheet\ cash_flow\ key_stats\<SYM>.parquet      (yfinance)
    screener\pnl\ balance_sheet\ cash_flow\ quarters\<SYM>.parquet      (Screener, Rs Cr)
  macro\worldbank\<code>.parquet   macro\rbi\REPO_RATE.parquet
  news\
    rss\date=YYYY-MM-DD\articles.parquet
    commoncrawl\
      _catalog\<CC-MAIN-index>.parquet    (URL lists, phase 1)
      year=YYYY\month=MM\part-*.parquet   (article text, phase 2)
    gdelt\symbol=<SYM>\<YYYY-MM>.parquet
  _manifests\<dataset>.json          resumability checkpoints
  _logs\<dataset>-<date>.log
  _reports\sanity_<timestamp>.{md,parquet}

F:\quants project\Screener data\     your downloaded Screener .xlsx files
```

`F:\quants project\portfolio manager\`
```text
data_pipeline\          this package (config, sources, ingest, tools, postgres)
scripts\
  rss_news.cmd                    RSS poll, run by the scheduled task
  install_rss_schedule.ps1        registers the Windows Scheduled Task
  watch_news_cc.ps1               live dashboard for the CC news fetch
src\ml_portfolio\       the Streamlit allocator - NOT yet reading real data
app.py                  the Streamlit app
```

## Known gaps

Full ranked list with the "why it matters" reasoning now lives in **[`DATA.md`](DATA.md)**
(the "Not started" and "Known data-quality issues" sections). Short version, ranked:

1. Real data not wired into `src/ml_portfolio` (the app)
2. No point-in-time layer for fundamentals (look-ahead bias risk)
3. No computed ratio features (PE/PB/ROE/momentum/vol)
4. Price data has bad bars in 2003-2005 for ~15 symbols
5. Universe is Nifty Total Market (750), not the true Nifty 1000, current membership only
6. News 2020-2026 not yet catalogued/fetched
7. Shareholding pattern, FRED, deeper India macro, BSE announcements, news sentiment, Postgres deployment - all not started

## Recommended order of work

1. Wire real prices into `src/ml_portfolio` (highest leverage — makes everything else testable)
2. Fix corp actions + clean bad price bars (makes #1's backtests trustworthy)
3. Point-in-time fundamentals + computed ratio features
4. Everything else, as needed
