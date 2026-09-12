# News data — sources, coverage, how to run

See [`DATA.md`](DATA.md) for how this fits into every other data source's status,
[`DATAFLOW.md`](DATAFLOW.md) for pipeline mechanics, and [`README.md`](README.md) for
every dataset's commands.

The pipeline pulls company news from three independent sources. They are complementary,
not interchangeable: one is for **history**, one for **live**, one is a **cheap extra**.

## Source summary

| Source | Command | Cost | History | Ongoing | Granularity | Status |
|---|---|---|---|---|---|---|
| **Common Crawl** | `news-cc` | Free | **2017 → present** | No (crawl lags weeks) | Full article text | ⏳ **437,912 articles fetched (2007–2019)**; 2020–2023 catalogued (2.05M captures), not yet fetched; 2024–2026 not yet catalogued |
| **RSS feeds** | `news --source rss` | Free | **None** (rolling 1–3 days) | ✅ yes, scheduled | Headline + summary | ✅ **~3,000 articles and growing**, scheduled every 2 h |
| **GDELT DOC API** | `news --source gdelt` | Free | 2017 → present | Yes | URL + metadata (no body) | ⚠️ built, IP-rate-limited to ~useless |
| BSE announcements | — | Free | ~2010 → present | Yes | Filing text/PDF | ❌ old API dead; new one not reverse-engineered |

*(Numbers as of the last update to this file - `run status` always has the live count.)*

## 1. Common Crawl — historical backfill

`data_pipeline/sources/commoncrawl.py` + `ingest/news_cc.py`

Common Crawl runs a broad web crawl ~monthly (`CC-MAIN-YYYY-NN`, ~10/year since 2017).
Each crawl has a CDX URL index queryable by domain/path. For every capture we HTTP
range-GET just that gzipped WARC record (~40 KB) from `data.commoncrawl.org` — no S3
auth — decompress, and run **trafilatura** to extract title / date / body text.

### Two phases (both resumable)

| Phase | What it does | Resumes via |
|---|---|---|
| `catalog` | Query each CC-MAIN index × domain-pattern → capture list at `news/commoncrawl/_catalog/<index>.parquet` | manifest `news_cc_catalog.json` |
| `fetch` | Parallel range-GET + extract + entity-match each unique URL → `news/commoncrawl/year=YYYY/month=MM/part-*.parquet` | URLs already in the output + `_failed.parquet` |

Flags: `--phase {catalog,fetch,both}`, `--since <year>` (default 2017), `--domains et,bs,…`,
`--limit <n>` (articles this run), `--workers <n>` (parallel fetch, **default 6**).

### Progress log

| Date | What happened |
|---|---|
| 2026-09-08/09 | First catalog pass: 2017–2019 fully catalogued (437,903 unique URLs). Learned 24 workers gets the IP 403-blocked by `data.commoncrawl.org` within minutes. |
| 2026-09-10 | Full 2017–2019 fetch completed at 6 workers: **437,912 articles**, 40% symbol-matched. Also hit and fixed a hang bug (`pool.map` blocks on a single stuck request) - fetch now uses `as_completed` with a bounded window + a 5-min-no-progress circuit breaker. |
| 2026-09-11/12 | Resumed `catalog` for 2020+. `index.commoncrawl.org` (the CDX query server) is much flakier than the data host - it has died mid-run twice; the job stops cleanly each time (20 failures in a row) and resumes cleanly on re-run. Currently through **2020 → mid/late-2023** (2.05M captures catalogued, not yet fetched). |

### Domains covered

| key | site | patterns |
|---|---|---|
| `et` | Economic Times | `/markets/*`, `/industry/*`, `/news/company/*` |
| `bs` | Business Standard | `/markets/*`, `/companies/*`, `/article/markets/*`, `/article/companies/*` |
| `mint` | LiveMint | `/market/*`, `/companies/*`, `/industry/*` |
| `hbl` | Hindu BusinessLine | `/markets/*`, `/companies/*`, `/money-and-banking/*` |
| `mc` | Moneycontrol | `/news/business/*` |
| `fe` | Financial Express | `/market/*`, `/industry/*`, `/business/*` |
| `bt` | Business Today | `/markets/*`, `/industry/*` |

### Run

```powershell
py -3 -m data_pipeline.run news-cc --phase catalog                  # resumable, re-run if CC's index server drops
py -3 -m data_pipeline.run news-cc --phase fetch --limit 20000      # then fetch in chunks, or omit --limit
py -3 -m data_pipeline.run news-cc --phase fetch                    # workers defaults to 6 - don't raise much, see below
# scoped:
py -3 -m data_pipeline.run news-cc --since 2020 --domains et,bs
```

Live progress: `Get-Content "F:\quants project\stock Data\_logs\news_cc_fetch.out" -Tail 5 -Wait`
(or `..._catalog.out` during the catalog phase), or `scripts\watch_news_cc.ps1` for a dashboard.

### Scale (rough)

- ~8,800 ET-markets captures per monthly index; ~1,000 BS-companies; ~4,400 LiveMint.
  One domain (BS) over 2 years = ~155 K captures → ~125 K unique.
- All domains × ~100 indexes → ~1 M raw rows → **~300–500 K unique articles** after de-dup.
- Fetch is parallel (`--workers`, **default 6 — do not raise much**). 24 workers got our
  IP temp-blocked by `data.commoncrawl.org` (403 on everything for ~hours). 6 workers is
  gentle enough; ~3-6 articles/s → full backfill ~1-2 days. Stop/resume freely.
- If `data.commoncrawl.org` starts returning 403, or `index.commoncrawl.org` refuses
  connections: that's CC throttling or an outage. Both jobs stop cleanly after 20-40
  consecutive failures. Wait a few hours and re-run the same command.
- Article *dates* are the original publish date, so a narrow catalog window (e.g. the 2025
  crawls) still yields articles spanning many prior years.

### Caveats

- The main crawl visits each site ~monthly with limited depth — you get a large **sample**
  of each period, not every article ever published.
- `date` comes from trafilatura's metadata parse; falls back to the crawl timestamp.
- Article text truncated to 8 KB per row.

## 2. RSS — ongoing signal

`data_pipeline/sources/rss.py` + `ingest/news_rss.py`

19 feeds, fetched in parallel. Rolling window only (feeds hold ~1–3 days) — **cannot
backfill**, accumulates forward from first run.

| Publisher | feeds |
|---|---|
| Economic Times | markets, stocks, stocks-to-watch, earnings, IPOs, industry |
| Business Standard | markets, companies, finance, economy |
| LiveMint | markets, companies, industry |
| Hindu BusinessLine | markets, companies, money-and-banking, economy |
| NDTV Profit | latest |
| BSE | exchange notices |

Per poll: ~770 articles → ~170 matched to ≥1 symbol → ~98 distinct companies, in ~1.5 s.
Output: `news/rss/date=<YYYY-MM-DD>/articles.parquet` (feeds merged, de-duped on url).

### Schedule

`scripts/install_rss_schedule.ps1` registers a Windows Scheduled Task **`QuantsRssNews`**
running `scripts/rss_news.cmd` every **2 hours** while the PC is on (`StartWhenAvailable`
catches a missed run on next wake). Only runs while the machine is powered on — windows
where it is off are lost.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_rss_schedule.ps1  # install once
schtasks /run   /tn QuantsRssNews                                          # run now
schtasks /query /tn QuantsRssNews                                          # status
schtasks /delete /tn QuantsRssNews /f                                      # remove
```

Moneycontrol RSS is **frozen** (stuck in Apr 2024) and excluded.

## 3. GDELT — historical, but throttled

`data_pipeline/sources/gdelt.py` + `ingest/news.py`

GDELT DOC 2.0 `artlist`, queried per company name, monthly chunks. Coverage from ~2017.
Returns URL + title + domain + tone, **no article body**.

GDELT rate-limits this IP hard (429 on nearly every request even at ~1 req/9 s). Stuck at
~100 chunks. Left in place; the real fix would be GDELT's **bulk 15-min CSV files**
(`data.gdeltproject.org/gdeltv2/`) which are un-throttled — not built.

```powershell
py -3 -m data_pipeline.run news --source gdelt --limit 25
```

## 4. Entity matching

`data_pipeline/entities.py` maps free text → universe symbols by company-name alias:

- normalised company name (corp suffixes stripped), plus
- ~60 curated short forms (`HDFC Bank`→HDFCBANK, `L&T`→LT, `Infosys`→INFY, …)
- word-boundary match, longest alias first

Every news row carries `symbols` (comma-joined) and `n_symbols`.

## Storage layout

```
news/
  commoncrawl/
    _catalog/<CC-MAIN-index>.parquet     capture lists (phase 1)
    _failed.parquet                      URLs that would not extract
    year=YYYY/month=MM/part-*.parquet    articles (phase 2)
  rss/
    date=YYYY-MM-DD/articles.parquet
  gdelt/
    symbol=<SYM>/<YYYY-MM>.parquet
```

## Next (not built)

- GDELT bulk 15-min CSV pipeline (un-throttled historical media tone).
- BSE company-announcement history (new Angular API needs reverse-engineering, or a paid feed).
- Local-LLM sentiment / event extraction over all three stores → feature layer.
- A cloud/VPS cron so RSS runs 24/7 instead of only when the PC is on.
