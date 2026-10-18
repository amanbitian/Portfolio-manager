"""Pipeline configuration: paths, date ranges and per-source rate limits.

Everything is overridable through environment variables so the same code runs on a
laptop and on a server without edits:

- ``STOCK_DATA_ROOT``   where Parquet is written (default ``F:/quants project/stock Data``)
- ``FRED_API_KEY``      free key from https://fred.stlouisfed.org/docs/api/api_key.html
- ``PG_DSN``            Postgres connection string (Phase 2 loader only)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

DEFAULT_DATA_ROOT = Path("F:/quants project/stock Data")
DEFAULT_SCREENER_ROOT = Path("F:/quants project/Screener data")

# History boundaries (inclusive start).
DAILY_START = date(2000, 1, 1)
MINUTE_START = date(2022, 1, 1)

# News backfill defaults to a rolling window; full history is a separate long job.
NEWS_BACKFILL_YEARS = 3

# Full-text news from publisher sitemaps (ingest/news_fulltext.py). The run-time request
# rate and fetch-worker count come from the command line (--rps, --workers) so a running
# job's values are visible in its launch command; these are the defaults.
FULLTEXT_ET_RPS = 8.0            # article requests per second to Economic Times
FULLTEXT_WORKERS = 10            # fetch threads
FULLTEXT_EXTRACT_PROCESSES = 30  # CPU processes for text extraction (~0.03 s CPU per article)
FULLTEXT_CHUNK = 200             # articles per batch; the block/throttle brake is checked after each batch
FULLTEXT_SAVE_EVERY = 2000       # rows buffered before writing the month file (each write rewrites the
                                 # whole file, ~20 s on the HDD for a full month, so write rarely)
FULLTEXT_THROTTLE_LIMIT = 0.01   # stop the run if more than this share of a batch is throttled
FULLTEXT_FULL_MIN_CHARS = 300    # body at least this long is completeness "full"
# Sent to publishers instead of a browser-style string, so they can see what is crawling.
FULLTEXT_USER_AGENT = "trendfinder"


@dataclass(frozen=True)
class RateLimit:
    """Requests-per-second ceiling for a single upstream host."""

    rps: float
    burst: int = 1


@dataclass(frozen=True)
class PipelineConfig:
    data_root: Path
    screener_root: Path
    fred_api_key: str | None
    pg_dsn: str | None
    screener_sessionid: str | None = None
    daily_start: date = DAILY_START
    minute_start: date = MINUTE_START
    end: date = field(default_factory=date.today)
    news_backfill_years: int = NEWS_BACKFILL_YEARS
    rate_limits: dict[str, RateLimit] = field(
        default_factory=lambda: {
            "upstox": RateLimit(rps=3.0),
            "gdelt": RateLimit(rps=0.11),      # GDELT throttles hard; ~1 request / 9s
            "fred": RateLimit(rps=8.0),
            "worldbank": RateLimit(rps=5.0),
            "nse": RateLimit(rps=1.0),
            "niftyindices": RateLimit(rps=1.0),
            "mfapi": RateLimit(rps=10.0),       # community-run free API; no published limit, just asks callers to cache
            "yfinance": RateLimit(rps=1.5),
            "assets": RateLimit(rps=2.0),
            "screener": RateLimit(rps=0.22),   # be gentle: ~1 request / 4.5s
            "et_news": RateLimit(rps=FULLTEXT_ET_RPS),  # full-text article pages (Economic Times)
        }
    )
    http_retries: int = 5
    http_timeout: float = 40.0

    # ---- derived paths -------------------------------------------------
    @property
    def reference_dir(self) -> Path:
        return self.data_root / "reference"

    @property
    def market_dir(self) -> Path:
        return self.data_root / "market"

    @property
    def corp_actions_dir(self) -> Path:
        return self.data_root / "corporate_actions"

    @property
    def fundamentals_dir(self) -> Path:
        return self.data_root / "fundamentals"

    @property
    def macro_dir(self) -> Path:
        return self.data_root / "macro"

    @property
    def news_dir(self) -> Path:
        return self.data_root / "news"

    @property
    def manifest_dir(self) -> Path:
        return self.data_root / "_manifests"

    @property
    def log_dir(self) -> Path:
        return self.data_root / "_logs"

    @property
    def report_dir(self) -> Path:
        return self.data_root / "_reports"

    @property
    def screener_parquet_dir(self) -> Path:
        return self.fundamentals_dir / "screener"

    @property
    def shareholding_dir(self) -> Path:
        return self.fundamentals_dir / "shareholding"

    def ohlcv_dir(self, interval: str) -> Path:
        return self.market_dir / f"ohlcv_{interval}"

    def ensure_dirs(self) -> None:
        for p in (
            self.reference_dir,
            self.market_dir,
            self.corp_actions_dir,
            self.fundamentals_dir,
            self.macro_dir,
            self.news_dir,
            self.manifest_dir,
            self.log_dir,
            self.report_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)


def load_config() -> PipelineConfig:
    root = Path(os.environ.get("STOCK_DATA_ROOT", str(DEFAULT_DATA_ROOT))).expanduser()
    screener = Path(
        os.environ.get("SCREENER_DATA_ROOT", str(DEFAULT_SCREENER_ROOT))
    ).expanduser()
    cfg = PipelineConfig(
        data_root=root,
        screener_root=screener,
        fred_api_key=os.environ.get("FRED_API_KEY") or None,
        pg_dsn=os.environ.get("PG_DSN") or None,
        screener_sessionid=os.environ.get("SCREENER_SESSIONID") or None,
    )
    return cfg
