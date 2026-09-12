"""Phase 2: load the Parquet lake into PostgreSQL.

Prerequisites:
    py -3 -m pip install "psycopg[binary]"
    createdb quant           # or any database
    psql "$PG_DSN" -f data_pipeline/postgres/schema.sql
    set PG_DSN=postgresql://user:pass@localhost:5432/quant

Then:
    py -3 -m data_pipeline.postgres.load universe
    py -3 -m data_pipeline.postgres.load market_1day
    py -3 -m data_pipeline.postgres.load market_1min      # long
    py -3 -m data_pipeline.postgres.load corp_actions fundamentals macro news
    py -3 -m data_pipeline.postgres.load all

Strategy: stream each Parquet file into a TEMP table with COPY, then
``INSERT ... ON CONFLICT DO UPDATE`` into the target. Idempotent and resumable.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    import psycopg
except ImportError:  # pragma: no cover
    psycopg = None

from ..config import PipelineConfig, load_config

# target table -> (columns, conflict key)
TARGETS: dict[str, tuple[str, list[str], list[str]]] = {
    "universe": (
        "reference.universe",
        ["symbol", "isin", "name", "industry", "instrument_key",
         "yf_ticker", "upstox_tradingsymbol", "source", "ingested_at"],
        ["symbol"],
    ),
    "corp_actions": (
        "corp.actions",
        ["symbol", "ex_date", "action_type", "ratio", "amount",
         "split_adj_factor", "source"],
        ["symbol", "ex_date", "action_type"],
    ),
    "macro": (
        "macro.observations",
        ["series_id", "label", "date", "value", "unit", "source", "realtime_start"],
        ["series_id", "date", "source"],
    ),
    "news": (
        "news.articles",
        ["symbol", "url", "seendate", "title", "domain", "language",
         "sourcecountry", "query_company"],
        ["symbol", "url"],
    ),
}


def _connect() -> "psycopg.Connection":
    if psycopg is None:
        sys.exit("psycopg not installed: py -3 -m pip install \"psycopg[binary]\"")
    cfg = load_config()
    if not cfg.pg_dsn:
        sys.exit("set PG_DSN=postgresql://user:pass@host:5432/db")
    return psycopg.connect(cfg.pg_dsn)


def _upsert(conn: "psycopg.Connection", table: str, cols: list[str],
            key: list[str], df: pd.DataFrame) -> int:
    df = df.reindex(columns=cols)
    if df.empty:
        return 0
    updates = ", ".join(f"{c}=EXCLUDED.{c}" for c in cols if c not in key)
    collist = ", ".join(cols)
    with conn.cursor() as cur:
        cur.execute(f"CREATE TEMP TABLE _stg (LIKE {table} INCLUDING DEFAULTS) ON COMMIT DROP")
        buf = io.StringIO()
        df.to_csv(buf, index=False, header=False)
        buf.seek(0)
        with cur.copy(f"COPY _stg ({collist}) FROM STDIN WITH (FORMAT csv)") as cp:
            cp.write(buf.read())
        cur.execute(
            f"INSERT INTO {table} ({collist}) SELECT {collist} FROM _stg "
            f"ON CONFLICT ({', '.join(key)}) DO UPDATE SET {updates}"
        )
        n = cur.rowcount
    conn.commit()
    return n


def _iter_parquet(paths: Iterable[Path]) -> Iterable[pd.DataFrame]:
    for p in paths:
        yield pd.read_parquet(p)


def load_flat(cfg: PipelineConfig, name: str) -> None:
    table, cols, key = TARGETS[name]
    roots = {
        "universe": [cfg.reference_dir / "universe.parquet"],
        "corp_actions": sorted(cfg.corp_actions_dir.glob("*.parquet")),
        "macro": sorted(cfg.macro_dir.rglob("*.parquet")),
        "news": sorted(cfg.news_dir.rglob("*.parquet")),
    }[name]
    conn = _connect()
    total = 0
    for df in _iter_parquet(roots):
        total += _upsert(conn, table, cols, key, df)
    conn.close()
    print(f"{name}: upserted {total} rows into {table}")


def load_market(cfg: PipelineConfig, interval: str) -> None:
    table = f"market.ohlcv_{interval}"
    cols = ["symbol", "ts_utc", "ts_ist", "open", "high", "low", "close",
            "volume", "open_interest", "instrument_key"]
    conn = _connect()
    total = 0
    for p in sorted(cfg.ohlcv_dir(interval).rglob("*.parquet")):
        df = pd.read_parquet(p)
        total += _upsert(conn, table, cols, ["symbol", "ts_utc"], df)
    conn.close()
    print(f"market_{interval}: upserted {total} rows into {table}")


def load_fundamentals(cfg: PipelineConfig) -> None:
    conn = _connect()
    stmt_cols = ["symbol", "period_end", "freq", "statement", "line_item", "value", "retrieved_at"]
    ks_cols = ["symbol", "key", "value_str", "value_num", "retrieved_at"]
    total = 0
    for sub in ("income_stmt", "balance_sheet", "cash_flow"):
        for p in sorted((cfg.fundamentals_dir / sub).glob("*.parquet")):
            total += _upsert(
                conn, "fundamentals.statements", stmt_cols,
                ["symbol", "period_end", "freq", "statement", "line_item"],
                pd.read_parquet(p),
            )
    for p in sorted((cfg.fundamentals_dir / "key_stats").glob("*.parquet")):
        total += _upsert(
            conn, "fundamentals.key_stats", ks_cols, ["symbol", "key"],
            pd.read_parquet(p),
        )
    conn.close()
    print(f"fundamentals: upserted {total} rows")


def main(argv: list[str] | None = None) -> int:
    args = argv or sys.argv[1:]
    if not args:
        args = ["all"]
    cfg = load_config()
    jobs = args if args != ["all"] else [
        "universe", "market_1day", "corp_actions", "fundamentals", "macro", "news"
    ]
    for job in jobs:
        if job in ("universe", "corp_actions", "macro", "news"):
            load_flat(cfg, job)
        elif job == "market_1day":
            load_market(cfg, "1day")
        elif job == "market_1min":
            load_market(cfg, "1min")
        elif job == "fundamentals":
            load_fundamentals(cfg)
        else:
            print(f"unknown job: {job}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
