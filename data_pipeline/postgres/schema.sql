-- ML Portfolio Allocator - raw data warehouse schema (Phase 2).
--
-- Apply with:  psql "$PG_DSN" -f data_pipeline/postgres/schema.sql
-- TimescaleDB is optional; if the extension is available the minute table becomes a
-- hypertable, otherwise it stays a native monthly-range-partitioned table.

CREATE SCHEMA IF NOT EXISTS reference;
CREATE SCHEMA IF NOT EXISTS market;
CREATE SCHEMA IF NOT EXISTS corp;
CREATE SCHEMA IF NOT EXISTS fundamentals;
CREATE SCHEMA IF NOT EXISTS macro;
CREATE SCHEMA IF NOT EXISTS news;

-- ---------------------------------------------------------------------------
-- reference
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS reference.universe (
    symbol               text PRIMARY KEY,
    isin                 text NOT NULL,
    name                 text,
    industry             text,
    instrument_key       text,
    yf_ticker            text,
    upstox_tradingsymbol text,
    source               text,
    ingested_at          timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_universe_isin ON reference.universe (isin);

-- ---------------------------------------------------------------------------
-- market OHLCV (raw candles as returned by Upstox - already split/bonus adjusted)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS market.ohlcv_1day (
    symbol         text        NOT NULL REFERENCES reference.universe (symbol),
    ts_utc         timestamptz NOT NULL,
    ts_ist         timestamptz NOT NULL,
    open           double precision,
    high           double precision,
    low            double precision,
    close          double precision,
    volume         bigint,
    open_interest  bigint,
    instrument_key text,
    ingested_at    timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol, ts_utc)
);
CREATE INDEX IF NOT EXISTS ix_ohlcv_1day_ts ON market.ohlcv_1day (ts_utc);

-- Minute bars: ~325M rows for the full universe. Range-partitioned by month.
CREATE TABLE IF NOT EXISTS market.ohlcv_1min (
    symbol         text        NOT NULL,
    ts_utc         timestamptz NOT NULL,
    ts_ist         timestamptz NOT NULL,
    open           double precision,
    high           double precision,
    low            double precision,
    close          double precision,
    volume         bigint,
    open_interest  bigint,
    instrument_key text,
    ingested_at    timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol, ts_utc)
) PARTITION BY RANGE (ts_utc);

-- Create monthly partitions 2022-01 .. 2027-12 (extend as needed).
DO $$
DECLARE
    d date := date '2022-01-01';
BEGIN
    WHILE d < date '2028-01-01' LOOP
        EXECUTE format(
            'CREATE TABLE IF NOT EXISTS market.ohlcv_1min_%s PARTITION OF market.ohlcv_1min '
            'FOR VALUES FROM (%L) TO (%L)',
            to_char(d, 'YYYY_MM'), d, (d + interval '1 month')::date
        );
        d := (d + interval '1 month')::date;
    END LOOP;
END $$;

-- Optional: promote to a TimescaleDB hypertable when the extension exists.
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_available_extensions WHERE name = 'timescaledb') THEN
        CREATE EXTENSION IF NOT EXISTS timescaledb;
        PERFORM create_hypertable('market.ohlcv_1min', 'ts_utc',
                                  chunk_time_interval => interval '1 month',
                                  migrate_data => true, if_not_exists => true);
    END IF;
END $$;

-- ---------------------------------------------------------------------------
-- corporate actions
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS corp.actions (
    symbol           text NOT NULL REFERENCES reference.universe (symbol),
    ex_date          date NOT NULL,
    action_type      text NOT NULL,          -- dividend | split | bonus | rights
    ratio            double precision,       -- split/bonus ratio
    amount           double precision,       -- dividend per share
    split_adj_factor double precision,
    source           text,
    ingested_at      timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol, ex_date, action_type)
);

-- ---------------------------------------------------------------------------
-- fundamentals (long form; point-in-time columns for leak-free training)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fundamentals.statements (
    symbol           text NOT NULL REFERENCES reference.universe (symbol),
    period_end       date NOT NULL,
    freq             text NOT NULL,          -- A | Q
    statement        text NOT NULL,          -- income_stmt | balance_sheet | cash_flow
    line_item        text NOT NULL,
    value            double precision,
    currency         text,
    publication_date date,                   -- filing date when known
    available_at     timestamptz,            -- when a strategy may use it
    retrieved_at     timestamptz,
    ingested_at      timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol, period_end, freq, statement, line_item)
);

CREATE TABLE IF NOT EXISTS fundamentals.key_stats (
    symbol       text NOT NULL REFERENCES reference.universe (symbol),
    key          text NOT NULL,
    value_str    text,
    value_num    double precision,
    retrieved_at timestamptz,
    ingested_at  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol, key)
);

-- ---------------------------------------------------------------------------
-- macro (long form; realtime_start captures the data vintage / ALFRED-style)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS macro.observations (
    series_id      text NOT NULL,
    label          text,
    date           date NOT NULL,
    value          double precision,
    unit           text,
    source         text NOT NULL,            -- worldbank | fred | rbi_manual
    realtime_start date,
    ingested_at    timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (series_id, date, source)
);

-- ---------------------------------------------------------------------------
-- news
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS news.articles (
    symbol         text NOT NULL REFERENCES reference.universe (symbol),
    url            text NOT NULL,
    seendate       timestamptz,
    title          text,
    domain         text,
    language       text,
    sourcecountry  text,
    query_company  text,
    ingested_at    timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol, url)
);
CREATE INDEX IF NOT EXISTS ix_news_symbol_date ON news.articles (symbol, seendate);
