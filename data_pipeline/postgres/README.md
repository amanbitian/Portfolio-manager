# Phase 2 - Load the Parquet lake into PostgreSQL

See [`../DATA.md`](../DATA.md) for what data is available to load, [`../DATAFLOW.md`](../DATAFLOW.md)
for pipeline mechanics, and [`../README.md`](../README.md) for the ingestion commands.

Nothing here runs automatically. Do it once the Parquet lake under
`F:\quants project\stock Data` is populated.

## 1. Install PostgreSQL 17 (Windows)

Download the EDB installer: <https://www.postgresql.org/download/windows/>
("PostgreSQL 17" -> installer). During setup keep the default port `5432` and set a
password for the `postgres` superuser.

Optional but recommended for the ~325M-row minute table - **TimescaleDB**:
<https://docs.timescale.com/self-hosted/latest/install/installation-windows/>.
`schema.sql` auto-detects the extension and turns `market.ohlcv_1min` into a hypertable
if it is present; without it the table is still monthly range-partitioned.

## 2. Create the database and schema

```powershell
$env:PGPASSWORD = "<postgres password>"
& "C:\Program Files\PostgreSQL\17\bin\createdb.exe" -U postgres quant
& "C:\Program Files\PostgreSQL\17\bin\psql.exe" -U postgres -d quant -f data_pipeline\postgres\schema.sql
```

## 3. Load

```powershell
$env:PG_DSN = "postgresql://postgres:<password>@localhost:5432/quant"
py -3 -m pip install "psycopg[binary]"

py -3 -m data_pipeline.postgres.load universe
py -3 -m data_pipeline.postgres.load market_1day
py -3 -m data_pipeline.postgres.load corp_actions fundamentals macro news
py -3 -m data_pipeline.postgres.load market_1min      # long-running
```

Every loader is idempotent (`COPY` into a temp table -> `INSERT ... ON CONFLICT DO
UPDATE`), so re-running after adding more Parquet is safe.

## Point-in-time columns

`fundamentals.statements`, `macro.observations` and `news.articles` carry
`publication_date` / `available_at` / `realtime_start` per `architecture.md`. yfinance
does not give filing dates, so `publication_date` is left NULL for now - populate it
later from an NSE filings pipeline, then derive `available_at`.
