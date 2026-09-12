# ML Portfolio Allocation

An interactive research workbench for testing portfolio allocation ideas with a hybrid architecture:

```text
ML forecasts returns, risk, regime, and confidence.
The optimizer converts those forecasts into constrained weights.
The backtester evaluates the strategy after costs.
```

## What This App Does

- Builds a daily-price research dataset from synthetic demo data or uploaded CSV data.
- Generates leak-aware momentum, volatility, drawdown, and relative-strength features.
- Runs baseline allocators and an ML-conditioned allocation model.
- Applies long-only constraints, position caps, cash buffers, and transaction costs.
- Performs walk-forward backtesting with label cutoffs that prevent future-return leakage.
- Simulates buy-and-hold weight drift between rebalance dates.
- Shows equity curves, drawdowns, final allocations, forecast diagnostics, and risk metrics.

## Project Structure

```text
.
  app.py
  architecture.md
  README.md
  requirements.txt
  tests/
    engine_checks.py
  src/
    ml_portfolio/
      __init__.py
      backtest.py
      data.py
      features.py
      models.py
      optimizers.py
      reporting.py
  data_pipeline/          real-data ingestion (prices, fundamentals, macro, news) - see below
  scripts/                scheduled-task / monitoring helpers for data_pipeline
```

## Data Pipeline

A separate package, `data_pipeline/`, ingests real market data (Upstox prices, Screener +
yfinance fundamentals, corporate actions, macro, historical + live news) into Parquet
under `F:\quants project\stock Data`. **The app above does not read it yet** - Version 1
still runs on synthetic/uploaded data only; wiring the two together is the next step.

| Doc | Purpose |
|---|---|
| [`data_pipeline/README.md`](data_pipeline/README.md) | Install + every ingestion command |
| [`data_pipeline/DATA.md`](data_pipeline/DATA.md) | **What data we have and what's missing** - every source, status, pending work |
| [`data_pipeline/DATAFLOW.md`](data_pipeline/DATAFLOW.md) | Pipeline mechanics: diagram, storage layout |
| [`data_pipeline/NEWS.md`](data_pipeline/NEWS.md) | News sources in depth + a running backfill progress log |
| [`data_pipeline/postgres/README.md`](data_pipeline/postgres/README.md) | Phase 2: loading the Parquet lake into PostgreSQL (not run yet) |

## Quick Start

Create a virtual environment if desired, then install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

The app starts with synthetic market data, so no external data source is required. Version 1 runs without Docker and without a database.

Run the local engine checks:

```bash
python tests/engine_checks.py
python tests/app_logic_checks.py
python tests/ui_checks.py
```

## CSV Input Format

You can upload either wide or long daily price data.

Use adjusted close prices where possible. The loader rejects non-positive prices, duplicate symbols after normalization, and the reserved `CASH` symbol. It only forward-fills short gaps of up to five rows.

Wide format:

```csv
date,RELIANCE,HDFCBANK,INFY,TCS,ICICIBANK
2020-01-01,100,100,100,100,100
2020-01-02,101,99,102,100.5,98
```

Long format:

```csv
date,symbol,close
2020-01-01,RELIANCE,100
2020-01-01,HDFCBANK,100
2020-01-02,RELIANCE,101
2020-01-02,HDFCBANK,99
```

## Research Workflow

The upgraded Portfolio view accepts cash and current holdings, calculates fresh
proposals, previews trades and risk checks, and saves decisions locally. Research
uses a common evaluation period and chronological forecast validation with an
explicit baseline fallback. No Docker, database, or broker connection is required.

1. Select or upload daily prices and choose a strategy.
2. Apply a training window, forecast horizon, position/cash limits, costs, and optional risk limits.
3. In Portfolio, enter whole-share holdings and available cash, then calculate a fresh proposal.
4. Review the validation gate, target and rounded weights, trades, fees, and post-trade checks.
5. Save accepted decisions locally; Saved Decisions shows hypothetical buy-and-hold values.
6. In Research, choose a common evaluation period and rebalance interval, then run all four strategies.

Evaluation requires 63 daily returns after the selected training history. Changing
training length does not move an explicitly selected test start. Forecast and rebalance
horizons are independent controls. A short training history may not support three
validation blocks; this triggers a documented equal-weight fallback.

Local snapshots under `experiments/` contain the input prices and their hash, settings,
forecasts, validation, holdings, trades, costs, and risk checks. Saving does not trade.
Saved comparisons are blocked if historical reference prices no longer match (for
example, after changing a demo seed). Reference-price trades are estimates, not fills.

## Implemented Methods

- Equal weight.
- Inverse volatility.
- Minimum variance allocator using the full shrinkage covariance matrix.
- ML mean-variance allocator using a local ridge-style return forecast and shrinkage covariance.

The ML allocator can hold extra cash when validated forecasts do not compensate for
modeled risk and trading costs. Minimum variance keeps the maximum feasible risky
budget before minimizing risk plus costs. Equal-weight and inverse-volatility methods
minimize deviation from their capped reference allocations with a fee penalty. All
methods obey position, cash, sector, projected annual volatility, and one-way turnover
limits. Limits are checked at decisions; realized volatility and drift can exceed them.

ML confidence is relative RMSE improvement over a historical-mean forecast on matched
forward horizons, not a probability. At least three matured, non-overlapping validation
blocks, positive aggregate improvement, a majority of block wins, and positive mean
rank correlation are required. No future evaluation outcome controls earlier decisions.

Backtest cash earns zero and Sharpe assumes a zero risk-free rate. Undefined Sharpe
values are shown as N/A. Fees paid are summed in initial-capital units; cost drag is
the terminal gross-minus-net difference for the same sequence of allocation decisions.
Fees are self-financing against post-fee target holdings. The optimizer uses a linear
pre-trade cost estimate; realized fees and rounded proposals are checked separately.

## Interpretation Notes

This app is a research tool. Synthetic demo data is useful for validating mechanics, not for making trades. Uploaded historical data can still suffer from survivorship bias, missing corporate actions, look-ahead bias, and unrealistic execution assumptions.

No output from this app should be treated as financial advice.

## Next Steps

- Add configurable risk-free/cash yields and fill/slippage models.
- Extend validation to multiple market periods and confidence reliability plots.
- Add stress tests for cost assumptions, rebalance frequency, and market regimes.
- Add DuckDB and Parquet persistence after the local file workflow is stable.
- Add point-in-time fundamentals, index membership, and stronger ML models after the backtest remains stable on clean price data.
