# ML Portfolio Allocation Architecture

## Decision Engine Upgrade

Historical rebalances and fresh next-session proposals use one shared decision path.
SciPy constrained optimization enforces position, cash, annual volatility, sector,
and one-way turnover limits. Incompatible constraints produce an explicit error.
ML objectives include trading fees and a minimum expected benefit per traded unit.

ML eligibility uses chronological validation blocks inside each training window,
with label purging at every validation origin. Predictions must beat a historical
mean forecast in aggregate and in a majority of at least three blocks. Otherwise
the decision falls back to constrained equal weight. Confidence is measured error
improvement, not a probability of success.

Backtests use an explicit common evaluation period with at least 63 daily returns.
The initial equity anchor is excluded from return statistics. Gross and net paths
follow identical decisions; their terminal difference measures compounded fee drag.
Cash earns zero; Sharpe uses a zero risk-free rate.

Portfolio proposals accept cash and whole-share holdings, show fresh target weights,
rounded trades, fees, and post-rounding checks. Failed checks block saving as an
accepted proposal. Local JSON snapshots preserve decisions for comparison and never
submit orders. Risk limits apply at decisions; they are not guarantees between them.

## Purpose

This project is a research workbench for portfolio allocation. The core design principle is separation of responsibilities:

```text
data -> features -> forecasts -> deterministic optimizer -> backtest -> evaluation
```

The ML layer predicts expected return, risk, market regime, and confidence. The optimizer converts those forecasts into feasible portfolio weights under explicit constraints. The backtester determines whether the allocation process survives costs, turnover, and out-of-sample validation.

This is intentionally not an end-to-end neural network that emits portfolio weights directly from historical prices.

> **Data layer status:** the "raw data lake" and "point-in-time validation" boxes below
> are partially built already, in the separate `data_pipeline/` package - see
> [`data_pipeline/DATA.md`](data_pipeline/DATA.md) for what's live (prices,
> fundamentals, corp actions, macro, news) versus still-synthetic in the app below.

## Version 1 Scope

The first application is an interactive Streamlit dashboard with a local Python research engine. It does not require Docker, a database, or any external data account. It focuses on daily data and implements a practical MVP:

- Synthetic demo market data with regime behavior.
- Optional CSV upload for user-supplied price history.
- Feature generation using only historical information available at each date.
- Baseline allocators: equal weight, inverse volatility, minimum variance with full covariance.
- ML-conditioned allocator using a ridge-style return forecast and shrinkage covariance.
- Long-only constrained weights with max position, cash buffer, and transaction-cost assumptions.
- Walk-forward backtesting with train, rebalance, and holding windows.
- Explicit label cutoffs so a training sample is only used after its full forward-return horizon is known.
- Buy-and-hold portfolio drift between rebalance dates, with transaction costs applied to traded risky assets.
- Portfolio metrics: CAGR, volatility, Sharpe, max drawdown, turnover, cost drag, and final equity.
- Visual dashboard for equity curves, drawdowns, target weights, forecast signals, and run diagnostics.

## Target End-State Architecture

```text
data sources
    |
    v
raw data lake
    |
    v
point-in-time validation
    |
    v
feature store
    |
    +--> alpha model --------+
    |                        |
    +--> risk model ---------+--> forecast combiner --> constrained optimizer
    |                        |                              |
    +--> regime model -------+                              v
                                                        target portfolio
                                                             |
                                                             v
                                                    backtest / execution
                                                             |
                                                             v
                                                       evaluation gate
                                                             |
                                                             v
                                                    paper/live monitoring
```

## Data Layer

Version 1 expects daily adjusted close prices. The CSV loader rejects non-positive prices, duplicate normalized symbols, and the reserved `CASH` symbol. It allows only short forward-filled gaps before dropping incomplete assets. Later versions should add a point-in-time storage layer with these timestamp fields:

- `observation_date`: the period the data describes.
- `publication_date`: when the source published it.
- `available_at`: when the strategy may legally use it.
- `ingested_at`: when the platform received it.

The intended storage layout is:

```text
data/
  raw/
    market/
    fundamentals/
    macro/
    news/
  cleaned/
  point_in_time/
  features/
  predictions/
  portfolios/
  backtests/
  experiments/
```

## Core Modules

```text
src/ml_portfolio/
  data.py          data loading, synthetic data generation, CSV normalization
  features.py      leak-aware feature construction
  models.py        return forecasting and risk/regime estimation
  optimizers.py    portfolio construction and constraints
  backtest.py      walk-forward simulation and metrics
  reporting.py     chart/table helpers for the app
app.py             Streamlit interface
```

## Forecasting Design

The first ML layer is deliberately simple:

- Rolling momentum and volatility features.
- Cross-sectional standardization on each rebalance date.
- Ridge-style linear forecast for forward returns.
- Regime detection from trailing market return and realized volatility.
- Confidence scaling from chronological out-of-sample RMSE improvement versus a historical-mean forecast.

This keeps the app explainable while preserving the production pattern needed for stronger models later.

Future model candidates:

- LightGBM or XGBoost alpha model.
- Shrinkage covariance and factor risk model.
- Black-Litterman posterior return layer.
- Hidden Markov or clustering-based regime model.
- Transformer and reinforcement-learning allocators as challenger models only.

## Optimizer Design

The optimizer owns feasibility. Forecasts are inputs, not commands.

Version 1 supports:

- Equal weight.
- Inverse volatility.
- Minimum variance allocation with the full shrinkage covariance matrix.
- ML mean-variance allocation with cash as the residual safe asset.

Shared constraints:

- Long-only weights.
- Per-asset maximum position size.
- Optional cash reserve.
- Residual cash when position caps make full risky investment infeasible.
- Risk-aversion and confidence controls that affect total risky exposure.
- Turnover measurement.
- Transaction-cost drag in backtests.

If position caps make full risky investment infeasible, the allocator leaves the residual in cash. If the ML expected-return signal is weak relative to modeled risk, it may also choose to hold extra cash.

## Backtesting Design

The backtester uses walk-forward validation:

```text
train on past window -> generate next allocation -> apply next holding-period returns
```

For a decision at date `t`, training labels are only included when their full forward-return horizon has ended before `t`. This prevents the common error where a model appears to train on past rows while the target column silently reaches into future prices.

After a target allocation is selected, the simulator lets asset weights drift with realized returns until the next rebalance. Transaction costs are charged on risky-asset turnover at the start of each holding period.

Every rebalance records:

- Training window.
- Rebalance date.
- Forecasts.
- Regime.
- Target weights.
- Turnover.
- Estimated cost.
- Estimated traded risky exposure.
- Portfolio return.

The test should compare ML-conditioned allocation against simple baselines. The next model level should only be promoted when it improves out-of-sample results after costs.

## Dashboard Views

The Streamlit app is organized around four views:

- Overview: headline metrics and equity curves.
- Allocation: latest forecast, confidence, regime, and target weights.
- Backtest: drawdown, return distribution, turnover, and cost diagnostics.
- Data: input preview, validation notes, and generated features.

## Roadmap

1. Version 1: Local Streamlit MVP with synthetic/uploaded daily data.
2. Version 2: Local saved experiment artifacts and reproducibility checks.
3. Version 3: Forecast diagnostics, stress tests, and validation reports.
4. Version 4: Persistent experiment store and saved backtest runs.
5. Version 5: Point-in-time DuckDB/Parquet data layer.
6. Version 6: LightGBM alpha model, richer feature store, and Black-Litterman confidence blending.
7. Version 7: Paper-trading interface and monitoring.
8. Version 8: Broker execution adapter with explicit kill switches.

## Non-Goals For Version 1

- Docker.
- Local or remote database setup.
- Live trading.
- Broker integration.
- Minute-level execution optimization.
- Paid market-data connectors.
- Deep learning or reinforcement-learning allocators.
- Investment advice or production financial recommendations.

The goal is a trustworthy local research foundation, not an overfit trading demo.
