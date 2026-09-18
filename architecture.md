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

Version 1 expects daily adjusted close prices. The CSV loader rejects non-positive prices, duplicate normalized symbols, and the reserved `CASH` symbol. It allows only short forward-filled gaps inside a symbol's history before dropping the asset; a symbol whose data starts late (a mid-window listing) or ends early (a delisting) in the window is kept, with its first/last real date recorded on the frame (`eligible_from` / `eligible_through`) and a flat placeholder outside that span that no consumer treats as history - see `data.clean_gappy_columns` and `decisions.MIN_LISTING_HISTORY_DAYS`. Later versions should add a point-in-time storage layer with these timestamp fields:

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
  decisions.py     shared historical and latest-decision orchestration
  optimizers.py    portfolio construction and constraints
  execution.py     self-financing transaction-cost accounting
  portfolio.py     holdings-to-trades proposal and snapshot workflow
  research_agent.py price-led company/peer briefs and hypothesis handoff
  backtest.py      walk-forward simulation and metrics
  strategy_copilot.py validated strategy DSL and template backtester
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
- Optional projected annual volatility ceiling.
- Optional sector caps.
- One-way turnover budget from current holdings.
- Residual cash when position caps make full risky investment infeasible.
- Transaction-cost and minimum expected-benefit assumptions.

If position caps make full risky investment infeasible, the allocator leaves the residual in cash. If the ML expected-return signal is weak relative to modeled risk, it may also choose to hold extra cash.

Above 250 risky assets, automatic solver selection uses CVXPY: OSQP for quadratic
programs and Clarabel when a volatility ceiling introduces a conic constraint.
Both optimize costs, turnover, and risk limits jointly, with post-solve checks.
SLSQP remains the automatic choice for small universes with convex retry on failure;
the convex backend can also be selected explicitly. Covariance above 400 assets uses a
PCA statistical factor-plus-specific-risk model rather than a directly estimated sample
covariance (which would be singular/noisy at that asset count relative to the training
window's length), so optimality is relative to that approximate risk model - it
preserves meaningful cross-asset correlation from a handful of common factors, but is
not the same as a fully estimated correlation matrix.

## Local Performance And Storage

Strategy comparisons share read-only price/return/feature tables and use a
bounded thread pool. A locked LRU cache reuses risk estimates with a 64 MiB target
budget (at least one matrix); the ML strategy owns the prediction memo. Each strategy
owns its holdings and result state. Numerical
library threads are limited around comparisons to avoid nested thread oversubscription.
Daily ridge sufficient statistics let chronological training windows be fitted using
small matrix differences. Only fully matured labels enter a fit. Predictions for
repeated validation origins are reused within the context.

Content-addressed Parquet caches persist returns, features, targets, and ridge
statistics across sessions. Keys include price values, dates, symbols, feature schema,
and horizon. Cache corruption or unavailable storage causes recomputation. Writes
publish atomically and the disposable cache has a 2 GiB disk budget. No pickle files
are used by this application persistence layer. Streamlit separately maintains a
bounded in-memory comparison cache. Backtest drift uses interval price relatives,
with fees applied once to the first day's holding-period return.

New experiment manifests reference compressed Parquet tables and shared input-price
artifacts. Proposal browsing reads only valuation metadata and trade columns. Legacy
JSON snapshots remain readable. Research runs can be saved and loaded through the
same artifact layer. No database service is required. Large UI tables use filtered
pages; charts summarize small positions while full data remains available for export.

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
- Raw and validation-adjusted forecasts.
- Chronological validation summary.
- Regime.
- Target weights.
- Turnover and constraint checks.
- Actual self-financing cost.
- Effective method and fallback reason.

The test should compare ML-conditioned allocation against simple baselines. The next model level should only be promoted when it improves out-of-sample results after costs.

## Dashboard Views

Purchase bookkeeping is isolated in `holdings.py`. Open lots retain instrument type,
purchase date, quantity, and unit cost. Aggregation computes weighted cost and current
valuation independently of forecasts and optimizer feasibility. The proposal engine
checks that lots reconcile with the supplied quantities and instrument types, values
positions at the latest quote, and uses per-instrument rounding (one stock share or
0.000001 fund units). It never casts fractional current holdings or trade deltas to int.
Original lots are attached as purchase metadata, not rewritten as post-trade tax lots.

Named purchase portfolios use the existing atomic manifest/Parquet store with kind
`purchase_portfolio`; existing proposal and research readers skip these manifests.
Portfolios can be saved without an allocation decision, reopened, and revalued against
loaded quotes. The app does not yet ingest mutual-fund NAVs from the separate pipeline.

The Streamlit app is organized around five workflows:

- Portfolio: enter whole-share holdings via Quantities, open lots via Purchases, or a full dated buy/sell/cash-flow ledger via Transactions; calculate a fresh proposal, review rounded trades, risk checks, validation, concentration/drift against an equal-weight reference, and save accepted local snapshots.
- Research: run all four methods on a common explicit evaluation period, compare metrics, equity curves, holdings, forecasts, validation, fallback behavior, costs, a per-position contribution-to-gross-return breakdown, and a rolling-return chart.
- Research Agent: generate a deterministic price-led symbol brief with peer comparisons, formula-based scores (exposed as data, not just implicit in code), a historical percentile-based scenario range, evidence notes, and a hypothesis that can be sent to the strategy tester. Reports can be saved and reopened from Saved Decisions; using a hypothesis auto-saves the report and carries its id into the resulting Strategy Copilot run.
- Strategy Copilot: compile a controlled prompt or template inputs into a validated JSON DSL, then run a repeatable equal-weight strategy backtest, with saved runs reopenable from Saved Decisions and traceable back to any research report that inspired them. The MVP supports momentum top-N, SMA trend/cross, and RSI oversold rules. A separate parameter-sweep mode runs a bounded grid over a chronological selection window (reporting every tested variant, not just the best) and reports a single held-out final-test result for the selected variant.
- Saved Decisions: inspect saved local proposals and compare their hypothetical buy-and-hold value against current prices when the original reference price basis still matches; also browse and reopen saved research comparisons and saved Strategy Copilot runs, flagging when the underlying data has changed since a run was saved.

The Research Agent and Strategy Copilot are the first local finance-AI integration
points. Today they are deterministic and price-only: Python calculates facts, scores,
signals, turnover, costs, and metrics. A future local LLM should sit above these
interfaces as a planner and explainer, while execution remains constrained to
validated schemas. This preserves the core integrity rule from the target design:
no arbitrary generated Python is executed during strategy tests.

## Roadmap

1. Version 1: Local Streamlit MVP with synthetic/uploaded daily data.
2. Version 2: Local saved proposal artifacts and reproducibility checks.
3. Implemented: Saved backtest artifacts and a local Parquet experiment store.
4. Next: Stress tests, confidence reliability reports, and cross-run experiment queries.
5. Implemented: Price-led Research Agent and validated Strategy Copilot MVP.
6. Version 6: Point-in-time DuckDB/Parquet data layer.
7. Version 7: Local LLM/RAG research synthesis over fundamentals, filings, news, and macro data.
8. Version 8: LightGBM alpha model, richer feature store, and Black-Litterman confidence blending.
9. Version 9: Paper-trading interface and monitoring.
10. Version 10: Broker execution adapter with explicit kill switches.

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
