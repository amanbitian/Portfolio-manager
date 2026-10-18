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

Portfolio proposals accept cash plus stock quantities, purchase lots, or a dated
transaction ledger; mutual-fund holdings may be fractional. They show fresh target
weights, rounded trades, fees, and post-rounding checks. Failed checks block saving as
an accepted proposal. Local JSON manifests plus Parquet tables preserve decisions for
comparison and never submit orders. Risk limits apply at decisions; they are not
guarantees between them.

## Purpose

This project is a local quantitative research workbench spanning portfolio allocation,
strategy research, paper bookkeeping, and theoretical options analytics. The core
design principle is separation of responsibilities:

```text
market data -> features -> forecasts -> deterministic optimizer -> backtest -> evaluation
manual option assumptions -> deterministic pricing/Greeks -> payoff and scenario analysis
loaded prices/NAVs -> validated ledger -> paper-account valuation
```

The ML layer predicts expected return, risk, market regime, and confidence. The
optimizer converts those forecasts into feasible portfolio weights under explicit
constraints. The backtester determines whether the allocation process survives costs,
turnover, and out-of-sample validation. Options math and paper-account bookkeeping are
separate deterministic paths; neither feeds or bypasses the optimizer.

This is intentionally not an end-to-end neural network that emits portfolio weights directly from historical prices.

> **Data layer status:** the "raw data lake" and "point-in-time validation" boxes below
> are partially built already, in the separate `data_pipeline/` package - see
> [`data_pipeline/DATA.md`](data_pipeline/DATA.md) for what is ingested. Daily equity
> prices are available to the main app, and the Paper Trading workspace also reads the
> generated mutual-fund catalog and individual NAV histories. Fundamentals, corporate
> actions, macro, and news are not yet model inputs because their point-in-time use is
> not fully enforced. Options Lab has no option-chain source; its inputs are manual.

## Current Scope

The application is an interactive Streamlit dashboard with local Python engines. Its
synthetic and upload modes require no external data account; the optional real-data
mode reads the separately built local data lake. The implemented scope is:

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
- Purchase books and a validated transaction ledger with FIFO realized P&L, cash, XIRR, and Modified Dietz reporting.
- A persistent paper account with immediate fills at the latest loaded stock close or mutual-fund NAV.
- A price-only Research Agent, validated Strategy Copilot DSL, experiment registry, and robustness checks.
- A theoretical European-options lab with editable strategy templates, Black-Scholes premiums/Greeks, payoff, hedge, and scenario analytics.

## Implemented And Target Architecture

```text
                         LOCAL STREAMLIT APPLICATION
                                     |
       +-----------------------------+-----------------------------+
       |                             |                             |
       v                             v                             v
PORTFOLIO RESEARCH              PAPER ACCOUNT                 OPTIONS LAB
       |                             |                             |
prices -> features              prices / NAVs                manual assumptions
       |                             |                             |
alpha + risk + regime           validated ledger             Black-Scholes engine
       |                             |                             |
forecast validation             FIFO + cash checks           legs + Greeks + payoff
       |                             |                             |
constrained optimizer           immediate paper fills        scenarios + hedge units
       |                             |
proposal / walk-forward         holdings + P&L
backtest
       |
Research Agent / Strategy Copilot / Research Registry

Target additions: point-in-time fundamental/news joins, historical option chains and
volatility surfaces, realistic fill/margin simulation, and finally a broker adapter
behind explicit risk controls. None of those target additions is implemented today.
```

## Data Layer

The portfolio/research path consumes a daily price matrix from synthetic generation,
uploaded CSV, or `market/ohlcv_1day` in the local data lake. The CSV loader rejects
non-positive prices, duplicate normalized symbols, and the reserved `CASH` symbol. It
allows only short forward-filled gaps inside a symbol's history before dropping the
asset. A symbol whose data starts late or ends early is retained with
`eligible_from`/`eligible_through` frame metadata and a finite placeholder outside the
real span that downstream consumers explicitly mask; see `data.clean_gappy_columns`
and `decisions.MIN_LISTING_HISTORY_DAYS`.

Paper Trading has its own loader. Real-data mode reads a recent window for the entire
current equity universe, an optional generated mutual-fund catalog from
`reference/mutual_fund_catalog.parquet`, and individual NAV histories from
`fundamentals/mutual_funds/nav/<scheme_code>.parquet` on demand. Synthetic and upload
modes provide stocks only. A paper order uses the latest loaded observation; there is
no streaming quote service.

Options Lab does not consume these price stores. Spot, volatility, rates, dividend
yield, expiry, strikes, and quantities are direct user assumptions. No option chain or
historical option dataset exists in the current application.

Fundamental, macro, and news datasets exist in the ingestion package but are not joined
to the research engines. A production point-in-time layer still needs these fields:

- `observation_date`: the period the data describes.
- `publication_date`: when the source published it.
- `available_at`: when the strategy may legally use it.
- `ingested_at`: when the platform received it.

The relevant implemented local storage is:

```text
<data-root>/
  reference/universe.parquet
  reference/mutual_fund_catalog.parquet
  market/ohlcv_1day/...
  fundamentals/mutual_funds/nav/...
  fundamentals/...
  macro/...
  news/...

<repository>/
  .portfolio-cache/       disposable content-addressed feature/model tables
  experiments/            durable artifact manifests, tables, shared price inputs,
                          ledgers, purchase portfolios, and the paper account
```

## Core Modules

```text
src/ml_portfolio/
  data.py          data loading, synthetic data generation, CSV normalization
  market_data.py   local data-lake equity, universe, mutual-fund, and corporate-action readers
  features.py      leak-aware feature construction
  models.py        return forecasting and risk/regime estimation
  decisions.py     shared historical and latest-decision orchestration
  optimizers.py    portfolio construction and constraints
  convex.py        CVXPY large-universe/conic solver backend
  execution.py     self-financing transaction-cost accounting
  portfolio.py     holdings-to-trades proposal and snapshot workflow
  holdings.py      purchase-lot validation, aggregation, valuation, and persistence
  ledger.py        transaction validation, FIFO, cash, XIRR/Dietz, and paper-account persistence
  options.py       Black-Scholes pricing/Greeks, multi-leg payoff, summary, and scenarios
  research_agent.py price-led company/peer briefs and hypothesis handoff
  research_registry.py experiment metadata (question/hypothesis/status/conclusion) and cross-run listing
  robustness.py    cost stress, sub-periods, parameter perturbation, randomized placebo, bootstrap CI
  backtest.py      walk-forward simulation and metrics
  strategy_copilot.py validated strategy DSL and template backtester
  reporting.py     chart/table helpers for the app
  storage.py       atomic JSON-manifest/Parquet artifacts and content-addressed price storage
app.py             Streamlit interface
data_pipeline/      external ingestion and catalog construction; not imported into model backtests
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
- Fundamental or richer statistical factor-risk models beyond the current PCA fallback.
- Black-Litterman posterior return layer.
- Hidden Markov or clustering-based regime model.
- Transformer and reinforcement-learning allocators as challenger models only.

## Optimizer Design

The optimizer owns feasibility. Forecasts are inputs, not commands.

The current optimizer supports:

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

Artifact manifests reference compressed Parquet tables and shared input-price
artifacts. The store contains proposals, research comparisons/reports, strategy runs
and sweeps, robustness reports, purchase portfolios, transaction ledgers, and one
fixed-key paper account. Proposal browsing reads only valuation metadata and trade
columns. Legacy JSON snapshots remain readable. The Research Registry indexes only
research kinds, not operational holdings/ledger/paper artifacts. No database service
is required. Large UI tables use filtered pages; charts summarize small positions
while full data remains available for export.

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

## Options Analytics Design

`options.py` is a pure deterministic engine with no Streamlit or market-data
dependency. `OptionAssumptions` carries spot, time, volatility, rates, dividend yield,
lot size, and strategy-lot count. `black_scholes` prices European calls/puts and
returns delta, gamma, theta per calendar day, and vega per one volatility point.

Strategy templates produce an editable normalized leg table (`side`, `option_type`,
`strike`, `quantity`). Current templates are Long Straddle, Long Strangle, Iron
Condor, Call Butterfly, Batman/double butterfly, and Custom. The engine validates the
table, signs/scales each leg, aggregates premiums and Greeks, samples the exact expiry
payoff including a zero underlying boundary, interpolates breakevens, and detects
unbounded high-price tails. Probability of profit and expected P&L integrate the
payoff over the same risk-neutral lognormal distribution implied by the assumptions.
Scenario P&L reprices every leg at a chosen remaining life across fixed spot and
absolute-IV shocks relative to entry model value.

This is intentionally an analytics sandbox, not an options trading system. It has no
observed premiums, IV solver, smile/skew/surface, American exercise, discrete dividends,
margin, costs, liquidity, assignment, or backtest. The delta hedge is a point-in-time
underlying-unit offset, not a futures contract-sizing or rehedging policy engine.

## Paper Trading Design

Paper Trading reuses `ledger.py` rather than maintaining a second accounting model.
Every deposit/buy/sell is appended through the same transaction validation, stable-id,
chronological cash, FIFO, and valuation functions used by saved ledgers. The account
has a fixed artifact key (`paper_trading_account`) so app restarts reopen one local
account instead of generating timestamped snapshots for each order.

The page deliberately bypasses allocation controls: it owns its own data selector,
exploration views, watchlist, order ticket, holdings, order history, funding, and reset.
Real-data mode loads a recent whole-universe stock frame and optionally a precomputed
mutual-fund catalog; per-scheme NAV histories are lazy-loaded. Orders fill immediately
at the latest loaded close/NAV and are persisted only after cash and position checks.
There are no pending orders, partial fills, spreads, slippage, fees, exchange sessions,
or broker APIs. Consequently it is ledger-backed practice, not execution simulation.

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
loaded quotes. The Portfolio workflow can value a mutual fund only when its NAV series
is part of that workflow's loaded matrix. Direct mutual-fund catalog/NAV browsing from
the separate pipeline is implemented only in Paper Trading.

The Streamlit app is organized around eight sidebar workspaces:

- Portfolio: enter whole-share holdings via Quantities, open lots via Purchases, or a full dated buy/sell/cash-flow ledger via Transactions; calculate a fresh proposal, review rounded trades, risk checks, validation, concentration/drift against an equal-weight reference, and save accepted local snapshots.
- Paper Trading: fund one persistent virtual account, browse/search stocks and optional curated mutual funds, maintain a watchlist, submit immediate latest-loaded-price paper orders, and inspect cash, holdings, FIFO realized trades, P&L, and order history.
- Options Lab: edit a template or custom multi-leg European-option structure and inspect theoretical premiums, payoff, breakevens, max risk/reward, Greeks, a delta hedge estimate, and spot/IV scenarios without loading an option chain or placing an order.
- Research: run all four methods on a common explicit evaluation period, compare metrics, equity curves, holdings, forecasts, validation, fallback behavior, costs, a per-position contribution-to-gross-return breakdown, and a rolling-return chart.
- Research Agent: generate a deterministic price-led symbol brief with peer comparisons, formula-based scores (exposed as data, not just implicit in code), a historical percentile-based scenario range, evidence notes, and a hypothesis that can be sent to the strategy tester. Reports can be saved and reopened from Saved Decisions; using a hypothesis auto-saves the report and carries its id into the resulting Strategy Copilot run.
- Strategy Copilot: compile a controlled prompt or template inputs into a validated JSON DSL, then run a repeatable equal-weight strategy backtest, with saved runs reopenable from Saved Decisions and traceable back to any research report that inspired them. The MVP supports momentum top-N, SMA trend/cross, and RSI oversold rules. A separate parameter-sweep mode runs a bounded grid over a chronological selection window (reporting every tested variant, not just the best) and reports a single held-out final-test result for the selected variant.
- Saved Decisions: inspect saved local proposals and compare their hypothetical buy-and-hold value against current prices when the original reference price basis still matches; also browse and reopen saved research comparisons and saved Strategy Copilot runs, flagging when the underlying data has changed since a run was saved.
- Research Registry: every saved proposal, research run, research report, Strategy Copilot run, and sweep in one filterable table, tagged (optionally, at save time or later) with a research question, hypothesis, status (`unlabeled`/`supported`/`falsified`/`inconclusive`), conclusion, and parent experiment id - the metadata layer, not a database, that later cross-run analysis would query. For a saved Strategy Copilot run, a "Run robustness checks" action reruns it under cost stress, chronological sub-periods, parameter perturbation, and a randomized-selection placebo, with a bootstrap confidence interval on Sharpe, and can write a transparent rule-based verdict straight into that run's status and conclusion instead of a manual edit.

The Research Agent and Strategy Copilot are the first local finance-AI integration
points. Today they are deterministic and price-only: Python calculates facts, scores,
signals, turnover, costs, and metrics. A future local LLM should sit above these
interfaces as a planner and explainer, while execution remains constrained to
validated schemas. This preserves the core integrity rule from the target design:
no arbitrary generated Python is executed during strategy tests.

## Roadmap

Implemented now:

1. Local Streamlit allocation/research workflows over synthetic, uploaded, or local-lake daily prices.
2. Atomic local artifacts, reproducibility hashes, saved comparisons/reports/strategies/sweeps, and a research registry.
3. Price-led Research Agent, validated Strategy Copilot DSL, and robustness/falsification checks for saved Copilot runs.
4. Purchase books, a full transaction ledger, and one persistent stock/mutual-fund paper account.
5. Assumption-driven European Options Lab with pricing, Greeks, payoff, hedge, and scenario analysis.

Open architecture work:

1. Point-in-time/availability-safe joins for fundamentals, historical index membership, macro, filings, and news.
2. Confidence reliability reports, multiple-testing correction, ML/research robustness checks, and DuckDB-backed cross-run queries.
3. Historical option chains, IV inversion and surfaces, realistic costs/margin/liquidity, and event-driven options backtesting.
4. More realistic paper fills, pending/partial orders, market calendars, fees, slippage, and explicit data-freshness monitoring.
5. Challenger alpha/risk models and local LLM/RAG explanation above deterministic schemas.
6. Only after those controls: a broker adapter with permissions, position/risk limits, idempotency, audit logs, and kill switches.

## Current Non-Goals

- Docker.
- Local or remote database setup.
- Live orders or exchange-grade paper execution.
- Broker integration.
- Minute-level execution optimization.
- Paid market-data connectors.
- Live option-chain analytics or historical options claims.
- Deep learning or reinforcement-learning allocators.
- Investment advice or production financial recommendations.

The goal is a trustworthy local research foundation, not an overfit trading demo.
