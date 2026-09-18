# Project Update Backlog

Reviewed against the code and project docs on 2026-09-13.

This is the update plan for ML Portfolio Allocation, purchase portfolios, Research
Agent, and Strategy Copilot. Unchecked items are planned work, not implemented
capabilities. Performance investigations are proposals unless explicitly described
as observed in the current implementation.

## Scope And Priorities

- Keep the default app local, without Docker, a database service, or an LLM server.
- Data acquisition and pipeline changes belong to Claude/the data owner. Coordinate
  the handoff contracts below before changing that package.
- App integration, accounting, research logic, and performance belong to the app work.
- P0: correctness or integration prerequisite; complete before the dependent feature.
- P1: next functional or performance improvement.
- P2: expansion after the core workflows are reliable and measured.
- Deferred: optional infrastructure or capabilities requiring a separate decision.

Data-source coverage and ingestion status remain in [data_pipeline/DATA.md](data_pipeline/DATA.md).
Use [README.md](README.md) for current behavior and [architecture.md](architecture.md)
for design. This backlog does not replace those documents or duplicate their live counts.

## Existing Foundations

- [x] Local Streamlit application with synthetic prices and CSV uploads.
- [x] Shared allocation decisions, chronological ML validation, cost accounting,
  constrained proposals, and vectorized holding-period calculations.
- [x] Parallel strategy comparisons, shared preprocessing, bounded risk caches,
  and ridge sufficient statistics with prediction reuse.
- [x] CVXPY large-universe backend with OSQP/Clarabel and small-universe SLSQP.
- [x] Content-addressed Parquet feature caches, atomic writes, cache eviction,
  and JSON manifests referencing Parquet experiment tables.
- [x] Purchase-date/cost entry, fractional mutual-fund units, saved open-lot books,
  and unrealized gain calculations.
- [x] Price-based Research Agent and three-template Strategy Copilot MVP.
- [x] Table pagination, symbol filters, summarized allocations, and exports.

Current limits: the app does not read the separate data lake yet; purchase books are
open lots rather than a transaction ledger; Research Agent does not use an LLM or
document retrieval; covariance becomes diagonal above 400 assets. Existing speed
measurements cover selected workloads, not every feature or concurrent user load.

## Data: Flow And Storage

### D01 - App Data Contract And Local Loader [P0]

- [ ] Agree a versioned contract with the data owner for instrument IDs, dates,
  currency, units, raw/adjusted/total-return price basis, NAV, and source provenance.
- [ ] Add a configurable data-root setting and a read-only adapter for daily
  equity prices and fund NAVs. Keep synthetic and CSV modes available.
- [ ] Filter instruments, columns, and dates before materializing the price matrix;
  never load the full minute archive or all fund histories for a daily request.
- [ ] Return dataset version, coverage, quote timestamps, and exclusion reasons
  alongside values. Resolve a consistent snapshot if ingestion runs concurrently.

Done when: selected stocks and schemes load from fixture Parquet files, match an
equivalent CSV calculation, and missing roots or schema errors produce useful messages.
Ownership: data owner supplies the contract; app work implements the adapter.

### D02 - Instrument Identity And Fund Metadata [P0]

- [ ] Use stable IDs with symbol/name aliases; distinguish fund scheme, direct/regular
  plan, growth/distribution option, and active/closed status without merging histories.
- [ ] Carry instrument type, trading calendar, currency, sector/category, benchmark,
  quantity precision, and metadata effective dates into app calculations.
- [ ] Add searchable instrument selection; avoid rendering tens of thousands of
  fund names or treating every plan as an independent diversification opportunity.

Done when: renamed stocks and similarly named fund plans remain distinct and saved
holdings resolve consistently after metadata updates. Depends on D01.

### D03 - Availability, Freshness, And Point-In-Time Rules [P0]

- [ ] Preserve observation date, publication date, available-at time, ingestion time,
  and revisions. Join fundamentals, macro, and news using availability at the decision.
- [ ] Label estimated publication lags explicitly; do not treat estimates as verified
  historical availability or use crawl time as an article publication timestamp.
- [ ] Define calendar-aware missing/stale quote rules. Show exclusions and fill counts;
  prevent suspended stocks or stale NAVs from appearing tradable through forward fill.
- [ ] Support historical eligibility masks for IPOs, delistings, and index membership
  when supplied. Flag current-membership-only research until history is available.
- [ ] Define a separate valuation-only path for one instrument or short histories;
  keep each strategy's required training and evaluation history checks.

Done when: future releases/revisions cannot alter earlier decisions, stale holdings
are identified, and incomplete instruments are handled without silent universe changes.
Ownership: data owner supplies availability/quality metadata; app work enforces usage.

### D04 - Price Basis And Corporate-Action Handoff [P0]

- [ ] Separate analytical return series from executable quotes and purchase costs.
  Record adjustment conventions so dividends or splits are not counted twice.
- [ ] Consume versioned split, bonus, merger, demerger, and distribution events with
  stable event IDs. Define how units, cost basis, and cash change for each supported type.
- [ ] Require explicit handling for unsupported events and inconsistent price bases.

Done when: known-event fixtures reconcile units, cash, value, and returns without
artificial gains or duplicate adjustments. Depends on D01 and D02; feeds remain with
the data owner, accounting behavior with app work.

### D05 - Selective Storage And Incremental Updates [P1]

- [ ] Profile existing partition sizes and reads; tune dataset-specific partitioning,
  row groups, compression, and small-file compaction only where measurements justify it.
- [ ] Extend existing checkpoints with a published snapshot manifest: schema version,
  file checksums, covered dates, source version, and successful update timestamp.
- [ ] Track changed partitions and corrections. Recompute affected feature windows,
  including rolling lookbacks and newly matured forward labels, instead of all history.
- [ ] Preserve immutable raw inputs separately from corrected/derived views and make
  retrying an update idempotent.

Done when: appending one day and correcting one historical row both produce the same
derived output as a full rebuild, while reading and recomputing less data.
Coordinate pipeline changes with the data owner; depends on D01-D04.

### D06 - Cache And Experiment Lifecycle [P1]

- [ ] Extend current cache keys with all new dataset, adjustment, feature, and model
  versions; share dataset fingerprints instead of repeatedly hashing identical frames.
- [ ] Add cache hit/miss, disk usage, and rebuild controls with separate cache and
  saved-experiment lifecycles. Preserve accepted artifacts when clearing the cache.
- [ ] Add artifact schema migrations, integrity checks, portable backup/export, and
  restore verification including every referenced price/table file.
- [ ] Test concurrent reads, writes, eviction, interrupted saves, and disk-full errors;
  add coordination across processes if local workers share storage.
- [ ] Provide a reviewed cleanup path for orphaned staging files and unreferenced
  artifacts; never delete experiment dependencies through ordinary cache eviction.

Done when: old artifacts remain readable, backups restore independently, and failed
or concurrent writes cannot publish a partial result. Build on storage.py.

### D07 - Document And Derived-Factor Storage [P2]

- [ ] Define document IDs, hashes, company/scheme links, publication timestamps,
  revisions, and source references for filings, announcements, and news.
- [ ] Deduplicate text and cache extracted facts/chunks by document, parser, and model
  version. Persist retrieval metadata locally before choosing a vector database.
- [ ] Store sentiment/events as timestamped derived features with evidence references,
  extraction confidence, and availability rules, not as untraceable numeric columns.

Done when: a report claim and a backtested text factor can be traced to the exact
source revision available at that date. Depends on D03 and the data-owner handoff.

## App: Features

### A01 - Strict Strategy Interpretation [P0]

- [ ] Replace permissive fallback parsing with a versioned, typed DSL. Reject unknown
  fields, unsupported requests, ambiguous numbers, and invalid types with clear errors.
- [ ] Parse position count, rebalance frequency, indicator windows, and costs
  independently; show resolved rules and defaults before execution.
- [ ] Fix ambiguous cases such as `RSI below 30` being read as period 30, or an SMA
  prompt's position count being mistaken for a moving-average window.
- [ ] Distinguish an SMA trend state from a crossover event. Specify RSI smoothing
  and flat/zero-loss behavior, and verify against independent indicator fixtures.
- [ ] Define execution time, calendar alignment, cash behavior, and entry/exit rules
  explicitly in the schema; validate imported strategies through the same path.

Done when: supported prompts and equivalent controls compile to the same rules;
unsupported clauses cannot silently become a different strategy. Start in strategy_copilot.py.

### A02 - Consistent Backtest And Risk Behavior [P0]

- [ ] Reconcile Copilot and allocator evaluation windows, signal/fill timing,
  turnover definitions, fee totals, gross/net returns, and metric conventions.
- [ ] Expose which constraints each workflow enforces. Route richer Copilot weighting
  through shared allocation logic when sector, cash, position, or turnover limits apply.
- [ ] Add meaningful correctness checks: future-data perturbation, hand-calculated
  cash/fee/drift paths, invalid date bounds, constant prices, and no eligible assets.
- [ ] Attach benchmark, price basis, data version, and assumptions to every result.

Done when: equivalent fixed decisions produce matching accounting in both engines,
and modifying data after a decision cannot change that decision. Depends on A01.

### A03 - Complete Purchase And Cash Ledger [P1]

- [ ] Extend saved open lots into dated buys, sells/redemptions, deposits, withdrawals,
  fees, and distributions with stable transaction IDs and duplicate-import detection.
- [ ] Add partial-sale lot matching with an explicit policy, realized/unrealized P&L,
  available cash, and historical valuations. Preserve existing saved purchase books.
- [ ] Calculate portfolio XIRR from external dated cash flows plus terminal value;
  add time-weighted returns separately and handle missing/multiple XIRR roots.
- [ ] Import/export transactions and support SIP history. Distinguish actual units/NAV
  from planned SIP amounts and estimated allotments.
- [ ] Model fund-specific purchase/redemption settlement, available NAV, lock-ins, and
  exit-load schedules using supplied metadata; identify estimates and unsupported rules.
- [ ] Reconcile corporate actions through D04. Treat tax reporting as a separate future
  feature requiring verified jurisdiction-specific rules.

Done when: a mixed stock/fund portfolio with repeated purchases, a partial sale,
cash flows, and a distribution reconciles against an independent ledger calculation.
Depends on D02-D04; start in holdings.py and portfolio.py.

### A04 - Useful Portfolio And Fund Analysis [P1]

- [ ] Add instrument/category filters, selectable benchmarks, allocation history,
  concentration, drift, rolling returns, and contribution to portfolio performance.
- [ ] Compare funds within compatible categories/plans and common date windows.
- [ ] Add stock/fund and fund/fund underlying-holdings overlap only after dated fund
  constituent data is available; show coverage and stale disclosures.
- [ ] Add flow-based signals (net flow / AUM, AUM growth, category market share) once
  the AMFI monthly AUM/flows/folios source in `DATA.md` is ingested; treat as a
  category-relative signal, not a standalone recommendation.
- [ ] Add configurable cash yield and risk-free rate, plus separately specified
  commission, slippage, liquidity, and fund execution assumptions where data supports them.

Done when: displayed contributions reconcile to portfolio results and comparisons
share dates, currency, and return conventions. Depends on D01-D04 and A03 for cash flows.

### A05 - Evidence-Based Research Agent [P1/P2]

- [ ] Expand deterministic facts with point-in-time valuation, profitability, growth,
  leverage, cash flow, ownership, and fund-specific measures as feeds become available.
- [ ] Add relevant peer/benchmark selection, bull/base/bear cases, contradictory
  evidence, missing-data flags, and explicit score formulas and as-of timestamps.
- [ ] Introduce an optional local LLM adapter for planning and explanation, with
  structured outputs, timeouts, bounded tool calls, and a usable deterministic fallback.
- [ ] Retrieve only relevant dated evidence; keep financial calculations in Python
  and validate numeric claims/source references before presenting generated reports.
- [ ] Save reports, evidence bundles, model/prompt versions, and hypotheses so a
  Copilot result can link back to the original research.

Done when: facts are reproducible, material narrative claims have source references,
and unavailable evidence or a stopped model server does not break existing workflows.
Deterministic expansion is P1; LLM/retrieval work is P2 and depends on D03/D07.

### A06 - Richer Copilot And Robustness Research [P1/P2]

- [ ] Extend the validated DSL with combined conditions, rankings, fundamentals,
  alternative weighting, holding periods, and explicit exits as inputs become available.
- [ ] Specify fill semantics before adding stops; close-only data must not imply
  verified intraday stop execution. Use a proven engine for richer execution if suitable.
- [ ] Compare strategy variants against baselines over identical evaluation periods;
  explain drawdowns by period, sector, instrument, and exposure where data permits.
- [ ] Add bounded parameter sweeps, chronological validation, a final untouched test
  period, sensitivity plots, and reporting of every tested variant to expose selection bias.
- [ ] Add block-based uncertainty estimates and stress scenarios for costs, rebalance
  frequency, market regimes, missing inputs, and correlation changes.
- [ ] Retain hypothesis/variant lineage when an LLM suggests changes; bound any
  automated research loop and keep the evaluation gate deterministic.

Done when: a saved hypothesis can be revised, tested, compared, and replayed without
changing its original inputs or selecting parameters using the final test period.
Basic comparisons are P1; richer DSL and automated research are P2. Depends on A01/A02.

### A07 - Saved Work And Job Experience [P1]

- [ ] Save/reopen Copilot runs and research briefs using the existing artifact layer;
  add tags, search, run comparison, and export rather than keeping results only in session state.
- [ ] Mark results stale when data, dates, holdings, prompt, or settings change, and
  preserve separate settings/results across workspace navigation.
- [ ] Add progress, cancellation, retry, and failure details for long jobs; distinguish
  cancelling a running computation from merely hiding its spinner.
- [ ] Reconcile README/architecture claims with actual risk, storage, instrument, and
  AI behavior as these items ship; some architecture descriptions still imply full covariance.

Done when: users can recover saved work after restart, compare variants, and cannot
mistake a previous result for output calculated from current controls.

### A08 - Model Performance Dashboard [P1/P2]

- [ ] Add a dashboard for local and hosted model performance using the telemetry
  format in `LLM output format.md`.
- [ ] Track per-run TTFT, decode speed, end-to-end speed, total time, prefill speed,
  inter-token latency P50/P95, prompt tokens, output tokens, thinking tokens, failures,
  cancellations, retries, and structured-output validity.
- [ ] Store model metadata such as provider, base model, parameter count, quantization,
  precision, context size, reasoning mode, offline capability, backend, and task tags.
- [ ] Add dashboard views for overview ranking, model profile, run history, model
  comparison, and benchmark results across Research Agent and Strategy Copilot tasks.
- [ ] Add profile cards for metrics such as token efficiency, structured JSON pass
  rate, strategy prompt quality, math/code benchmark scores, median TTFT, and decode speed.
- [ ] Show the source, sample size, date range, prompt template version, and baseline
  model for every quality or benchmark score; do not promote marketing-only metrics.
- [ ] Use the dashboard to recommend default models per task only after measured
  latency, quality, reliability, and resource budgets are available.

Done when: a user can compare two candidate models on identical prompts, see both
quality and performance tradeoffs, and identify the best default model for each AI
workspace without leaving the app. Telemetry capture is P1; richer benchmarks and
automatic model recommendations are P2. Depends on A05/A06/O08.

## Optimisation Gaps

### O01 - Reproducible End-To-End Performance Coverage [P1]

- [ ] Extend scripts/benchmark_app.py to cover loading, synthetic generation, proposal
  creation, both new workspaces, purchase valuation, persistence, exports, and UI rendering.
- [ ] Benchmark 50/250/1,000 stocks at 650 and roughly 2,500 daily rows, plus selected
  mixed fund portfolios. Add a 5,000-asset stress case only after resource budgets are set.
- [ ] Record cold/warm timings, repeated-run median/p95, peak memory, cache size/hits,
  solver diagnostics, hardware, dependency versions, and dataset seed/hash.
- [ ] Measure two simultaneous sessions and repeated reruns. Agree per-workflow time
  and memory budgets from the baseline before enforcing regression thresholds.

Done when: a repeatable report covers the entire user path and reports numerical
equivalence alongside speed. Existing short benchmarks are reference points only.

### O02 - Synthetic Generator Scaling [P1, Observed]

- [ ] Remove repeated per-day multivariate decompositions in make_synthetic_prices;
  use its equicorrelation structure or a reused factorization and batched draws.
- [ ] Preserve regime drift, volatility scaling, deterministic seeds, positive prices,
  and exact date counts. Version the generator if seeded outputs change.

Done when: 1,000-asset generation is measured separately and no longer dominated by
repeated decompositions; distribution checks preserve intended covariance/regime behavior.

### O03 - Shared Features And Memory Budgets [P1, Measure First]

- [ ] Profile full-frame copies, hashing, repeated price validation, Pandas reshaping,
  indicator computation, and per-rebalance sorting before changing implementations.
- [ ] Reuse validated dataset contexts, fingerprints, research market statistics,
  and Copilot indicator caches with complete parameter/version keys.
- [ ] Benchmark partial top-N selection with deterministic ties and array-based result
  assembly; preserve ranking and accounting behavior when adopting either.
- [ ] Budget retained results and caches by bytes as well as entry count; assess
  long-history feature/ridge-statistic memory and selective result materialization.

Done when: representative profiles show reduced time or memory, and outputs match
the reference within declared tolerances. Depends on O01 and D06.

### O04 - Correlated Risk At 1,000 Assets [P1]

- [ ] Replace the automatic diagonal-only approximation above 400 assets with a
  scalable factor-plus-specific-risk option that preserves meaningful correlations.
- [ ] Fit factors using training history only; handle insufficient history and
  covariance conditioning, and expose the selected risk model in every result.
- [ ] Benchmark risk estimates and allocations against full shrinkage covariance
  on manageable universes, correlated sectors, and stress scenarios.
- [ ] Carry low-rank/sparse structure into the solver rather than reconstructing
  dense matrices unnecessarily; keep post-solve feasibility checks.

Done when: correlated exposures affect large-universe risk and allocations as
expected, with measured runtime/memory and out-of-sample risk diagnostics.

### O05 - Reuse Solver Setup Across Decisions [P1, Measure First]

- [ ] Measure CVXPY problem construction/canonicalization separately from solve time.
  Current calls build a new problem even though warm_start=True is supplied.
- [ ] Reuse parameterized problems and prior solutions where universe and constraint
  structure permit; invalidate reuse when structure changes.
- [ ] Record iterations, residuals, tolerances, infeasibility reasons, and fallback
  frequency; compare costs and constraint feasibility as well as elapsed time.

Done when: repeated decisions demonstrate a measurable gain without stale constraints
or degraded feasibility. Depends on O01; coordinate with O04 risk representations.

### O06 - Background Work And Bounded Concurrency [P1]

- [ ] Separate expensive runs from Streamlit reruns using a local job runner with
  bounded workers, queued/running/completed/failed/cancelled states, and job IDs.
- [ ] Preserve existing four-strategy parallelism while bounding total concurrency
  across sessions and sweeps; avoid duplicating full datasets per worker unnecessarily.
- [ ] Support cancellation checks between decisions, job deduplication, recovery
  metadata, and explicit CPU/memory limits. Keep BLAS thread limits coordinated.

Done when: navigation remains usable during a long run, cancelling stops computation,
and concurrent jobs respect budgets. No broker/Redis/database service is required.
Depends on O01/D06; enables A06/A07.

### O07 - Large Results And Experiment Browsing [P1]

- [ ] Extend current pagination to metadata-first artifact search and selective table
  reads, so browsing saved runs does not decode every complete result.
- [ ] Downsample long chart series while preserving drawdown extrema and provide
  full-resolution exports on demand; avoid constructing huge downloads on every rerun.
- [ ] Measure long rebalance/forecast histories, many saved experiments, and large
  fund selectors separately from backend computation.

Done when: opening a page reads only necessary data and exports preserve all rows;
render latency and memory remain within the budgets agreed in O01.

### O08 - Future LLM And Retrieval Performance [P2]

- [ ] Measure local hardware/backend compatibility before selecting models; benchmark
  latency, context limits, memory, and extraction accuracy on representative documents.
- [ ] Batch/deduplicate embeddings and extraction, cache by source/model/prompt version,
  bound retrieved context, and use separate concurrency budgets for LLM and CPU work.
- [ ] Add interruption, timeout, malformed-output, and evidence-quality checks before
  allowing multi-step research loops or large document jobs.

Done when: the optional AI path has measured resource budgets and reproducible evidence
retrieval while the deterministic application remains usable. Depends on A05/D07/O06.

## Suggested Delivery Order

| Stage | Work | Completion Gate |
|---|---|---|
| 1. Correctness and baseline | A01, A02, O01, O02 | Rules/accounting verified; end-to-end baseline recorded |
| 2. Real-data handoff | D01-D04, D06 | Selected stocks/funds load with identity, basis, and freshness checks |
| 3. Portfolio workflow | A03, A04, A07 | Ledger reconciliation, saved work, and comparable performance views |
| 4. Scale and responsiveness | D05, O03-O07 | 1,000-asset and concurrent-job budgets met without numerical regressions |
| 5. Research expansion | D07, A05, A06, O08 | Evidence-linked hypotheses, reproducible tests, and bounded local AI jobs |

Data contract work can run alongside Stage 1 with Claude. Point-in-time inputs are a
prerequisite for fundamental/news backtests even if price-only work ships earlier.

## Deferred Infrastructure And Expansion

- [ ] Consider optional embedded DuckDB when measured cross-experiment queries outgrow
  selective Parquet reads; retain file-based operation as the default until then.
- [ ] Evaluate a vector index/database only after document volume and retrieval
  measurements justify it. No database installation is part of the immediate backlog.
- [ ] Consider FastAPI and a separate frontend only when job APIs, multiple users, or
  UI requirements justify migration from Streamlit.
- [ ] Add stronger alpha/regime models and confidence calibration only after reliable
  point-in-time features and comparison gates exist; retain simple model baselines.
- [ ] Defer Docker/PostgreSQL deployment, paper/live broker execution, intraday engines,
  short selling, and autonomous trading to separately scoped work.

## Completion Practice

For each completed item, record the changed modules, relevant correctness checks,
benchmark evidence where applicable, and remaining limits. Update this checklist and
the current-behavior docs together. Keep unsupported scenarios explicit; a fast
synthetic run alone does not establish correctness or capacity for the full platform.
