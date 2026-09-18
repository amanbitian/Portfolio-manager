from __future__ import annotations

import hashlib
import json
from io import BytesIO
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
if (ROOT / ".app-deps").exists():
    sys.path.append(str(ROOT / ".app-deps"))

from ml_portfolio import AllocationConfig, BacktestConfig, load_price_csv, make_synthetic_prices
from ml_portfolio.data import validate_prices
from ml_portfolio.decisions import MIN_LISTING_HISTORY_DAYS
from ml_portfolio.market_data import (MarketDataError, dataset_fingerprint, load_market_prices,
                                      load_universe, resolve_data_root)
from ml_portfolio.optimizers import METHODS
from ml_portfolio.portfolio import build_proposal, save_proposal, saved_proposals
from ml_portfolio.reporting import (concentration, downsample_preserving_extrema, drawdown_series, metric_table,
                                    return_contributions, weight_drift)
from ml_portfolio.storage import price_key
from ml_portfolio.backtest import run_strategy_comparison
from ml_portfolio.experiments import research_overview, save_research
from ml_portfolio.holdings import (INSTRUMENT_TYPES, PURCHASE_COLUMNS, load_purchase_portfolio,
                                  purchase_portfolios, save_purchase_portfolio, value_purchases)
from ml_portfolio.ledger import (TRANSACTION_ACTIONS, append_transactions, load_transaction_ledger,
                                 save_transaction_ledger, transaction_ledgers, value_ledger)
from ml_portfolio.research_agent import (build_price_research_report, load_research_report,
                                         research_reports, save_research_report)
from ml_portfolio.strategy_copilot import (compile_strategy_prompt, load_strategy_run, load_strategy_sweep,
                                           run_strategy_dsl_backtest, save_strategy_run, save_strategy_sweep,
                                           strategy_from_controls, strategy_runs, strategy_sweeps, sweep_strategy)

st.set_page_config(page_title="ML Portfolio Allocation", layout="wide")


@st.cache_data(show_spinner=False, max_entries=3)
def run_comparison(prices, config, backtest):
    return run_strategy_comparison(prices, config, backtest)


@st.cache_data(show_spinner=False, max_entries=3)
def uploaded_prices(contents):
    return load_price_csv(BytesIO(contents))


@st.cache_data(show_spinner=False, max_entries=3)
def saved_research_overview(path, modified_ns):
    return research_overview(Path(path))


def _experiments_fingerprint(root, pattern="*.json"):
    # Cheap: count + max mtime, no file-content reads - mirrors Stage 2's
    # dataset_fingerprint pattern. Changes whenever a proposal is saved/removed,
    # so cached_saved_proposals below correctly invalidates without needing to
    # re-decode every proposal's trades table on every unrelated rerun (e.g. a
    # table_preview page-number click).
    paths = list(Path(root).glob(pattern))
    if not paths:
        return (0, 0.0)
    return (len(paths), max(p.stat().st_mtime for p in paths))


@st.cache_data(show_spinner=False, max_entries=5)
def cached_saved_proposals(prices, latest_prices, as_of, dir_fingerprint):
    return saved_proposals(ROOT / "experiments", latest_prices, as_of, prices)


@st.cache_data(show_spinner=False)
def demo_prices(periods, seed, end_date):
    return make_synthetic_prices(periods=periods, seed=seed, end_date=end_date)


@st.cache_data(show_spinner="Loading real data...", max_entries=3)
def real_data_prices(data_root, symbols, start, end, fingerprint_key):
    # `fingerprint_key` (unused inside) participates in Streamlit's cache key so a
    # fresh pipeline run (which changes the manifest mtime) invalidates the cache
    # even when the symbol/date selection itself is unchanged.
    return load_market_prices(data_root, list(symbols), start or None, end or None)


def price_hash(prices):
    """Content hash of `prices`, memoized against the exact same object reference
    for the rest of this rerun. O03 profiling measured the full-frame hash this
    replaces at ~16ms per call at 1000 assets/1500 rows, and several views call
    `fingerprint()` on the same `prices` two or more times in a single script
    run (e.g. `portfolio_view`'s `source_key` and `signature`) - Streamlit
    reruns the whole script on every widget interaction, so that was tens of ms
    of pure repeated waste on unchanged data per click. Guarding the memo with
    `is` (not just a cached id) means a garbage-collected id can never be
    mistaken for a hit: the cached tuple holds a live reference to the exact
    `prices` object it was computed from. Cleared automatically on the next
    rerun where a freshly loaded/uploaded `prices` object replaces it - no
    manual invalidation needed since `st.cache_data`'s loaders already return a
    fresh object whenever the underlying data actually changes.
    """
    cached = st.session_state.get("_price_hash_cache")
    if cached is not None and cached[0] is prices:
        return cached[1]
    digest = hashlib.sha256(pd.util.hash_pandas_object(prices, index=True).values.tobytes()).hexdigest()
    st.session_state["_price_hash_cache"] = (prices, digest)
    return digest


def fingerprint(prices, *settings):
    digest = hashlib.sha256(price_hash(prices).encode())
    digest.update(repr((list(prices.columns), settings)).encode())
    return digest.hexdigest()


@st.cache_data(show_spinner=False, max_entries=3)
def cached_price_diagnostics(price_signature, _prices):
    # `_prices` (leading underscore) opts out of Streamlit's own argument
    # hashing - `price_signature` (the memoized `price_hash` above) is what
    # actually keys the cache, so this neither re-hashes the full frame nor
    # re-runs validate_prices's pct_change/abs/isna reductions on every rerun
    # while the diagnostics expander stays open.
    return validate_prices(_prices)


def metric(value, percent=False):
    if value is None or not np.isfinite(value):
        return "N/A"
    return f"{value:.2%}" if percent else f"{value:,.2f}"


def display_columns(prices, limit=40):
    if prices.shape[1] <= limit:
        return prices
    volatility = prices.pct_change().tail(126).std().sort_values(ascending=False)
    return prices[volatility.head(limit).index]


def allocation_frame(proposal, limit=80):
    frame = pd.DataFrame({"Current": proposal["current_weights"],
                          "Target": proposal["decision"]["weights"],
                          "Rounded": proposal["rounded_weights"]})
    if len(frame) <= limit:
        return frame
    score = frame.drop(index="CASH", errors="ignore").abs().max(axis=1).sort_values(ascending=False)
    keep = list(score.head(limit - 2).index)
    if "CASH" in frame.index:
        keep.append("CASH")
    other = frame.drop(index=keep, errors="ignore").sum().rename("Other")
    return pd.concat([frame.loc[keep], other.to_frame().T])


def table_preview(frame, key, page_size=100):
    if "symbol" in frame.columns and len(frame) > page_size:
        query = st.text_input("Symbol filter", key=f"{key}_filter")
        if query:
            frame = frame.loc[frame.symbol.astype(str).str.contains(query, case=False, regex=False)]
    pages = max(1, (len(frame) + page_size - 1) // page_size)
    page_key = f"{key}_page"
    if st.session_state.get(page_key, 1) > pages:
        st.session_state[page_key] = 1
    page = int(st.number_input("Page", min_value=1, max_value=pages,
                               value=1, step=1, key=page_key)) if pages > 1 else 1
    start = (page - 1) * page_size
    st.caption(f"{len(frame):,} rows | Page {page:,} of {pages:,}")
    st.dataframe(frame.iloc[start:start+page_size], width="stretch", hide_index=True)


def summarized_holdings(holdings, limit=40):
    if len(holdings) <= limit:
        return holdings
    keep = list(holdings.drop(index="CASH", errors="ignore").nlargest(limit-2).index)
    if "CASH" in holdings:
        keep.append("CASH")
    return pd.concat([holdings.loc[keep], pd.Series({"Other positions": holdings.drop(keep).sum()})])


def show_validation(decision):
    summary = decision["validation_summary"]
    if summary["reason"] == "Baseline selected":
        return
    st.subheader("Forecast Evidence")
    cols = st.columns(4)
    cols[0].metric("Validation blocks", summary["blocks"])
    cols[1].metric("Error improvement", metric(summary.get("skill"), True))
    cols[2].metric("Block win rate", metric(summary.get("block_win_rate"), True))
    cols[3].metric("Rank correlation", metric(summary.get("rank_ic")))
    st.caption("Chronological validation within the training window. Confidence is error improvement, not a probability.")
    if not decision["validation"].empty:
        table_preview(decision["validation"], "proposal_validation")


def purchase_entry(prices):
    loaded = False
    choices = purchase_portfolios(ROOT / "experiments")
    if choices:
        with st.expander("Saved portfolios"):
            selected = st.selectbox("Portfolio snapshot", list(choices), format_func=choices.get)
            if st.button("Load portfolio"):
                try:
                    payload = load_purchase_portfolio(selected)
                    st.session_state["purchase_draft"] = payload["book"]["lots"][PURCHASE_COLUMNS]
                    st.session_state["purchase_name"] = payload["name"]
                    st.session_state["purchase_cash"] = float(payload["cash"])
                    st.session_state["purchase_revision"] = st.session_state.get("purchase_revision", 0) + 1
                    st.session_state.pop("proposal", None)
                    loaded = True
                except (ValueError, KeyError, OSError) as exc:
                    st.error(f"Could not load portfolio: {exc}")
    empty = pd.DataFrame({"symbol": pd.Series(dtype=str), "instrument_type": pd.Series(dtype=str),
                          "purchase_date": pd.Series(dtype="datetime64[ns]"),
                          "quantity": pd.Series(dtype=float), "purchase_price": pd.Series(dtype=float)})
    editor_key = f"purchase_lots_{st.session_state.get('purchase_revision', 0)}"
    if editor_key not in st.session_state and "purchase_last_valid" in st.session_state and not loaded:
        st.session_state["purchase_draft"] = st.session_state["purchase_last_valid"].copy()
    for key, default in [("purchase_name", "My portfolio"), ("purchase_cash", 0.0)]:
        if key not in st.session_state:
            st.session_state[key] = st.session_state.get(f"last_{key}", default)
    with st.form("purchase_holdings"):
        name = st.text_input("Portfolio name", key="purchase_name")
        cash = st.number_input("Available cash", min_value=0.0, step=1000.0, key="purchase_cash")
        purchases = st.data_editor(st.session_state.get("purchase_draft", empty), num_rows="dynamic",
                                  key=editor_key,
                                  hide_index=True, width="stretch", height=320,
                                  column_config={
                                      "symbol": st.column_config.SelectboxColumn("Symbol / Scheme", options=list(prices.columns), required=True),
                                      "instrument_type": st.column_config.SelectboxColumn("Instrument", options=INSTRUMENT_TYPES, required=True),
                                      "purchase_date": st.column_config.DateColumn("Purchase date", max_value=prices.index[-1].date(), required=True),
                                      "quantity": st.column_config.NumberColumn("Shares / Units", min_value=0., format="%.6f", required=True),
                                      "purchase_price": st.column_config.NumberColumn("Purchase price / NAV", min_value=0., format="%.6f", required=True)})
        save = st.form_submit_button("Save portfolio")
        submitted = st.form_submit_button("Calculate proposal", type="primary")
    if purchases.empty and not (save or submitted):
        return None
    try:
        book = value_purchases(purchases, prices.iloc[-1], prices.index[-1])
        st.session_state["purchase_last_valid"] = purchases.copy()
        st.session_state["last_purchase_name"] = name
        st.session_state["last_purchase_cash"] = cash
        if save:
            path = save_purchase_portfolio(ROOT / "experiments", name, purchases, cash,
                                            prices.iloc[-1], prices.index[-1])
            st.success(f"Saved {name}: {path.stem}")
    except (ValueError, OSError) as exc:
        st.error(str(exc))
        st.session_state.pop("proposal", None)
        return None
    st.caption(f"Valuation: {prices.index[-1].date()}")
    cols = st.columns(4)
    for col, label, field in zip(cols, ["Invested cost", "Holdings value", "Unrealized gain", "Return on cost"],
                                 ["invested_cost", "current_value", "unrealized_gain", "return_pct"]):
        col.metric(label, metric(book[field], field == "return_pct"))
    table_preview(book["positions"].reset_index(), "purchase_positions")
    with st.expander("Purchase lots"):
        table_preview(book["lots"], "purchase_lots")
    st.download_button("Download purchases", book["lots"][PURCHASE_COLUMNS].to_csv(index=False),
                       file_name="purchases.csv", mime="text/csv")
    return book["positions"].quantity, cash, purchases, book["positions"].instrument_type.to_dict(), submitted


_LEDGER_ENTRY_COLUMNS = ["action", "symbol", "instrument_type", "date", "quantity", "price", "amount", "notes"]


def transaction_entry(prices):
    loaded = False
    choices = transaction_ledgers(ROOT / "experiments")
    if choices:
        with st.expander("Saved ledgers"):
            selected = st.selectbox("Ledger snapshot", list(choices), format_func=choices.get, key="ledger_picker")
            if st.button("Load ledger"):
                try:
                    payload = load_transaction_ledger(selected)
                    st.session_state["ledger_draft"] = payload["transactions"][_LEDGER_ENTRY_COLUMNS]
                    st.session_state["ledger_name"] = payload["name"]
                    st.session_state["ledger_revision"] = st.session_state.get("ledger_revision", 0) + 1
                    st.session_state.pop("proposal", None)
                    loaded = True
                except (ValueError, KeyError, OSError) as exc:
                    st.error(f"Could not load ledger: {exc}")

    empty = pd.DataFrame({"action": pd.Series(dtype=str), "symbol": pd.Series(dtype=str),
                          "instrument_type": pd.Series(dtype=str), "date": pd.Series(dtype="datetime64[ns]"),
                          "quantity": pd.Series(dtype=float), "price": pd.Series(dtype=float),
                          "amount": pd.Series(dtype=float), "notes": pd.Series(dtype=str)})

    uploaded_csv = st.file_uploader("Import transactions CSV", type=["csv"], key="ledger_csv")
    just_imported = False
    if uploaded_csv is not None:
        # Hash the content, not `.name` - re-uploading a file that keeps the
        # same filename but has different (or corrected) rows must not be
        # silently skipped as "already processed".
        csv_signature = hashlib.md5(uploaded_csv.getvalue()).hexdigest()
        if st.session_state.get("ledger_csv_processed") != csv_signature:
            try:
                imported = pd.read_csv(uploaded_csv)
                current = st.session_state.get("ledger_draft", empty)
                existing = append_transactions(None, current, prices.index[-1])[0] if not current.empty else None
                combined, duplicates = append_transactions(existing, imported, prices.index[-1])
                st.session_state["ledger_draft"] = combined[_LEDGER_ENTRY_COLUMNS]
                st.session_state["ledger_revision"] = st.session_state.get("ledger_revision", 0) + 1
                st.session_state["ledger_csv_processed"] = csv_signature
                just_imported = True
                st.success(f"Imported {len(imported) - len(duplicates)} new transaction(s); "
                          f"skipped {len(duplicates)} already in the ledger.")
            except (ValueError, KeyError) as exc:
                st.error(f"Could not import CSV: {exc}")

    editor_key = f"ledger_transactions_{st.session_state.get('ledger_revision', 0)}"
    # `just_imported` guards against the case just handled above: bumping
    # `ledger_revision` on a successful import makes `editor_key` brand new in
    # this very run, which would otherwise make this restore-a-draft block
    # immediately overwrite the freshly-imported `ledger_draft` with the
    # pre-import `ledger_last_valid` snapshot before the editor ever renders it.
    if editor_key not in st.session_state and "ledger_last_valid" in st.session_state and not loaded and not just_imported:
        st.session_state["ledger_draft"] = st.session_state["ledger_last_valid"].copy()
    if "ledger_name" not in st.session_state:
        st.session_state["ledger_name"] = st.session_state.get("last_ledger_name", "My ledger")

    with st.form("transaction_ledger"):
        name = st.text_input("Ledger name", key="ledger_name")
        transactions = st.data_editor(
            st.session_state.get("ledger_draft", empty), num_rows="dynamic", key=editor_key,
            hide_index=True, width="stretch", height=320,
            column_config={
                "action": st.column_config.SelectboxColumn("Action", options=TRANSACTION_ACTIONS, required=True),
                "symbol": st.column_config.SelectboxColumn("Symbol / Scheme", options=list(prices.columns)),
                "instrument_type": st.column_config.SelectboxColumn("Instrument", options=INSTRUMENT_TYPES),
                "date": st.column_config.DateColumn("Date", max_value=prices.index[-1].date(), required=True),
                "quantity": st.column_config.NumberColumn("Quantity", min_value=0., format="%.6f"),
                "price": st.column_config.NumberColumn("Price / NAV", min_value=0., format="%.6f"),
                "amount": st.column_config.NumberColumn("Cash amount", min_value=0., format="%.2f"),
                "notes": st.column_config.TextColumn("Notes"),
            })
        save = st.form_submit_button("Save ledger")
        submitted = st.form_submit_button("Calculate proposal", type="primary")
    if transactions.empty and not (save or submitted):
        return None
    try:
        validated, _ = append_transactions(None, transactions, prices.index[-1])
        book = value_ledger(validated, prices.iloc[-1], prices.index[-1])
        st.session_state["ledger_last_valid"] = transactions.copy()
        st.session_state["last_ledger_name"] = name
        if save:
            path = save_transaction_ledger(ROOT / "experiments", name, validated, prices.iloc[-1], prices.index[-1])
            st.success(f"Saved {name}: {path.stem}")
    except (ValueError, OSError) as exc:
        st.error(str(exc))
        st.session_state.pop("proposal", None)
        return None

    st.caption(f"Valuation: {prices.index[-1].date()}")
    cols = st.columns(4)
    cols[0].metric("Holdings value", metric(book["holdings_value"]))
    cols[1].metric("Cash balance", metric(book["cash_balance"]))
    cols[2].metric("Realized gain", metric(book["realized_gain"]))
    cols[3].metric("Unrealized gain", metric(book["unrealized_gain"]))
    xirr = book["xirr"]
    cols = st.columns(2)
    cols[0].metric("XIRR", metric(xirr["xirr"], True) if xirr["converged"] else "N/A")
    cols[1].metric("Modified Dietz return (approx.)", metric(book["modified_dietz_return"], True))
    if not xirr["converged"] or xirr["multiple_roots_possible"]:
        st.caption(f"XIRR note: {xirr['note']}")
    if not book["positions"].empty:
        table_preview(book["positions"].reset_index(), "ledger_positions")
    if not book["realized"].empty:
        with st.expander("Realized sales"):
            table_preview(book["realized"], "ledger_realized")
    with st.expander("All transactions"):
        table_preview(validated, "ledger_transactions")
    st.download_button("Download transactions", validated[_LEDGER_ENTRY_COLUMNS].to_csv(index=False),
                       file_name="transactions.csv", mime="text/csv")
    if book["positions"].empty:
        return pd.Series(dtype=float), book["cash_balance"], {}, submitted
    return (book["positions"].quantity, book["cash_balance"],
            book["positions"].instrument_type.to_dict(), submitted)


def portfolio_view(prices, config, train_days, cost_bps):
    st.subheader("Current Portfolio")
    source_key = fingerprint(prices)
    mode = st.radio("Holdings entry", ["Quantities", "Purchases", "Transactions"], horizontal=True)
    purchases, types = None, None
    if mode == "Purchases":
        entry = purchase_entry(prices)
        if entry is None:
            return
        quantities, cash, purchases, types, submitted = entry
    elif mode == "Transactions":
        entry = transaction_entry(prices)
        if entry is None:
            return
        quantities, cash, types, submitted = entry
    else:
        with st.form("holdings"):
            cash = st.number_input("Available cash", min_value=0.0, value=100000.0, step=1000.0)
            holdings = st.data_editor(pd.DataFrame({"symbol": prices.columns, "quantity": 0.0, "instrument_type": "Stock"}),
                                     key=f"holdings_{source_key}", disabled=["symbol"], hide_index=True,
                                     column_config={"quantity": st.column_config.NumberColumn("Shares / Units", min_value=0., format="%.6f"),
                                                    "instrument_type": st.column_config.SelectboxColumn("Instrument", options=INSTRUMENT_TYPES, required=True)},
                                     width="stretch", height=360 if prices.shape[1] > 50 else "content")
            submitted = st.form_submit_button("Calculate proposal", type="primary")
        quantities = holdings.set_index("symbol").quantity
        types = holdings.set_index("symbol").instrument_type.to_dict()
    last_prices = prices.iloc[-1].reindex(quantities.index).fillna(0)
    values = quantities * last_prices
    capital = float(values.sum() + cash)
    if capital > 0:
        current_weights = values / capital
        current_weights["CASH"] = cash / capital
        risky_symbols = [s for s in quantities.index if s != "CASH"]
        equal_weight_reference = pd.Series(1.0 / max(len(risky_symbols), 1), index=risky_symbols)
        with st.expander("Concentration and drift", expanded=False):
            conc = concentration(current_weights)
            cols = st.columns(3)
            cols[0].metric("HHI", f"{conc['hhi']:.3f}")
            cols[1].metric("Top-3 weight share", metric(conc["top_3_share"], True))
            cols[2].metric("Top-5 weight share", metric(conc["top_5_share"], True))
            drift = weight_drift(current_weights.drop("CASH", errors="ignore"), equal_weight_reference)
            st.caption("Drift vs. an equal-weight reference across currently held risky assets.")
            st.bar_chart(drift)
    signature = fingerprint(prices, config, train_days, cost_bps, cash, quantities.to_dict(), types,
                            None if purchases is None else purchases.to_dict("records"))
    if submitted:
        try:
            with st.spinner("Evaluating forecasts and portfolio constraints..."):
                proposal = build_proposal(prices, quantities, cash, config, train_days, cost_bps,
                                           instrument_types=types, purchases=purchases)
            st.session_state["proposal"] = {"signature": signature, "result": proposal}
            st.session_state.pop("saved_proposal", None)
        except (ValueError, ArithmeticError) as exc:
            st.session_state.pop("proposal", None)
            st.error(str(exc))
    saved = st.session_state.get("proposal")
    if not saved:
        st.line_chart(display_columns(prices.tail(126)).div(display_columns(prices.tail(126)).iloc[0]))
        if prices.shape[1] > 40:
            st.caption("Preview chart shows the 40 most volatile assets over the last 126 sessions.")
        return
    if saved["signature"] != signature:
        st.info("Settings or holdings changed. Recalculate the proposal.")
        return
    proposal = saved["result"]
    decision = proposal["decision"]
    st.subheader("Proposed Allocation")
    st.caption(f"Valuation: {decision['as_of'].date()} | Information through: {decision['information_through'].date()} | {decision['effective_method']}")
    st.info(decision["reason"])
    cols = st.columns(4)
    cols[0].metric("Portfolio value", metric(proposal["capital"]))
    cols[1].metric("Estimated fees", metric(proposal["estimated_fees"]))
    cols[2].metric("Cash after trades", metric(proposal["cash_after"]))
    cols[3].metric("Checks", "Passed" if proposal["accepted"] else "Blocked")
    st.bar_chart(allocation_frame(proposal))
    st.subheader("Trade Preview")
    table_preview(proposal["trades"], "trades")
    st.caption("Stocks use whole shares; mutual-fund targets use six-decimal units at the latest loaded NAV. Fees are estimates; no orders are submitted.")
    st.subheader("Post-Trade Risk Checks")
    st.dataframe(proposal["checks"], width="stretch", hide_index=True)
    if not proposal["accepted"]:
        st.error("Rounded trades fail one or more limits. Adjust the portfolio or limits and recalculate.")
    with st.expander("Continuous target checks and solver"):
        st.dataframe(decision["checks"], width="stretch", hide_index=True)
        st.json(decision["solver"])
    show_validation(decision)
    if st.button("Save decision", disabled=not proposal["accepted"] or st.session_state.get("saved_proposal") == signature):
        try:
            path = save_proposal(ROOT / "experiments", proposal, prices, config, train_days, cost_bps)
            st.session_state["saved_proposal"] = signature
            st.success(f"Saved decision {path.stem}")
        except (ValueError, OSError) as exc:
            st.error(f"Could not save decision: {exc}")
    st.download_button("Download trade preview", proposal["trades"].to_csv(index=False),
                       file_name="trade_preview.csv", mime="text/csv")


def research_view(prices, config, train_days, cost_bps):
    st.subheader("Evaluation Period")
    default_start = prices.index[min(504, len(prices) - 64)]
    with st.form("evaluation"):
        cols = st.columns(3)
        start = cols[0].date_input("Start", value=default_start.date(),
                                  min_value=prices.index[0].date(), max_value=prices.index[-1].date())
        end = cols[1].date_input("End", value=prices.index[-1].date(),
                                min_value=prices.index[0].date(), max_value=prices.index[-1].date())
        rebalance = cols[2].number_input("Rebalance interval", min_value=1, max_value=126, value=21)
        submitted = st.form_submit_button("Run comparison", type="primary")
    backtest = BacktestConfig(train_days, int(rebalance), config.forecast_horizon_days, cost_bps,
                              str(start), str(end))
    signature = fingerprint(prices, replace(config, method="Equal weight"), backtest)
    if submitted:
        try:
            with st.spinner("Running strategies over a common evaluation period..."):
                results = run_comparison(prices, replace(config, method="Equal weight"), backtest)
            st.session_state["research"] = {"signature": signature, "results": results}
            st.session_state.pop("saved_research", None)
        except (ValueError, ArithmeticError) as exc:
            st.session_state.pop("research", None)
            st.error(str(exc))
    saved = st.session_state.get("research")
    if not saved:
        return
    if saved["signature"] != signature:
        st.info("Evaluation settings changed. Run the comparison again.")
        return
    results = saved["results"]
    primary = results[config.method]
    values = primary["metrics"]
    cols = st.columns(4)
    for col, name in zip(cols, ["CAGR", "Sharpe", "Volatility", "Max drawdown"]):
        col.metric(name, metric(values[name], name != "Sharpe"))
    st.line_chart(pd.DataFrame({name: result["equity_curve"].equity for name, result in results.items()}))
    st.dataframe(metric_table(results), width="stretch", hide_index=True)
    st.caption("Cash earns zero; Sharpe uses a zero risk-free rate. Cost drag compares identical decisions with and without fees. Risk limits apply at rebalances.")
    overview, allocations, contribution, evidence = st.tabs(
        ["Backtest", "Historical Holdings", "Contribution", "Forecast Outcomes"])
    with overview:
        drawdown = drawdown_series(primary["equity_curve"])
        if len(drawdown) > 2000:
            st.caption("Long series: chart downsampled while preserving drawdown extrema.")
        st.area_chart(downsample_preserving_extrema(drawdown))
        table_preview(primary["rebalances"], "rebalances")
        equity_cols = primary["equity_curve"][["equity", "gross_equity"]]
        if len(equity_cols) > 2000:
            kept_dates = downsample_preserving_extrema(equity_cols["equity"]).index
            equity_cols = equity_cols.loc[kept_dates]
        st.line_chart(equity_cols)
        st.subheader("63-Session Rolling Return")
        st.line_chart(primary["equity_curve"]["equity"].pct_change(63).rename("rolling_return"))
    with contribution:
        contributions = return_contributions(primary, prices)
        gross_total_return = float(primary["equity_curve"]["gross_equity"].iloc[-1]) - 1.0
        net_total_return = float(primary["equity_curve"]["equity"].iloc[-1]) - 1.0
        st.caption(f"Sum of contributions = gross return ({metric(gross_total_return, True)}). "
                  f"Net return ({metric(net_total_return, True)}) = gross return - Cost drag "
                  f"({metric(values['Cost drag'], True)}); cost is a single portfolio-level charge, "
                  "not attributed per position.")
        st.bar_chart(contributions)
        table_preview(contributions.rename_axis("symbol").reset_index(), "contributions")
    with allocations:
        st.subheader("Final Drifted Holdings")
        st.bar_chart(summarized_holdings(primary["final_holdings"]))
        st.subheader("Historical Rebalance Targets")
        weights = primary["weights"]
        if weights.shape[1] > 20:
            columns = st.multiselect("Assets", list(weights.columns), default=list(weights.columns[:20]),
                                      key=f"weight_assets_{signature}")
            weights = weights.loc[:, columns]
        table_preview(weights.rename_axis("date").reset_index(), "weight_history")
    with evidence:
        ml = results["ML mean-variance"]
        fallback = float((ml["rebalances"].effective_method != "ML mean-variance").mean())
        st.metric("ML fallback frequency", metric(fallback, True))
        table_preview(ml["forecasts"], "forecasts")
        st.subheader("Prior Validation Blocks")
        table_preview(ml["validation"], "research_validation")
    st.download_button("Download rebalance log", primary["rebalances"].to_csv(index=False),
                       file_name="rebalances.csv", mime="text/csv")
    with st.expander("Full table exports"):
        table = st.selectbox("Table", ["forecasts", "validation", "weights", "holdings"])
        if st.button("Prepare CSV"):
            frame = primary[table]
            st.download_button("Download CSV", frame.to_csv(index=table in ("weights", "holdings")),
                               file_name=f"{table}.csv", mime="text/csv", on_click="ignore")
    if st.button("Save research run", disabled=st.session_state.get("saved_research") == signature):
        try:
            path = save_research(ROOT / "experiments", prices, config, backtest, results)
            st.session_state["saved_research"] = signature
            st.success(f"Saved research {path.stem}")
        except (ValueError, OSError) as exc:
            st.error(f"Could not save research: {exc}")


def research_agent_view(prices):
    st.subheader("AI Portfolio Research Agent")
    with st.form("price_research_agent"):
        cols = st.columns([1, 2])
        symbol = cols[0].selectbox("Company / scheme", list(prices.columns))
        default_peers = [col for col in prices.columns if col != symbol][: min(4, prices.shape[1] - 1)]
        peers = cols[1].multiselect("Peers", [col for col in prices.columns if col != symbol],
                                    default=default_peers)
        question = st.text_area("Research question",
                                value="What looks attractive or risky compared with peers?",
                                height=90)
        submitted = st.form_submit_button("Generate research brief", type="primary")
    signature = fingerprint(prices, symbol, peers, question)
    if submitted:
        try:
            report = build_price_research_report(prices, symbol, peers, question)
            st.session_state["research_agent_report"] = {"signature": signature, "report": report}
        except ValueError as exc:
            st.session_state.pop("research_agent_report", None)
            st.error(str(exc))
    saved = st.session_state.get("research_agent_report")
    if not saved:
        st.info("Select a symbol and generate a price-led brief. Fundamentals, filings, and news can plug into this page later.")
        return
    if saved["signature"] != signature:
        st.info("Research inputs changed. Generate the brief again.")
        return
    report = saved["report"]
    st.caption(f"As of {report['as_of'].date()} | Uses loaded price/NAV data only")
    cols = st.columns(4)
    score_map = report["scores"].set_index("dimension").score
    for col, name in zip(cols, ["Overall", "Momentum", "Trend", "Risk stability"]):
        col.metric(name, f"{score_map[name]:.0f}/100")
    st.write(report["thesis"])
    score_tab, peer_tab, metric_tab, evidence_tab = st.tabs(["Scores", "Peers", "Market Facts", "Evidence"])
    with score_tab:
        st.bar_chart(report["scores"].set_index("dimension"))
        st.dataframe(report["scores"], width="stretch", hide_index=True)
    with peer_tab:
        st.dataframe(report["peers"], width="stretch", hide_index=True)
    with metric_tab:
        st.dataframe(report["summary"], width="stretch", hide_index=True)
    with evidence_tab:
        for item in report["evidence"]:
            st.caption(item)
        scenario = report["scenario_range"]
        if "bear_10th_percentile" in scenario:
            st.caption(f"Historical 1-year return range (10th/50th/90th percentile, "
                      f"n={scenario['sample_size']}, historical distribution not a forecast): "
                      f"{scenario['bear_10th_percentile']:.1%} / {scenario['base_median']:.1%} / "
                      f"{scenario['bull_90th_percentile']:.1%}")
        else:
            st.caption(scenario["note"])
        with st.expander("Score formulas"):
            st.json(report["formulas"])

    report_name = st.text_input("Report name", value=f"{report['symbol']} research", key="research_report_name")
    if st.button("Save this report", disabled=st.session_state.get("saved_research_report") == signature):
        try:
            path = save_research_report(ROOT / "experiments", report_name, prices, report)
            st.session_state["saved_research_report"] = signature
            st.session_state["saved_research_report_id"] = path.stem
            st.success(f"Saved research report {path.stem}")
        except (ValueError, OSError) as exc:
            st.error(f"Could not save research report: {exc}")

    if st.button("Use hypothesis in Strategy Copilot"):
        report_id = st.session_state.get("saved_research_report_id")
        if st.session_state.get("saved_research_report") != signature:
            try:
                path = save_research_report(ROOT / "experiments", report_name or f"{report['symbol']} research",
                                            prices, report)
                st.session_state["saved_research_report"] = signature
                report_id = path.stem
                st.session_state["saved_research_report_id"] = report_id
            except (ValueError, OSError) as exc:
                st.warning(f"Could not auto-save the report for linkage: {exc}")
                report_id = None
        st.session_state["copilot_prompt"] = report["hypothesis"]["strategy_prompt"]
        st.session_state["copilot_source_report_id"] = report_id
        st.success("Hypothesis copied to Strategy Copilot."
                  + (f" Linked to saved report {report_id}." if report_id else ""))


def _parse_candidates(text: str) -> list[int]:
    values = [int(piece.strip()) for piece in text.split(",") if piece.strip()]
    if not values:
        raise ValueError("Enter at least one candidate value.")
    return values


def strategy_sweep_view(prices):
    st.caption("Bounded parameter sweep: every combination is reported, not just the best, to expose "
              "selection bias. The final test window is never used to pick parameters.")
    with st.form("strategy_sweep"):
        template = st.selectbox("Template", ["momentum_top", "sma_cross", "rsi_oversold"], key="sweep_template")
        cols = st.columns(3)
        max_positions = cols[0].number_input("Max positions", min_value=1, max_value=int(prices.shape[1]),
                                             value=min(10, int(prices.shape[1])), step=1, key="sweep_max_positions")
        rebalance = cols[1].number_input("Rebalance interval", min_value=1, max_value=126, value=21, step=1,
                                         key="sweep_rebalance")
        cost = cols[2].number_input("Trading cost (bps)", min_value=0.0, max_value=1000.0, value=10.0, step=1.0,
                                    key="sweep_cost")
        if template == "momentum_top":
            grid_text = {"momentum_lookback": st.text_input("Momentum lookback candidates (days, comma-separated)",
                                                             value="63,126,189", key="sweep_grid_momentum")}
        elif template == "sma_cross":
            cols2 = st.columns(2)
            grid_text = {"fast_window": cols2[0].text_input("Fast SMA candidates", value="20,50", key="sweep_grid_fast"),
                        "slow_window": cols2[1].text_input("Slow SMA candidates", value="100,200", key="sweep_grid_slow")}
        else:
            cols2 = st.columns(2)
            grid_text = {"rsi_period": cols2[0].text_input("RSI period candidates", value="10,14", key="sweep_grid_period"),
                        "rsi_threshold": cols2[1].text_input("RSI threshold candidates", value="20,30", key="sweep_grid_threshold")}
        selection_metric = st.selectbox("Selection metric", ["Sharpe", "CAGR", "Max drawdown"], key="sweep_metric")
        default_split = prices.index[max(len(prices) // 2, 200)]
        cols3 = st.columns(2)
        selection_end = cols3[0].date_input("Selection window ends", value=default_split.date(),
                                            min_value=prices.index[0].date(), max_value=prices.index[-2].date(),
                                            key="sweep_selection_end")
        final_test_end = cols3[1].date_input("Final test ends", value=prices.index[-1].date(),
                                             min_value=prices.index[0].date(), max_value=prices.index[-1].date(),
                                             key="sweep_final_end")
        submitted = st.form_submit_button("Run sweep", type="primary")
    signature = fingerprint(prices, template, max_positions, rebalance, cost, grid_text,
                            selection_metric, selection_end, final_test_end)
    if submitted:
        try:
            param_grid = {name: _parse_candidates(text) for name, text in grid_text.items()}
            fixed_params = {"max_positions": int(max_positions), "rebalance_days": int(rebalance),
                            "transaction_cost_bps": float(cost)}
            final_test_start = prices.index[prices.index.searchsorted(pd.Timestamp(selection_end), side="right")]
            with st.spinner("Running bounded parameter sweep..."):
                sweep = sweep_strategy(prices, template, fixed_params, param_grid, selection_end,
                                       final_test_start, final_test_end, selection_metric)
            st.session_state["strategy_sweep_result"] = {"signature": signature, "sweep": sweep}
        except ValueError as exc:
            st.session_state.pop("strategy_sweep_result", None)
            st.error(str(exc))
    saved = st.session_state.get("strategy_sweep_result")
    if not saved:
        st.info("Set a bounded parameter grid and a selection/final-test split, then run a sweep.")
        return
    if saved["signature"] != signature:
        st.info("Sweep inputs changed. Run the sweep again.")
        return
    sweep = saved["sweep"]
    st.subheader("All tested variants (selection window)")
    st.caption("Every combination is shown, not just the winner - this is what makes selection bias visible.")
    table_preview(sweep["variants"], "sweep_variants")
    st.write(f"Selected by best **{sweep['selection_metric']}**: `{sweep['selected_params']}`")
    st.subheader("Final test (held out - never used to pick parameters)")
    cols = st.columns(4)
    ft_metrics = sweep["final_test_result"]["metrics"]
    for col, name in zip(cols, ["CAGR", "Sharpe", "Volatility", "Max drawdown"]):
        col.metric(name, metric(ft_metrics[name], name != "Sharpe"))
    st.line_chart(sweep["final_test_result"]["equity_curve"][["equity", "gross_equity"]])
    sweep_name = st.text_input("Sweep name", value=f"{template} sweep", key="sweep_name")
    if st.button("Save this sweep", disabled=st.session_state.get("saved_strategy_sweep") == signature):
        try:
            path = save_strategy_sweep(ROOT / "experiments", sweep_name, prices, sweep)
            st.session_state["saved_strategy_sweep"] = signature
            st.success(f"Saved sweep {path.stem}")
        except (ValueError, OSError) as exc:
            st.error(f"Could not save sweep: {exc}")


def strategy_copilot_view(prices, cost_bps, config):
    st.subheader("AI Backtesting Copilot")
    mode = st.radio("Mode", ["Single run", "Parameter sweep"], horizontal=True, key="copilot_mode")
    if mode == "Parameter sweep":
        strategy_sweep_view(prices)
        return
    st.caption(f"Applies the sidebar's position limit ({config.max_weight:.0%}) and cash reserve "
              f"({config.cash_weight:.0%}) - the same allocation controls Portfolio/Research use.")
    default_prompt = st.session_state.get(
        "copilot_prompt",
        "Test a 6-month momentum strategy, hold the top 10 assets, rebalance monthly, include costs.",
    )
    with st.form("strategy_copilot"):
        prompt = st.text_area("Strategy prompt", value=default_prompt, height=90)
        source = st.radio("Strategy source", ["Prompt parser", "Template controls"], horizontal=True)
        cols = st.columns(4)
        template = cols[0].selectbox("Template", ["momentum_top", "sma_cross", "rsi_oversold"])
        max_positions = cols[1].number_input("Max positions", min_value=1, max_value=int(prices.shape[1]),
                                             value=min(10, int(prices.shape[1])), step=1)
        rebalance = cols[2].number_input("Rebalance interval", min_value=1, max_value=126, value=21, step=1)
        cost = cols[3].number_input("Trading cost (bps)", min_value=0.0, max_value=1000.0,
                                    value=float(cost_bps), step=1.0)
        cols = st.columns(4)
        momentum = cols[0].number_input("Momentum lookback", min_value=2, max_value=min(504, len(prices) - 2),
                                        value=min(126, len(prices) - 2), step=1)
        fast = cols[1].number_input("Fast SMA", min_value=2, max_value=min(252, len(prices) - 2),
                                    value=min(50, len(prices) - 2), step=1)
        slow = cols[2].number_input("Slow SMA", min_value=3, max_value=min(504, len(prices) - 1),
                                    value=min(200, len(prices) - 1), step=1)
        rsi_threshold = cols[3].number_input("RSI entry", min_value=1, max_value=99, value=30, step=1)
        cols = st.columns(4)
        rsi_period = cols[0].number_input("RSI period", min_value=2, max_value=126, value=14, step=1)
        signal_mode = cols[1].selectbox("SMA signal", ["trend", "cross"],
                                        help="'trend' stays selected while fast > slow; "
                                             "'cross' fires only on a recent upward cross.")
        default_start = prices.index[min(252, len(prices) - 64)]
        start = cols[2].date_input("Start", value=default_start.date(),
                                   min_value=prices.index[0].date(), max_value=prices.index[-1].date())
        end = cols[3].date_input("End", value=prices.index[-1].date(),
                                 min_value=prices.index[0].date(), max_value=prices.index[-1].date())
        submitted = st.form_submit_button("Run strategy", type="primary")
    signature = fingerprint(prices, prompt, source, template, max_positions, rebalance, cost, momentum,
                            fast, slow, rsi_period, rsi_threshold, signal_mode, start, end,
                            config.max_weight, config.cash_weight)
    if submitted:
        try:
            if source == "Prompt parser":
                strategy = compile_strategy_prompt(prompt, max_positions=int(max_positions),
                                                   rebalance_days=int(rebalance),
                                                   transaction_cost_bps=float(cost))
            else:
                strategy = strategy_from_controls(template, fast_window=int(fast), slow_window=int(slow),
                                                  rsi_period=int(rsi_period), rsi_threshold=int(rsi_threshold),
                                                  momentum_lookback=int(momentum),
                                                  max_positions=int(max_positions),
                                                  rebalance_days=int(rebalance),
                                                  transaction_cost_bps=float(cost),
                                                  signal_mode=signal_mode)
            with st.expander("Resolved strategy rules", expanded=True):
                st.caption("Fields marked 'prompt' were parsed from your text; 'default' fields used "
                          "the built-in default; 'controls' fields came from the form controls.")
                st.json({"schema_version": strategy["schema_version"], "name": strategy["name"],
                        "entry": strategy["entry"], "portfolio": strategy["portfolio"],
                        "rebalance_days": strategy["rebalance_days"],
                        "transaction_cost_bps": strategy["transaction_cost_bps"],
                        "resolved_from": strategy["resolved_from"]})
            with st.spinner("Validating strategy DSL and running deterministic backtest..."):
                result = run_strategy_dsl_backtest(prices, strategy, evaluation_start=str(start), evaluation_end=str(end),
                                                   max_weight=config.max_weight, cash_weight=config.cash_weight)
            st.session_state["strategy_copilot_result"] = {"signature": signature, "result": result}
        except ValueError as exc:
            st.session_state.pop("strategy_copilot_result", None)
            st.error(str(exc))
    saved = st.session_state.get("strategy_copilot_result")
    if not saved:
        st.info("Describe a strategy or use template controls, then run a deterministic DSL backtest.")
        return
    if saved["signature"] != signature:
        st.info("Strategy inputs changed. Run the strategy again.")
        return
    result = saved["result"]
    st.caption("Signals are shifted by one trading day. The DSL is validated before execution; no arbitrary code is generated.")
    cols = st.columns(4)
    metrics = result["metrics"]
    for col, name in zip(cols, ["CAGR", "Sharpe", "Volatility", "Max drawdown"]):
        col.metric(name, metric(metrics[name], name != "Sharpe"))
    st.line_chart(result["equity_curve"][["equity", "gross_equity"]])
    dsl_tab, rebalance_tab, score_tab = st.tabs(["Strategy JSON", "Rebalances", "Latest Scores"])
    with dsl_tab:
        st.code(json.dumps(result["strategy"], indent=2), language="json")
    with rebalance_tab:
        table_preview(result["rebalances"], "copilot_rebalances")
        st.download_button("Download copilot rebalances", result["rebalances"].to_csv(index=False),
                           file_name="copilot_rebalances.csv", mime="text/csv")
    with score_tab:
        table_preview(result["latest_scores"], "copilot_scores")
    run_name = st.text_input("Run name", value="My strategy run", key="strategy_run_name")
    if st.button("Save this run", disabled=st.session_state.get("saved_strategy_run") == signature):
        try:
            path = save_strategy_run(ROOT / "experiments", run_name, prices, result,
                                     source_research_report_id=st.session_state.get("copilot_source_report_id"))
            st.session_state["saved_strategy_run"] = signature
            st.success(f"Saved strategy run {path.stem}")
        except (ValueError, OSError) as exc:
            st.error(f"Could not save strategy run: {exc}")


def main():
    st.title("ML Portfolio Allocation")
    page = st.sidebar.radio("Workspace", ["Portfolio", "Research", "Research Agent",
                                          "Strategy Copilot", "Saved Decisions"])
    source = st.sidebar.radio("Data source", ["Synthetic demo", "Upload CSV", "Real data (data lake)"])
    if source == "Synthetic demo":
        seed = st.sidebar.number_input("Demo seed", min_value=1, max_value=9999, value=42)
        periods = st.sidebar.slider("Demo trading days", 520, 2600, 1300, 20)
        prices = demo_prices(periods, seed, str(pd.Timestamp.today().date()))
        st.caption("Synthetic demo prices")
    elif source == "Upload CSV":
        uploaded = st.sidebar.file_uploader("Price CSV", type=["csv"])
        if uploaded is None:
            st.info("Upload a CSV, or switch Data source in the sidebar.")
            return
        try:
            prices = uploaded_prices(uploaded.getvalue())
        except Exception as exc:
            st.error(str(exc))
            return
        st.caption(f"Price file: {uploaded.name}")
    else:
        data_root_input = st.sidebar.text_input("Data root", value=str(resolve_data_root()))
        data_root = Path(data_root_input)
        try:
            universe = load_universe(data_root)
        except MarketDataError as exc:
            st.sidebar.error(str(exc))
            return
        labels = dict(zip(universe.symbol, universe.symbol + " — " + universe.name.fillna("")))
        selected = st.sidebar.multiselect("Symbols", list(universe.symbol), default=list(universe.symbol[:10]),
                                          format_func=lambda s: labels.get(s, s))
        default_start = (pd.Timestamp.today() - pd.DateOffset(years=3)).date()
        start_date = st.sidebar.date_input("Start date", value=default_start)
        end_date = st.sidebar.date_input("End date", value=pd.Timestamp.today().date())
        if len(selected) < 2:
            st.info("Select at least two symbols in the sidebar.")
            return
        fingerprint_key = dataset_fingerprint(data_root, selected, str(start_date), str(end_date))
        try:
            prices, report = real_data_prices(str(data_root), tuple(selected), str(start_date),
                                              str(end_date), fingerprint_key)
        except MarketDataError as exc:
            st.error(str(exc))
            return
        st.caption(f"Real data: {report['loaded_symbols']}/{report['requested_symbols']} symbols loaded, "
                  f"{report['coverage_start'][:10]} to {report['coverage_end'][:10]}")
        with st.expander("Data coverage and caveats"):
            st.write(report["price_basis"])
            st.write(report["universe_caveat"])
            if report["excluded"]:
                st.write("Excluded symbols:")
                st.json(report["excluded"])
            if report.get("listed_mid_window"):
                st.write(f"Listed inside the window ({len(report['listed_mid_window'])} symbols) - held at "
                         f"zero weight until {MIN_LISTING_HISTORY_DAYS} sessions after listing, and not "
                         "counted in the benchmark until they exist:")
                st.json(report["listed_mid_window"])
            if report.get("truncated"):
                st.write(f"Stopped trading inside the window ({len(report['truncated'])} symbols) - "
                         "sold to cash at the last real price:")
                st.json(report["truncated"])
            diagnostics = cached_price_diagnostics(price_hash(prices), prices)
            for note in diagnostics.notes:
                st.write("-", note)
    if len(prices) < 230:
        st.error("At least 230 daily price rows are required.")
        return
    with st.sidebar.form("allocation_settings"):
        method = st.selectbox("Strategy", METHODS)
        backend = st.selectbox("Allocation solver", ["auto", "convex", "slsqp"])
        train_days = st.number_input("Training rows", min_value=166, max_value=min(1008, len(prices)-64),
                                     value=min(504, len(prices)-64), step=1)
        horizon = st.number_input("Forecast horizon", min_value=5, max_value=63, value=21)
        max_weight = st.slider("Position limit", .05, 1.0, .25, .01)
        reserve = st.slider("Cash reserve", 0.0, .8, .05, .01)
        risk = st.slider("Risk aversion", 1.0, 20.0, 6.0, .5)
        vol_limit = st.checkbox("Limit annual volatility", value=True)
        volatility = st.slider("Annual volatility ceiling", .03, .60, .20, .01)
        turnover = st.slider("One-way turnover budget", 0.0, 1.0, 1.0, .05)
        cost = st.number_input("Trading cost (bps)", min_value=0.0, max_value=100.0, value=10.0)
        edge = st.number_input("ML minimum trade benefit (bps)", min_value=0.0, max_value=100.0, value=2.0)
        sector_limits = st.checkbox("Apply sector limits")
        sector_cap = st.slider("Maximum sector exposure", .05, 1.0, .4, .05)
        sectors = None
        if sector_limits:
            sectors = st.data_editor(pd.DataFrame({"symbol": prices.columns, "sector": "Unassigned"}),
                                     disabled=["symbol"], hide_index=True, key=f"sectors_{fingerprint(prices)}",
                                     width="stretch", height=360 if prices.shape[1] > 50 else "content")
        st.form_submit_button("Apply settings")
    mapping = ({symbol: "Unassigned" for symbol in prices.columns} if sectors is None else
               dict(zip(sectors.symbol, sectors.sector.fillna("Unassigned").replace("", "Unassigned"))))
    config = AllocationConfig(method, max_weight, reserve, risk, int(horizon),
                               volatility if vol_limit else None, turnover, edge,
                               mapping, {s: sector_cap for s in set(mapping.values())} if sector_limits else {}, backend)
    if page == "Portfolio":
        portfolio_view(prices, config, int(train_days), cost)
    elif page == "Research":
        research_view(prices, config, int(train_days), cost)
    elif page == "Research Agent":
        research_agent_view(prices)
    elif page == "Strategy Copilot":
        strategy_copilot_view(prices, cost, config)
    else:
        st.subheader("Saved Decisions")
        dir_fingerprint = _experiments_fingerprint(ROOT / "experiments")
        history = cached_saved_proposals(prices, prices.iloc[-1], prices.index[-1], dir_fingerprint)
        if history.empty:
            st.info("No saved decisions.")
        else:
            table_preview(history, "saved_decisions")
            st.caption(f"Hypothetical holdings marked at {prices.index[-1].date()}; not a trade ledger.")
        # "research_report_*.json" (research_agent.py's newer, separate kind -
        # listed correctly below via research_reports()) is a substring match
        # for this older "research_*" glob too - excluded explicitly, since
        # opening one here raised "Unsupported research artifact" instead of
        # anything usable.
        research_paths = sorted((p for p in (ROOT / "experiments").glob("research_*.json")
                                 if not p.name.startswith("research_report_")), reverse=True)
        if research_paths:
            st.subheader("Saved Research")
            selected = st.selectbox("Run", research_paths, format_func=lambda p: p.stem)
            if st.button("Open saved run"):
                try:
                    results = saved_research_overview(str(selected), selected.stat().st_mtime_ns)
                    st.session_state["research_overview"] = {"path": str(selected), "results": results}
                except (ValueError, KeyError, OSError) as exc:
                    st.error(f"Could not open research: {exc}")
            saved = st.session_state.get("research_overview")
            if saved and saved["path"] == str(selected):
                st.dataframe(metric_table(saved["results"]), width="stretch", hide_index=True)
                st.line_chart(pd.DataFrame({name: result["equity_curve"].equity
                                            for name, result in saved["results"].items()}))
        runs = strategy_runs(ROOT / "experiments")
        if runs:
            st.subheader("Saved Strategy Runs")
            selected_run = st.selectbox("Strategy run", list(runs), format_func=runs.get)
            if st.button("Open saved strategy run"):
                try:
                    payload = load_strategy_run(selected_run)
                    st.session_state["strategy_run_overview"] = {"path": str(selected_run), "payload": payload}
                except (ValueError, KeyError, OSError) as exc:
                    st.error(f"Could not open strategy run: {exc}")
            saved_run = st.session_state.get("strategy_run_overview")
            if saved_run and saved_run["path"] == str(selected_run):
                payload = saved_run["payload"]
                if payload["price_hash"] != price_key(prices):
                    st.warning("The currently loaded data differs from what this run was saved against; "
                              "metrics below reflect the data at save time, not current data.")
                result = payload["result"]
                st.dataframe(pd.DataFrame([result["metrics"]]), width="stretch", hide_index=True)
                st.line_chart(result["equity_curve"][["equity", "gross_equity"]])
                source_id = payload.get("source_research_report_id")
                st.caption(f"Linked research report: {source_id}" if source_id
                          else "Not derived from a saved research report.")
        reports = research_reports(ROOT / "experiments")
        if reports:
            st.subheader("Saved Research Reports")
            selected_report = st.selectbox("Research report", list(reports), format_func=reports.get)
            if st.button("Open saved research report"):
                try:
                    st.session_state["research_report_overview"] = {
                        "path": str(selected_report), "payload": load_research_report(selected_report)}
                except (ValueError, KeyError, OSError) as exc:
                    st.error(f"Could not open research report: {exc}")
            saved_report = st.session_state.get("research_report_overview")
            if saved_report and saved_report["path"] == str(selected_report):
                report = saved_report["payload"]["report"]
                st.write(report["thesis"])
                st.dataframe(report["scores"], width="stretch", hide_index=True)
        sweeps = strategy_sweeps(ROOT / "experiments")
        if sweeps:
            st.subheader("Saved Parameter Sweeps")
            selected_sweep = st.selectbox("Sweep", list(sweeps), format_func=sweeps.get)
            if st.button("Open saved sweep"):
                try:
                    st.session_state["strategy_sweep_overview"] = {
                        "path": str(selected_sweep), "payload": load_strategy_sweep(selected_sweep)}
                except (ValueError, KeyError, OSError) as exc:
                    st.error(f"Could not open sweep: {exc}")
            saved_sweep = st.session_state.get("strategy_sweep_overview")
            if saved_sweep and saved_sweep["path"] == str(selected_sweep):
                sweep = saved_sweep["payload"]["sweep"]
                st.caption("Replayed exactly as saved - nothing recomputed with hindsight.")
                table_preview(sweep["variants"], "saved_sweep_variants")
                st.write(f"Selected by best **{sweep['selection_metric']}**: `{sweep['selected_params']}`")
                st.dataframe(pd.DataFrame([sweep["final_test_result"]["metrics"]]), width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
