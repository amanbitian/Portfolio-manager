from __future__ import annotations

import hashlib
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

from ml_portfolio import AllocationConfig, BacktestConfig, load_price_csv, make_synthetic_prices, run_walk_forward_backtest
from ml_portfolio.optimizers import METHODS
from ml_portfolio.portfolio import build_proposal, save_proposal, saved_proposals
from ml_portfolio.reporting import drawdown_series, metric_table

st.set_page_config(page_title="ML Portfolio Allocation", layout="wide")


@st.cache_data(show_spinner=False)
def run_strategy(prices, config, backtest):
    return run_walk_forward_backtest(prices, config, backtest)


@st.cache_data(show_spinner=False)
def demo_prices(periods, seed, end_date):
    return make_synthetic_prices(periods=periods, seed=seed, end_date=end_date)


def fingerprint(prices, *settings):
    digest = hashlib.sha256(pd.util.hash_pandas_object(prices, index=True).values.tobytes())
    digest.update(repr((list(prices.columns), settings)).encode())
    return digest.hexdigest()


def metric(value, percent=False):
    if value is None or not np.isfinite(value):
        return "N/A"
    return f"{value:.2%}" if percent else f"{value:,.2f}"


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
        st.dataframe(decision["validation"], use_container_width=True, hide_index=True)


def portfolio_view(prices, config, train_days, cost_bps):
    st.subheader("Current Portfolio")
    source_key = fingerprint(prices)
    with st.form("holdings"):
        cash = st.number_input("Available cash", min_value=0.0, value=100000.0, step=1000.0)
        holdings = st.data_editor(pd.DataFrame({"symbol": prices.columns, "quantity": 0}),
                                 key=f"holdings_{source_key}", disabled=["symbol"], hide_index=True,
                                 column_config={"quantity": st.column_config.NumberColumn("Shares", min_value=0, step=1)},
                                 use_container_width=True)
        submitted = st.form_submit_button("Calculate proposal", type="primary")
    quantities = holdings.set_index("symbol").quantity
    signature = fingerprint(prices, config, train_days, cost_bps, cash, quantities.to_dict())
    if submitted:
        try:
            with st.spinner("Evaluating forecasts and portfolio constraints..."):
                proposal = build_proposal(prices, quantities, cash, config, train_days, cost_bps)
            st.session_state["proposal"] = {"signature": signature, "result": proposal}
            st.session_state.pop("saved_proposal", None)
        except (ValueError, ArithmeticError) as exc:
            st.session_state.pop("proposal", None)
            st.error(str(exc))
    saved = st.session_state.get("proposal")
    if not saved:
        st.line_chart(prices.tail(126).div(prices.tail(126).iloc[0]))
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
    st.bar_chart(pd.DataFrame({"Current": proposal["current_weights"],
                               "Target": decision["weights"], "Rounded": proposal["rounded_weights"]}))
    st.subheader("Trade Preview")
    st.dataframe(proposal["trades"], use_container_width=True, hide_index=True)
    st.caption("Whole-share estimates at the latest close. Fees are estimated; no orders are submitted.")
    st.subheader("Post-Trade Risk Checks")
    st.dataframe(proposal["checks"], use_container_width=True, hide_index=True)
    if not proposal["accepted"]:
        st.error("Rounded trades fail one or more limits. Adjust the portfolio or limits and recalculate.")
    with st.expander("Continuous target checks and solver"):
        st.dataframe(decision["checks"], use_container_width=True, hide_index=True)
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
                results = {method: run_strategy(prices, replace(config, method=method), backtest) for method in METHODS}
            st.session_state["research"] = {"signature": signature, "results": results}
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
    st.dataframe(metric_table(results), use_container_width=True, hide_index=True)
    st.caption("Cash earns zero; Sharpe uses a zero risk-free rate. Cost drag compares identical decisions with and without fees. Risk limits apply at rebalances.")
    overview, allocations, evidence = st.tabs(["Backtest", "Historical Holdings", "Forecast Outcomes"])
    with overview:
        st.area_chart(drawdown_series(primary["equity_curve"]))
        st.dataframe(primary["rebalances"], use_container_width=True, hide_index=True)
        st.line_chart(primary["equity_curve"][["equity", "gross_equity"]])
    with allocations:
        st.subheader("Final Drifted Holdings")
        st.bar_chart(primary["final_holdings"])
        st.subheader("Historical Rebalance Targets")
        st.dataframe(primary["weights"], use_container_width=True)
    with evidence:
        ml = results["ML mean-variance"]
        fallback = float((ml["rebalances"].effective_method != "ML mean-variance").mean())
        st.metric("ML fallback frequency", metric(fallback, True))
        st.dataframe(ml["forecasts"], use_container_width=True, hide_index=True)
        st.subheader("Prior Validation Blocks")
        st.dataframe(ml["validation"], use_container_width=True, hide_index=True)
    st.download_button("Download rebalance log", primary["rebalances"].to_csv(index=False),
                       file_name="rebalances.csv", mime="text/csv")


def main():
    st.title("ML Portfolio Allocation")
    page = st.sidebar.radio("Workspace", ["Portfolio", "Research", "Saved Decisions"])
    uploaded = st.sidebar.file_uploader("Price CSV", type=["csv"])
    if uploaded is None:
        seed = st.sidebar.number_input("Demo seed", min_value=1, max_value=9999, value=42)
        periods = st.sidebar.slider("Demo trading days", 520, 2600, 1300, 20)
        prices = demo_prices(periods, seed, str(pd.Timestamp.today().date()))
        st.caption("Synthetic demo prices")
    else:
        try:
            prices = load_price_csv(uploaded)
        except Exception as exc:
            st.error(str(exc))
            return
        st.caption(f"Price file: {uploaded.name}")
    if len(prices) < 230:
        st.error("At least 230 daily price rows are required.")
        return
    with st.sidebar.form("allocation_settings"):
        method = st.selectbox("Strategy", METHODS)
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
        sectors = st.data_editor(pd.DataFrame({"symbol": prices.columns, "sector": "Unassigned"}),
                                 disabled=["symbol"], hide_index=True, key=f"sectors_{tuple(prices.columns)}")
        st.form_submit_button("Apply settings")
    mapping = dict(zip(sectors.symbol, sectors.sector.fillna("Unassigned").replace("", "Unassigned")))
    config = AllocationConfig(method, max_weight, reserve, risk, int(horizon),
                               volatility if vol_limit else None, turnover, edge,
                               mapping, {s: sector_cap for s in set(mapping.values())} if sector_limits else {})
    if page == "Portfolio":
        portfolio_view(prices, config, int(train_days), cost)
    elif page == "Research":
        research_view(prices, config, int(train_days), cost)
    else:
        st.subheader("Saved Decisions")
        history = saved_proposals(ROOT / "experiments", prices.iloc[-1], prices.index[-1], prices)
        if history.empty:
            st.info("No saved decisions.")
        else:
            st.dataframe(history, use_container_width=True, hide_index=True)
            st.caption(f"Hypothetical whole-share holdings marked at {prices.index[-1].date()}; not a trade ledger.")


if __name__ == "__main__":
    main()
