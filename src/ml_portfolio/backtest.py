from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import os
from pathlib import Path
from threading import Lock

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .data import eligibility_cutoffs, eligibility_starts
from .decisions import build_context, make_decision
from .execution import cutoff_positions, liquidate_at_cutoff, rebalance_cost_rate, start_positions
from .optimizers import METHODS, AllocationConfig, normalize_holdings
from .storage import DEFAULT_CACHE, price_key

_comparison_lock = Lock()

# Shared minimum-evaluation-length floor: CAGR/vol/Sharpe/drawdown estimated from
# fewer realized returns carry too much sampling noise to compare across engines.
# strategy_copilot.py imports this so its evaluation window is comparable to the
# allocator's, not just independently "valid."
MIN_EVALUATION_DAYS = 63


@dataclass(frozen=True)
class BacktestConfig:
    train_days: int = 504
    rebalance_days: int = 21
    forward_days: int = 21
    transaction_cost_bps: float = 10.0
    evaluation_start: str | None = None
    evaluation_end: str | None = None
    min_evaluation_days: int = MIN_EVALUATION_DAYS


def build_assumptions(prices: pd.DataFrame, start: int, end: int, cost_bps: float,
                      rebalance_days: int) -> dict:
    """Disclosure metadata attached to a backtest result: what was assumed and
    which data version produced it - neither engine carried this before."""
    return {
        "price_basis": "close",
        "cost_bps": float(cost_bps),
        "rebalance_days": int(rebalance_days),
        "evaluation_start": str(prices.index[start]),
        "evaluation_end": str(prices.index[end]),
        "risk_free_rate": 0.0,
        "data_version": price_key(prices),
    }


def equal_weight_benchmark(prices: pd.DataFrame, start: int, end: int) -> pd.Series:
    """An equal-weight buy-and-hold curve over the same window, for comparison.

    Only names that had listed by the window start are bought: a symbol
    whose real data begins later carries a flat back-filled placeholder
    before its `eligible_from` (data.clean_gappy_columns), and holding that
    would just dilute the benchmark with a fabricated zero return. No
    seasoning requirement applies here - buy-and-hold has no risk model to
    protect - and a name that later stops trading is held flat, i.e. sold at
    its last price without a fee, the buy-and-hold analogue of what the
    strategies do.
    """
    listed = start_positions(prices, eligibility_starts(prices))
    columns = [symbol for symbol in prices.columns if listed.get(symbol, 0) <= start]
    window = prices.iloc[start : end + 1].loc[:, columns]
    relatives = window / window.iloc[0]
    return relatives.mean(axis=1).rename("benchmark")


def evaluation_bounds(prices: pd.DataFrame, config: BacktestConfig) -> tuple[int, int]:
    if config.train_days < 166 or config.rebalance_days < 1 or config.forward_days < 1:
        raise ValueError("Training needs at least 166 rows; rebalance and forecast horizons must be positive.")
    if not np.isfinite(config.transaction_cost_bps) or not 0 <= config.transaction_cost_bps < 10000:
        raise ValueError("Transaction cost must be between zero and 10,000 bps.")
    if config.min_evaluation_days < MIN_EVALUATION_DAYS:
        raise ValueError(f"Evaluation requires at least {MIN_EVALUATION_DAYS} realized daily returns.")
    if not isinstance(prices.index, pd.DatetimeIndex) or not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError("Backtest requires unique, increasing dates.")
    if prices.shape[1] < 2 or not prices.columns.is_unique or "CASH" in prices.columns:
        raise ValueError("Backtest requires at least two unique risky assets.")
    if not np.isfinite(prices.to_numpy(dtype=float)).all() or (prices <= 0).any().any():
        raise ValueError("Backtest prices must be finite and positive.")
    start = int(prices.index.searchsorted(pd.Timestamp(config.evaluation_start))) if config.evaluation_start else config.train_days
    end = int(prices.index.searchsorted(pd.Timestamp(config.evaluation_end), side="right") - 1) if config.evaluation_end else len(prices) - 1
    if start < config.train_days:
        raise ValueError("The evaluation start must leave the full selected training history.")
    if end - start < config.min_evaluation_days or start >= len(prices) or end < 0:
        raise ValueError(f"Select an evaluation period with at least {config.min_evaluation_days} daily returns after training.")
    return start, end


def run_walk_forward_backtest(prices: pd.DataFrame, allocation_config: AllocationConfig,
                              backtest_config: BacktestConfig, *, _context=None,
                              cache_dir: Path | None = DEFAULT_CACHE) -> dict:
    start, end = evaluation_bounds(prices, backtest_config)
    prices = prices.iloc[:end + 1]
    symbols = list(prices.columns)
    # A symbol kept via clean_gappy_columns's trailing-gap carve-out (data.py)
    # has real data only through this cutoff date - converted to an index
    # position once here so the per-rebalance check below is a cheap integer
    # comparison, not a per-loop Timestamp search.
    cutoff_idx = cutoff_positions(prices, eligibility_cutoffs(prices))
    context = _context or build_context(prices, backtest_config.forward_days,
                                        allocation_config.method == "ML mean-variance", cache_dir)
    current = normalize_holdings(None, symbols)
    equity = gross_equity = 1.0
    # Array-based accumulation, not per-iteration DataFrame + to_dict("records"):
    # O03 profiling measured the to_dict("records") pattern here at ~10x slower
    # than building the four arrays directly and constructing one DataFrame at
    # the end (1435 daily rebalances, 1000 assets: 0.46s vs 0.05s) - the row-wise
    # Python-dict round trip dominated cost, not the actual per-day math.
    point_dates = [prices.index[start:start + 1]]
    point_equity = [np.array([1.0])]
    point_gross_equity = [np.array([1.0])]
    point_return = [np.array([np.nan])]
    records, targets, holdings, forecasts, validations = [], [], [], [], []
    holdings.append(current.rename(prices.index[start]))
    for idx in range(start, end, backtest_config.rebalance_days):
        decision = make_decision(context, idx, allocation_config, backtest_config.train_days,
                                 current, backtest_config.transaction_cost_bps)
        weights = decision["weights"]
        if cutoff_idx:
            # Past its cutoff, a symbol's price is a flat-filled placeholder,
            # not real data - redistribute any weight the solver still gave
            # it to CASH rather than dropping the symbol from the whole
            # backtest (which would let a LATE-window gap affect decisions
            # made before it - the actual lookahead-bias bug this fixes).
            # This is a pure post-hoc reduction: it can only move weight
            # toward CASH, so it can't violate max_weight, sector caps, or
            # the cash floor a decision already satisfied. A position that
            # is still open when its cutoff falls INSIDE the coming drift
            # interval is handled below by `liquidate_at_cutoff`, not here.
            ineligible = [s for s, cutoff in cutoff_idx.items() if idx > cutoff and s in weights.index]
            if ineligible:
                freed = float(weights.loc[ineligible].sum())
                if freed > 0:
                    weights = weights.copy()
                    weights.loc[ineligible] = 0.0
                    weights.loc["CASH"] += freed
        turnover = float((weights - current).abs().sum() / 2)
        asset_turnover = float((weights.drop("CASH") - current.drop("CASH")).abs().sum())
        cost_rate = rebalance_cost_rate(weights, current, backtest_config.transaction_cost_bps)
        cost_value = equity * cost_rate
        before = equity
        current = weights.copy()
        next_idx = min(idx + backtest_config.rebalance_days, end)
        # Buy-and-hold relatives give the entire interval's drift without Pandas row loops.
        relatives = prices.iloc[idx+1:next_idx+1].to_numpy() / prices.iloc[idx].to_numpy()
        values = np.column_stack([relatives * weights.reindex(symbols).to_numpy(),
                                  np.full(len(relatives), weights.CASH)])
        # A symbol whose eligibility cutoff falls inside this interval is
        # sold to CASH at its last real price on the first post-cutoff day,
        # fee charged there - not frozen at zero return until the next
        # rebalance with the exit billed as ordinary turnover then. The fee
        # is kept out of the gross path so "Cost drag" still measures fees.
        liquidation_fees, liquidated = (liquidate_at_cutoff(values, symbols, cutoff_idx, idx,
                                                            backtest_config.transaction_cost_bps)
                                        if cutoff_idx else (0.0, []))
        growth = values.sum(axis=1)
        net_values = equity * (1 - cost_rate) * growth
        gross_values = gross_equity * (growth + liquidation_fees)
        if liquidated:
            cost_value += equity * (1 - cost_rate) * float(liquidation_fees[-1])
        period_returns = net_values / np.r_[equity, net_values[:-1]] - 1
        dates = prices.index[idx+1:next_idx+1]
        point_dates.append(dates)
        point_equity.append(net_values)
        point_gross_equity.append(gross_values)
        point_return.append(period_returns)
        drifted = pd.DataFrame(values / growth[:, None], index=dates, columns=symbols + ["CASH"])
        holdings.append(drifted)
        current = drifted.iloc[-1].copy()
        current.name = None
        equity, gross_equity = float(net_values[-1]), float(gross_values[-1])
        solver = decision["solver"]
        records.append({"date": decision["date"], "train_start": prices.index[idx-backtest_config.train_days],
                        "train_end": decision["information_through"], "period_return": equity / before - 1,
                        "turnover": turnover, "asset_turnover": asset_turnover, "cost": cost_rate,
                        "cost_value": cost_value, "liquidated": ", ".join(liquidated),
                        "confidence": decision["confidence"],
                        "regime": decision["regime"], "effective_method": decision["effective_method"],
                        "reason": decision["reason"],
                        "solver_converged": solver.get("converged"), "solver_iterations": solver.get("iterations"),
                        "solver_objective": solver.get("objective"), "solver_mode": solver.get("mode"),
                        "solver_backend": solver.get("backend"), "solver_risk_model": solver.get("risk_model"),
                        "solver_fallback_reason": solver.get("fallback_reason"),
                        "solver_build_seconds": solver.get("build_seconds"),
                        "solver_solve_seconds": solver.get("solve_seconds"),
                        "validation_blocks": decision["validation_summary"]["blocks"], **decision["regime_stats"]})
        targets.append(weights.rename(decision["date"]))
        if allocation_config.method == "ML mean-variance":
            actual = (prices.iloc[idx+backtest_config.forward_days] / prices.iloc[idx] - 1
                      if idx+backtest_config.forward_days <= end else pd.Series(np.nan, index=symbols))
            forecasts.append(pd.DataFrame({"date": decision["date"], "symbol": symbols,
                                           "expected_return": decision["expected_returns"].to_numpy(),
                                           "adjusted_return": decision["adjusted_returns"].to_numpy(),
                                           "actual_return": actual.to_numpy(), "horizon_days": backtest_config.forward_days,
                                           # The covariance covers the investable names only
                                           # (decisions.MIN_LISTING_HISTORY_DAYS); a name still
                                           # warming up has no volatility estimate, so NaN.
                                           "annualized_volatility": pd.Series(
                                               np.sqrt(np.diag(decision["covariance"])),
                                               index=decision["covariance"].index).reindex(symbols).to_numpy(),
                                           "effective_method": decision["effective_method"]}))
            if not decision["validation"].empty:
                validations.append(decision["validation"].assign(decision_date=decision["date"]))
    curve = pd.DataFrame({
        "date": np.concatenate([d.to_numpy() for d in point_dates]),
        "equity": np.concatenate(point_equity),
        "gross_equity": np.concatenate(point_gross_equity),
        "return": np.concatenate(point_return),
    }).set_index("date")
    rebalances = pd.DataFrame(records)
    return {"equity_curve": curve, "rebalances": rebalances, "weights": pd.DataFrame(targets),
            "holdings": pd.concat([h.to_frame().T if isinstance(h, pd.Series) else h for h in holdings]),
            "final_holdings": current,
            "forecasts": pd.concat(forecasts, ignore_index=True) if forecasts else pd.DataFrame(),
            "validation": pd.concat(validations, ignore_index=True) if validations else pd.DataFrame(),
            "metrics": calculate_metrics(curve, rebalances),
            "assumptions": build_assumptions(prices, start, end, backtest_config.transaction_cost_bps,
                                             backtest_config.rebalance_days),
            "benchmark": equal_weight_benchmark(prices, start, end),
            "enforced_constraints": {
                "position_cap": True, "cash_buffer": allocation_config.cash_weight > 0,
                "equal_weight_only": False,
                "sector_caps": bool(allocation_config.sector_caps),
                "volatility_ceiling": allocation_config.max_volatility is not None,
                "turnover_budget": allocation_config.max_turnover < 1.0,
                "risk_model": decision["solver"].get("risk_model"),
                "min_trade_benefit_gating": allocation_config.min_trade_benefit_bps > 0,
            }}


def run_strategy_comparison(prices: pd.DataFrame, config: AllocationConfig,
                            backtest: BacktestConfig, max_workers: int = 4,
                            cache_dir: Path | None = DEFAULT_CACHE) -> dict:
    if max_workers < 1:
        raise ValueError("At least one strategy worker is required.")
    _, end = evaluation_bounds(prices, backtest)
    prices = prices.iloc[:end+1]
    # BLAS thread limits are process-global; serialize overlapping UI comparisons.
    with _comparison_lock, threadpool_limits(limits=1):
        context = build_context(prices, backtest.forward_days, True, cache_dir)

        def run(method):
            return run_walk_forward_backtest(prices, replace(config, method=method), backtest,
                                             _context=context, cache_dir=cache_dir)

        workers = min(max_workers, len(METHODS), os.cpu_count() or 1)
        if workers == 1:
            return {method: run(method) for method in METHODS}
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="portfolio") as executor:
            return dict(zip(METHODS, executor.map(run, METHODS)))


def calculate_metrics(equity_curve: pd.DataFrame, rebalances: pd.DataFrame) -> dict[str, float]:
    if equity_curve.empty or len(equity_curve) < 2:
        return {name: float("nan") for name in ("CAGR", "Volatility", "Sharpe", "Max drawdown",
                                                "Average turnover", "Cost drag", "Fees paid", "Final equity")}
    # The first row is initial capital, not an observed holding-period return.
    returns = equity_curve["return"].iloc[1:].dropna()
    years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
    initial, final = float(equity_curve.equity.iloc[0]), float(equity_curve.equity.iloc[-1])
    volatility = float(returns.std(ddof=1) * np.sqrt(252)) if len(returns) >= 2 else float("nan")
    if np.isfinite(volatility) and volatility < 1e-12:
        volatility = 0.0
    sharpe = float(returns.mean() * 252 / volatility) if volatility > 0 else float("nan")
    drawdown = equity_curve.equity / equity_curve.equity.cummax() - 1
    gross = float(equity_curve.gross_equity.iloc[-1]) if "gross_equity" in equity_curve else float("nan")
    return {"CAGR": (final / initial) ** (1 / years) - 1 if years > 0 else float("nan"),
            "Volatility": volatility, "Sharpe": sharpe, "Max drawdown": float(drawdown.min()),
            "Average turnover": float(rebalances.turnover.mean()) if not rebalances.empty else 0.0,
            "Cost drag": (gross - final) / initial,
            "Fees paid": float(rebalances.cost_value.sum()) / initial if "cost_value" in rebalances else 0.0,
            "Final equity": final / initial}
