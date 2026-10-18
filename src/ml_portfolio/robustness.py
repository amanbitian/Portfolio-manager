from __future__ import annotations

"""Automatic robustness / falsification checks for a Strategy Copilot DSL run.

Reruns a saved strategy under stress - higher costs, chronological
sub-periods, perturbed parameters, and a randomized-selection placebo - and
reports a structured evidence bundle plus a transparent, rule-based verdict,
rather than a single headline Sharpe or an opaque AI-generated score.

Scoped to Strategy Copilot's validated DSL (`kind="strategy_run"`), not the
ML allocator or the four-method Research comparison: its parameters are
small, numeric, and template-typed (a momentum lookback, an SMA window pair,
an RSI period/threshold), which is what makes "perturb by +/-20%" and
"rerun under 3x costs" well-defined operations without guessing at what a
parameter change even means for a fitted model.
"""

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from .backtest import MIN_EVALUATION_DAYS, calculate_metrics
from .data import eligibility_cutoffs, eligibility_starts
from .execution import cutoff_positions, rebalance_cost_rate, start_positions
from .optimizers import _proportional_weights
from .storage import atomic_parquet, load_artifact, price_key, write_artifact
from .strategy_copilot import run_strategy_dsl_backtest, validate_strategy_dsl

DEFAULT_COST_MULTIPLIERS = (1.0, 2.0, 3.0)
DEFAULT_PERTURBATION_PCT = 0.2
DEFAULT_SUBPERIODS = 2
DEFAULT_PLACEBO_TRIALS = 20
DEFAULT_BOOTSTRAP_TRIALS = 500

# Template-specific numeric fields that "perturb by +/-pct" is well-defined
# for - the DSL's own per-template field sets (strategy_copilot._ENTRY_KEYS).
_PERTURBABLE_FIELDS = {
    "momentum_top": ["lookback_days"],
    "sma_cross": ["fast_window", "slow_window"],
    "rsi_oversold": ["period", "threshold"],
}


def _bounds(prices, strategy, evaluation_start, evaluation_end, min_evaluation_days):
    """Resolve the same [start, end] integer window `run_strategy_dsl_backtest`
    itself computes, so every stress rerun operates on exactly the window the
    baseline used, not a slightly different one recomputed independently."""
    resolved = validate_strategy_dsl(strategy, prices)
    warmup = int(resolved["_warmup_days"])
    start = int(prices.index.searchsorted(pd.Timestamp(evaluation_start))) if evaluation_start else warmup
    end = (int(prices.index.searchsorted(pd.Timestamp(evaluation_end), side="right") - 1)
           if evaluation_end else len(prices) - 1)
    if start < warmup or start >= len(prices) or end < 0 or end - start < min_evaluation_days:
        raise ValueError("Evaluation window is not valid for this strategy.")
    return resolved, start, end


def cost_stress(prices, strategy, evaluation_start=None, evaluation_end=None, *,
                multipliers=DEFAULT_COST_MULTIPLIERS, max_weight=1.0, cash_weight=0.0) -> pd.DataFrame:
    """Rerun at each cost multiplier (1x is the baseline itself, included so
    the frame is self-contained). An edge with no real signal typically
    degrades roughly linearly with cost; a genuine one leaves headroom."""
    rows = []
    for multiplier in multipliers:
        stressed = deepcopy(strategy)
        stressed["transaction_cost_bps"] = float(strategy.get("transaction_cost_bps", 0.0)) * multiplier
        result = run_strategy_dsl_backtest(prices, stressed, evaluation_start=evaluation_start,
                                           evaluation_end=evaluation_end, max_weight=max_weight,
                                           cash_weight=cash_weight)
        rows.append({"cost_multiplier": multiplier, "cost_bps": stressed["transaction_cost_bps"],
                    **result["metrics"]})
    return pd.DataFrame(rows)


def subperiod_analysis(prices, strategy, evaluation_start=None, evaluation_end=None, *,
                       n_subperiods=DEFAULT_SUBPERIODS, min_evaluation_days=MIN_EVALUATION_DAYS,
                       max_weight=1.0, cash_weight=0.0) -> pd.DataFrame:
    """Split the evaluation window into `n_subperiods` contiguous chronological
    chunks and rerun each independently. A result driven by one lucky stretch
    looks very different here from one that holds up across the whole window."""
    if n_subperiods < 2:
        raise ValueError("Sub-period analysis needs at least two windows.")
    _, start, end = _bounds(prices, strategy, evaluation_start, evaluation_end, min_evaluation_days)
    edges = np.linspace(start, end, n_subperiods + 1).astype(int)
    rows = []
    for i in range(n_subperiods):
        sub_start, sub_end = int(edges[i]), int(edges[i + 1])
        row = {"subperiod": i + 1, "start": str(prices.index[sub_start]), "end": str(prices.index[sub_end])}
        if sub_end - sub_start < min_evaluation_days:
            rows.append({**row, "error": "Too short after splitting"})
            continue
        try:
            result = run_strategy_dsl_backtest(prices, strategy, evaluation_start=str(prices.index[sub_start]),
                                               evaluation_end=str(prices.index[sub_end]), max_weight=max_weight,
                                               cash_weight=cash_weight)
            rows.append({**row, **result["metrics"]})
        except ValueError as exc:
            rows.append({**row, "error": str(exc)})
    return pd.DataFrame(rows)


def parameter_perturbation(prices, strategy, evaluation_start=None, evaluation_end=None, *,
                           pct=DEFAULT_PERTURBATION_PCT, max_weight=1.0, cash_weight=0.0) -> pd.DataFrame:
    """Nudge each of the template's own numeric fields by +/-pct and rerun. A
    result that only exists at one exact parameter value is far weaker
    evidence than one that survives a neighborhood around it."""
    template = strategy["entry"]["template"]
    fields = _PERTURBABLE_FIELDS.get(template)
    if not fields:
        raise ValueError(f"No perturbable parameters known for template {template!r}.")
    baseline = run_strategy_dsl_backtest(prices, strategy, evaluation_start=evaluation_start,
                                         evaluation_end=evaluation_end, max_weight=max_weight,
                                         cash_weight=cash_weight)
    rows = [{"field": "baseline", "direction": "baseline", "value": None, **baseline["metrics"]}]
    for field in fields:
        original = strategy["entry"][field]
        for direction, factor in (("-", 1 - pct), ("+", 1 + pct)):
            # threshold (RSI) is a 0-100 float; every other perturbable field
            # is an integer window/lookback in trading days.
            new_value = (float(np.clip(original * factor, 1, 99)) if field == "threshold"
                        else max(2, int(round(original * factor))))
            if new_value == original:
                continue
            perturbed = deepcopy(strategy)
            perturbed["entry"][field] = new_value
            try:
                result = run_strategy_dsl_backtest(prices, perturbed, evaluation_start=evaluation_start,
                                                   evaluation_end=evaluation_end, max_weight=max_weight,
                                                   cash_weight=cash_weight)
                rows.append({"field": field, "direction": direction, "value": new_value, **result["metrics"]})
            except ValueError as exc:
                rows.append({"field": field, "direction": direction, "value": new_value, "error": str(exc)})
    return pd.DataFrame(rows)


def randomized_placebo(prices, strategy, evaluation_start=None, evaluation_end=None, *,
                       trials=DEFAULT_PLACEBO_TRIALS, max_weight=1.0, cash_weight=0.0, seed=42) -> pd.DataFrame:
    """How much of the strategy's apparent edge survives replacing its signal
    with a uniformly random pick of the same size, on the same rebalance
    schedule, costs, and weighting? If random picks perform comparably, the
    edge isn't coming from the signal.

    Deliberately a simplified execution model relative to
    `run_strategy_dsl_backtest`: candidates are drawn only from names that
    stay eligible (listed past warmup, not past their cutoff) through the
    END of the coming interval, so a placebo trial can never hold a name
    into a mid-interval forced liquidation - `liquidate_at_cutoff` is
    correctly inapplicable here by construction, not silently skipped.
    """
    resolved, start, end = _bounds(prices, strategy, evaluation_start, evaluation_end, MIN_EVALUATION_DAYS)
    warmup = int(resolved["_warmup_days"])
    max_positions = int(resolved["portfolio"]["max_positions"])
    rebalance_days = int(resolved["rebalance_days"])
    cost_bps = float(resolved["transaction_cost_bps"])
    symbols = list(prices.columns)
    listed = start_positions(prices, eligibility_starts(prices))
    cutoffs = cutoff_positions(prices, eligibility_cutoffs(prices))
    rng = np.random.default_rng(seed)

    rows = []
    for trial in range(trials):
        current = pd.Series(0.0, index=symbols + ["CASH"])
        current["CASH"] = 1.0
        equity = gross_equity = 1.0
        point_dates = [prices.index[start:start + 1]]
        point_equity = [np.array([1.0])]
        point_gross_equity = [np.array([1.0])]
        point_return = [np.array([np.nan])]
        rebalance_rows = []
        for idx in range(start, end, rebalance_days):
            next_idx = min(idx + rebalance_days, end)
            eligible = [s for s in symbols if listed.get(s, 0) + warmup <= idx
                       and (s not in cutoffs or cutoffs[s] > next_idx)]
            size = min(max_positions, len(eligible))
            chosen = list(rng.choice(eligible, size=size, replace=False)) if size else []
            target = pd.Series(0.0, index=symbols + ["CASH"])
            if chosen:
                picked_weights = _proportional_weights(np.ones(len(chosen)), 1.0 - cash_weight, max_weight)
                target.loc[chosen] = picked_weights
                target.CASH = 1.0 - float(picked_weights.sum())
            else:
                target.CASH = 1.0
            cost_rate = rebalance_cost_rate(target, current, cost_bps)
            before = equity
            relatives = prices.iloc[idx + 1:next_idx + 1].to_numpy() / prices.iloc[idx].to_numpy()
            risky = target.reindex(symbols).to_numpy()
            growth = relatives @ risky + target.CASH
            net_values = equity * (1 - cost_rate) * growth
            gross_values = gross_equity * growth
            period_returns = net_values / np.r_[equity, net_values[:-1]] - 1
            dates = prices.index[idx + 1:next_idx + 1]
            point_dates.append(dates)
            point_equity.append(net_values)
            point_gross_equity.append(gross_values)
            point_return.append(period_returns)
            turnover = float((target - current).abs().sum() / 2)
            rebalance_rows.append({"date": prices.index[idx], "turnover": turnover, "cost": cost_rate,
                                   "cost_value": before * cost_rate})
            if len(growth):
                last_values = np.r_[relatives[-1] * risky, target.CASH]
                current = pd.Series(last_values / last_values.sum(), index=symbols + ["CASH"])
                equity, gross_equity = float(net_values[-1]), float(gross_values[-1])
        curve = pd.DataFrame({
            "date": np.concatenate([d.to_numpy() for d in point_dates]),
            "equity": np.concatenate(point_equity),
            "gross_equity": np.concatenate(point_gross_equity),
            "return": np.concatenate(point_return),
        }).set_index("date")
        metrics = calculate_metrics(curve, pd.DataFrame(rebalance_rows))
        rows.append({"trial": trial, **metrics})
    return pd.DataFrame(rows)


def bootstrap_sharpe_ci(equity_curve: pd.DataFrame, *, trials=DEFAULT_BOOTSTRAP_TRIALS, block_days=21,
                        confidence=0.90, seed=42) -> dict:
    """Moving-block bootstrap (not i.i.d. resampling) confidence interval on
    Sharpe, from the realized daily returns of one already-run backtest.
    Blocks (default ~1 trading month) preserve short-horizon autocorrelation
    that resampling single days independently would destroy, which would
    otherwise understate the true sampling uncertainty."""
    returns = equity_curve["return"].iloc[1:].dropna().to_numpy()
    n = len(returns)
    if n < 30:
        raise ValueError("Not enough realized returns for a bootstrap confidence interval.")
    block = max(1, min(block_days, n // 2))
    n_blocks = int(np.ceil(n / block))
    rng = np.random.default_rng(seed)
    sharpes = np.full(trials, np.nan)
    for trial in range(trials):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        sample = np.concatenate([returns[s:s + block] for s in starts])[:n]
        vol = sample.std(ddof=1) * np.sqrt(252)
        if vol > 0:
            sharpes[trial] = sample.mean() * 252 / vol
    sharpes = sharpes[np.isfinite(sharpes)]
    if len(sharpes) == 0:
        return {"trials": 0, "mean_sharpe": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "fraction_positive": float("nan")}
    lower_q, upper_q = (1 - confidence) / 2, 1 - (1 - confidence) / 2
    return {"trials": int(len(sharpes)), "confidence": confidence, "mean_sharpe": float(sharpes.mean()),
            "ci_low": float(np.quantile(sharpes, lower_q)), "ci_high": float(np.quantile(sharpes, upper_q)),
            "fraction_positive": float((sharpes > 0).mean())}


def summarize_verdict(cost_stress_frame: pd.DataFrame, subperiod_frame: pd.DataFrame,
                      placebo_frame: pd.DataFrame, bootstrap: dict) -> dict:
    """A transparent, rule-based verdict from the evidence above - a count of
    which specific checks passed, not an opaque score. `falsified` requires
    either a non-positive baseline or that at most one check survives;
    `supported` requires at least four of the five; anything between is
    `inconclusive` - a real, honest outcome, not a failure to compute one.
    """
    baseline_sharpe = (cost_stress_frame.loc[cost_stress_frame["cost_multiplier"].idxmin(), "Sharpe"]
                       if "Sharpe" in cost_stress_frame and not cost_stress_frame.empty else float("nan"))
    stressed_sharpe = (cost_stress_frame.loc[cost_stress_frame["cost_multiplier"].idxmax(), "Sharpe"]
                       if "Sharpe" in cost_stress_frame and not cost_stress_frame.empty else float("nan"))
    subperiod_sharpes = subperiod_frame["Sharpe"].dropna() if "Sharpe" in subperiod_frame else pd.Series(dtype=float)
    placebo_sharpes = placebo_frame["Sharpe"].dropna() if "Sharpe" in placebo_frame else pd.Series(dtype=float)
    placebo_percentile = (float((placebo_sharpes < baseline_sharpe).mean())
                          if len(placebo_sharpes) and np.isfinite(baseline_sharpe) else float("nan"))
    checks = {
        "baseline_positive": bool(np.isfinite(baseline_sharpe) and baseline_sharpe > 0),
        "survives_max_cost_stress": bool(np.isfinite(stressed_sharpe) and stressed_sharpe > 0),
        "majority_subperiods_positive": bool(len(subperiod_sharpes) and (subperiod_sharpes > 0).mean() > 0.5),
        "beats_placebo_90th_percentile": bool(np.isfinite(placebo_percentile) and placebo_percentile >= 0.9),
        "bootstrap_ci_excludes_zero": bool(np.isfinite(bootstrap.get("ci_low", float("nan")))
                                           and bootstrap["ci_low"] > 0),
    }
    passed = sum(checks.values())
    if not checks["baseline_positive"] or passed <= 1:
        verdict = "falsified"
    elif passed >= 4:
        verdict = "supported"
    else:
        verdict = "inconclusive"
    return {"verdict": verdict, "checks": checks, "checks_passed": passed, "checks_total": len(checks)}


def run_robustness_checks(prices, strategy, evaluation_start=None, evaluation_end=None, *,
                          max_weight=1.0, cash_weight=0.0, cost_multipliers=DEFAULT_COST_MULTIPLIERS,
                          n_subperiods=DEFAULT_SUBPERIODS, perturbation_pct=DEFAULT_PERTURBATION_PCT,
                          placebo_trials=DEFAULT_PLACEBO_TRIALS, bootstrap_trials=DEFAULT_BOOTSTRAP_TRIALS,
                          seed=42) -> dict:
    """Run every check above against one saved Strategy Copilot run and
    return the full evidence bundle plus a transparent verdict."""
    baseline = run_strategy_dsl_backtest(prices, strategy, evaluation_start=evaluation_start,
                                         evaluation_end=evaluation_end, max_weight=max_weight,
                                         cash_weight=cash_weight)
    cost_frame = cost_stress(prices, strategy, evaluation_start, evaluation_end, multipliers=cost_multipliers,
                             max_weight=max_weight, cash_weight=cash_weight)
    try:
        subperiod_frame = subperiod_analysis(prices, strategy, evaluation_start, evaluation_end,
                                             n_subperiods=n_subperiods, max_weight=max_weight,
                                             cash_weight=cash_weight)
    except ValueError as exc:
        subperiod_frame = pd.DataFrame([{"error": str(exc)}])
    try:
        perturbation_frame = parameter_perturbation(prices, strategy, evaluation_start, evaluation_end,
                                                     pct=perturbation_pct, max_weight=max_weight,
                                                     cash_weight=cash_weight)
    except ValueError as exc:
        perturbation_frame = pd.DataFrame([{"error": str(exc)}])
    placebo_frame = randomized_placebo(prices, strategy, evaluation_start, evaluation_end, trials=placebo_trials,
                                       max_weight=max_weight, cash_weight=cash_weight, seed=seed)
    bootstrap = bootstrap_sharpe_ci(baseline["equity_curve"], trials=bootstrap_trials, seed=seed)
    verdict = summarize_verdict(cost_frame, subperiod_frame, placebo_frame, bootstrap)
    return {"baseline_metrics": baseline["metrics"], "cost_stress": cost_frame, "subperiods": subperiod_frame,
           "parameter_perturbation": perturbation_frame, "placebo": placebo_frame, "bootstrap": bootstrap,
           **verdict}


def verdict_conclusion(checks: dict) -> str:
    """A short human-readable summary of which checks passed/failed, meant to
    populate the Research Registry's free-text `conclusion` field - evidence
    a person reads, not just a status label."""
    lines = [f"{'PASS' if passed else 'FAIL'}: {name.replace('_', ' ')}" for name, passed in checks.items()]
    return "Automatic robustness checks - " + "; ".join(lines) + "."


def save_robustness_report(root: Path, name: str, prices: pd.DataFrame, source_strategy_run_id: str | None,
                           checks: dict) -> Path:
    """Persist a robustness evidence bundle, following the same
    artifact-envelope convention (schema_version=2) as every other saved
    result in this app. `source_strategy_run_id` links it back to the
    strategy_run it evaluated, mirroring `source_research_report_id`'s role
    on a strategy_run."""
    name = name.strip()
    if not name:
        raise ValueError("Enter a name for this robustness report.")
    root.mkdir(parents=True, exist_ok=True)
    key = "robustness_report_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:12]
    digest = price_key(prices)
    price_path = root / "prices" / f"{digest}.parquet"
    if not price_path.exists():
        atomic_parquet(prices, price_path)
    payload = {"schema_version": 2, "kind": "robustness_report", "id": key, "name": name,
              "saved_at": datetime.now(timezone.utc), "price_hash": digest,
              "source_strategy_run_id": source_strategy_run_id,
              "prices": {"$table": f"prices/{digest}.parquet"}, "checks": checks}
    return write_artifact(root, key, payload)


def load_robustness_report(path: Path) -> dict:
    payload = load_artifact(path)
    if payload.get("kind") != "robustness_report":
        raise ValueError("Unsupported robustness report.")
    return payload
