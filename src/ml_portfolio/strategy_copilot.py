from __future__ import annotations

import itertools
import re
from copy import deepcopy

import numpy as np
import pandas as pd

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .backtest import MIN_EVALUATION_DAYS, build_assumptions, calculate_metrics, equal_weight_benchmark
from .data import eligibility_cutoffs, eligibility_starts
from .execution import cutoff_positions, liquidate_at_cutoff, rebalance_cost_rate, start_positions
from .optimizers import _proportional_weights
from .storage import atomic_parquet, load_artifact, price_key, write_artifact


TEMPLATES = ("momentum_top", "sma_cross", "rsi_oversold")
SCHEMA_VERSION = 1  # DSL contract version - independent of storage.py's artifact envelope.


class StrategyParseError(ValueError):
    """Raised when a strategy prompt is unsupported or its numbers are ambiguous.

    Subclasses ValueError so existing `except ValueError` handlers keep working.
    """


def _validate_prices(prices: pd.DataFrame) -> None:
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("Strategy backtests require a DatetimeIndex.")
    if prices.empty or prices.shape[1] < 2:
        raise ValueError("Strategy backtests require at least two assets.")
    if not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError("Strategy backtests require unique, increasing dates.")
    if not prices.columns.is_unique:
        raise ValueError("Strategy backtests require unique symbols.")
    if "CASH" in prices.columns:
        raise ValueError("CASH is reserved and cannot be used as an asset symbol.")
    values = prices.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Strategy backtest prices must be finite and positive.")


# --- Prompt field extraction -------------------------------------------------
#
# Each helper removes its matched span from the text before returning, so a
# number consumed by one field (e.g. a position count) can never be
# misattributed to a later field (e.g. an SMA window). Extraction order is:
# position count, rebalance frequency, cost, then the template-specific fields.

_POSITION_PATTERNS = [
    r"top\s+(\d+)",
    r"hold\s+(\d+)\s*(?:positions|stocks|names|assets)",
    r"(\d+)\s*(?:positions|stocks|names|assets)",
]
_REBALANCE_KEYWORDS = {"daily": 1, "weekly": 5, "monthly": 21, "quarterly": 63}
_COST_PATTERN = r"(\d+(?:\.\d+)?)\s*(?:bps|basis points)"

_RSI_PERIOD_PATTERNS = [
    r"rsi\s*\(\s*(\d+)\s*\)",
    r"(\d+)[-\s]*day\s+rsi",
    r"rsi\s+period\s*(?:of|:)?\s*(\d+)",
]
_RSI_THRESHOLD_PATTERNS = [
    # The optional `(?:\(\s*\d+\s*\))?` lets a period specifier like "RSI(14)"
    # sit between "rsi" and "below" without its digits breaking the `[^0-9]*`
    # gap that follows - previously "RSI(14) below 20" failed this pattern
    # entirely (the "14" inside the parens is a digit, so `[^0-9]*` couldn't
    # span it) and silently fell through to the hardcoded default threshold
    # of 30 instead of raising or reading the 20 the user actually wrote.
    r"rsi\s*(?:\(\s*\d+\s*\))?[^0-9]*(?:<|below|under)\s*(\d+)",
    r"oversold[^0-9]*(\d+)",
]
_SMA_PAIR_PATTERNS = [
    r"(\d+)\s*/\s*(\d+)",
    r"fast\s*(\d+)\D*slow\s*(\d+)",
    r"(\d+)\s+and\s+(\d+)",
]
_MOMENTUM_MONTH_PATTERN = r"(\d+)[-\s]*(?:month|m)\s+momentum"
_MOMENTUM_DAY_PATTERN = r"(\d+)[-\s]*(?:day|session)\s+momentum"

# This DSL only ever produces long-only, rule-based, no-stop-loss strategies
# (see TEMPLATES/_assemble) - a prompt mentioning any of these has no way to
# be represented, so it must be rejected outright rather than silently
# compiled from whatever OTHER keywords it happens to also contain (e.g. a
# prompt asking for "momentum, short the bottom decile, 2x leverage, 5% stop
# loss" used to compile cleanly as a plain long-only momentum strategy, with
# the short/leverage/stop-loss request just discarded unmentioned).
_UNSUPPORTED_CONCEPTS = [
    (r"\bshort(?:ing|s|ed)?\b", "shorting/negative positions"),
    (r"\bleverage[d]?\b", "leverage"),
    (r"\bmargin\b", "margin"),
    (r"\bstop[\s-]?loss(?:es)?\b", "stop-loss orders"),
    (r"\btake[\s-]?profit\b", "take-profit orders"),
]


def _consume(text: str, pattern: str) -> tuple[str, re.Match | None]:
    match = re.search(pattern, text)
    if not match:
        return text, None
    return text[: match.start()] + text[match.end() :], match


def _extract_position_count(text: str, default: int) -> tuple[str, int, str]:
    for pattern in _POSITION_PATTERNS:
        text, match = _consume(text, pattern)
        if match:
            return text, int(match.group(1)), "prompt"
    return text, default, "default"


def _extract_rebalance_days(text: str, default: int) -> tuple[str, int, str]:
    text, match = _consume(text, r"rebalance\s+(?:every\s+)?(\d+)\s*days?")
    if match:
        return text, int(match.group(1)), "prompt"
    for keyword, days in _REBALANCE_KEYWORDS.items():
        text, match = _consume(text, rf"\b{keyword}\b")
        if match:
            return text, days, "prompt"
    return text, default, "default"


def _extract_cost_bps(text: str, default: float) -> tuple[str, float, str]:
    text, match = _consume(text, _COST_PATTERN)
    if match:
        return text, float(match.group(1)), "prompt"
    return text, default, "default"


def _parse_rsi(text: str) -> dict:
    period, period_source = 14, "default"
    for pattern in _RSI_PERIOD_PATTERNS:
        match = re.search(pattern, text)
        if match:
            period, period_source = int(match.group(1)), "prompt"
            break
    threshold, threshold_source = 30, "default"
    for pattern in _RSI_THRESHOLD_PATTERNS:
        match = re.search(pattern, text)
        if match:
            threshold, threshold_source = int(match.group(1)), "prompt"
            break
    if period_source == "default" and threshold_source == "default" and re.search(r"rsi\D*\d", text):
        # A bare "rsi <number>" with no period/day/paren keyword and no
        # below/under/oversold keyword is genuinely ambiguous - don't guess.
        raise StrategyParseError(
            "Ambiguous RSI parameter - say 'RSI period N' for the smoothing window "
            "or 'RSI below N' for the oversold threshold."
        )
    return {"period": period, "period_source": period_source,
            "threshold": threshold, "threshold_source": threshold_source}


def _parse_sma(text: str) -> dict:
    # Position count, rebalance frequency, and cost are already stripped from
    # `text` by this point, so any digit left over after removing a matched
    # fast/slow pair is a real extra number, not an already-attributed field.
    for pattern in _SMA_PAIR_PATTERNS:
        match = re.search(pattern, text)
        if match:
            remaining = text[: match.start()] + text[match.end() :]
            if re.search(r"\d", remaining):
                raise StrategyParseError(
                    "Ambiguous SMA windows - extra numbers found besides the fast/"
                    "slow pair. State only the two window lengths, e.g. 'SMA 50/200'."
                )
            return {"fast_window": int(match.group(1)), "slow_window": int(match.group(2)),
                    "source": "prompt"}
    numbers = re.findall(r"\b(\d{1,3})\b", text)
    if len(numbers) >= 1:
        # Even a SINGLE leftover number (e.g. "SMA 20") used to be accepted
        # and silently resolved to the 50/200 default, discarding the
        # number the user actually gave - genuinely ambiguous (is 20 meant
        # as the fast window? the slow window? something else entirely)
        # deserves the same rejection as the already-handled two-or-more case.
        raise StrategyParseError(
            "Ambiguous SMA windows - say e.g. 'SMA 50/200' or 'fast 50 slow 200'."
        )
    return {"fast_window": 50, "slow_window": 200, "source": "default"}


def _parse_momentum(text: str) -> dict:
    match = re.search(_MOMENTUM_MONTH_PATTERN, text)
    if match:
        return {"lookback_days": int(match.group(1)) * 21, "source": "prompt"}
    match = re.search(_MOMENTUM_DAY_PATTERN, text)
    if match:
        return {"lookback_days": int(match.group(1)), "source": "prompt"}
    return {"lookback_days": 126, "source": "default"}


VALID_FREQUENCIES = ("daily",)
EXECUTION_DESCRIPTION = "rebalance at next modeled interval using signals shifted by one trading day"


def _assemble(name: str, entry: dict, *, max_positions: int, rebalance_days: int,
             transaction_cost_bps: float, resolved_from: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "name": name,
        "universe": "loaded_prices",
        "frequency": VALID_FREQUENCIES[0],
        "entry": entry,
        "portfolio": {"max_positions": max_positions, "weighting": "equal"},
        "rebalance_days": rebalance_days,
        "transaction_cost_bps": float(transaction_cost_bps),
        "execution": EXECUTION_DESCRIPTION,
        "resolved_from": resolved_from,
    }


def compile_strategy_prompt(prompt: str, *, max_positions: int = 10,
                            rebalance_days: int = 21, transaction_cost_bps: float = 10.0) -> dict:
    """Compile a small, safe subset of strategy language into a DSL.

    A local LLM can replace this parser later, but the returned schema is the
    contract that keeps execution deterministic and reviewable. Numbers are
    attributed to a field only via an explicit, disjoint pattern for that
    field; ambiguous or unrecognized text raises StrategyParseError instead of
    guessing which field a number belongs to.
    """
    text = (prompt or "").strip().lower()
    if not np.isfinite(transaction_cost_bps) or transaction_cost_bps < 0:
        raise ValueError("Transaction cost must be non-negative.")

    for pattern, description in _UNSUPPORTED_CONCEPTS:
        if re.search(pattern, text):
            raise StrategyParseError(
                f"This prompt asks for {description}, which this DSL cannot represent "
                "(every template here is long-only, rule-based, with no stop rules). "
                "Remove that request, or describe a supported strategy directly "
                "(momentum, SMA cross, or RSI oversold)."
            )

    text, max_positions, positions_source = _extract_position_count(text, int(max(1, max_positions)))
    text, rebalance_days, rebalance_source = _extract_rebalance_days(text, int(max(1, rebalance_days)))
    text, transaction_cost_bps, cost_source = _extract_cost_bps(text, float(transaction_cost_bps))
    max_positions = int(max(1, max_positions))
    rebalance_days = int(max(1, rebalance_days))

    resolved_from = {
        "max_positions": positions_source,
        "rebalance_days": rebalance_source,
        "transaction_cost_bps": cost_source,
    }

    if "rsi" in text:
        rsi = _parse_rsi(text)
        entry = {"template": "rsi_oversold", "period": rsi["period"], "threshold": rsi["threshold"]}
        resolved_from["entry.period"] = rsi["period_source"]
        resolved_from["entry.threshold"] = rsi["threshold_source"]
        name = f"RSI oversold under {rsi['threshold']}"
    elif "cross" in text or "moving average" in text or "sma" in text:
        sma = _parse_sma(text)
        signal_mode = "cross" if "cross" in text else "trend"
        entry = {"template": "sma_cross", "fast_window": sma["fast_window"],
                 "slow_window": sma["slow_window"], "signal_mode": signal_mode}
        resolved_from["entry.windows"] = sma["source"]
        name = f"SMA {sma['fast_window']}/{sma['slow_window']} {signal_mode}"
    elif "momentum" in text:
        momentum = _parse_momentum(text)
        entry = {"template": "momentum_top", "lookback_days": momentum["lookback_days"],
                 "positive_only": True}
        resolved_from["entry.lookback_days"] = momentum["source"]
        name = f"Top {max_positions} momentum"
    else:
        raise StrategyParseError(
            "Unrecognized strategy prompt - use 'momentum', 'SMA cross', or 'RSI oversold'."
        )

    return _assemble(name, entry, max_positions=max_positions, rebalance_days=rebalance_days,
                     transaction_cost_bps=transaction_cost_bps, resolved_from=resolved_from)


def strategy_from_controls(template: str, *, fast_window: int = 50, slow_window: int = 200,
                           rsi_period: int = 14, rsi_threshold: int = 30,
                           momentum_lookback: int = 126, max_positions: int = 10,
                           rebalance_days: int = 21, transaction_cost_bps: float = 10.0,
                           signal_mode: str = "trend") -> dict:
    if template not in TEMPLATES:
        raise ValueError(f"Unknown strategy template: {template}")
    resolved_from = {"max_positions": "controls", "rebalance_days": "controls",
                     "transaction_cost_bps": "controls"}
    if template == "sma_cross":
        entry = {"template": template, "fast_window": int(fast_window), "slow_window": int(slow_window),
                 "signal_mode": signal_mode}
        resolved_from["entry.windows"] = "controls"
        name = f"SMA {int(fast_window)}/{int(slow_window)} {signal_mode}"
    elif template == "rsi_oversold":
        entry = {"template": template, "period": int(rsi_period), "threshold": int(rsi_threshold)}
        resolved_from["entry.period"] = "controls"
        resolved_from["entry.threshold"] = "controls"
        name = f"RSI oversold under {int(rsi_threshold)}"
    else:
        entry = {"template": template, "lookback_days": int(momentum_lookback), "positive_only": True}
        resolved_from["entry.lookback_days"] = "controls"
        name = f"Top {int(max_positions)} momentum"
    return _assemble(name, entry, max_positions=int(max_positions), rebalance_days=int(rebalance_days),
                     transaction_cost_bps=float(transaction_cost_bps), resolved_from=resolved_from)


_TOP_LEVEL_KEYS = {"schema_version", "name", "universe", "frequency", "entry", "portfolio",
                   "rebalance_days", "transaction_cost_bps", "execution", "resolved_from"}
_ENTRY_KEYS = {
    "momentum_top": {"template", "lookback_days", "positive_only"},
    "sma_cross": {"template", "fast_window", "slow_window", "signal_mode"},
    "rsi_oversold": {"template", "period", "threshold"},
}
_PORTFOLIO_KEYS = {"max_positions", "weighting"}


def validate_strategy_dsl(strategy: dict, prices: pd.DataFrame) -> dict:
    _validate_prices(prices)
    if not isinstance(strategy, dict):
        raise StrategyParseError("Strategy DSL must be a dictionary.")
    strategy = deepcopy(strategy)
    unknown_top = set(strategy) - _TOP_LEVEL_KEYS
    if unknown_top:
        raise StrategyParseError(f"Unknown strategy fields: {sorted(unknown_top)}.")
    if strategy.get("schema_version") != SCHEMA_VERSION:
        raise StrategyParseError(
            f"Unsupported strategy schema_version: {strategy.get('schema_version')!r} "
            f"(expected {SCHEMA_VERSION}).")
    if strategy.get("frequency") not in VALID_FREQUENCIES:
        raise StrategyParseError(f"Unsupported rebalance frequency: {strategy.get('frequency')!r}.")
    if strategy.get("execution") != EXECUTION_DESCRIPTION:
        raise StrategyParseError("Unrecognized execution description.")
    entry = strategy.get("entry")
    portfolio = strategy.get("portfolio")
    if not isinstance(entry, dict) or not isinstance(portfolio, dict):
        raise StrategyParseError("Strategy DSL requires entry and portfolio sections.")
    template = entry.get("template")
    if template not in TEMPLATES:
        raise StrategyParseError("Unsupported strategy template.")
    unknown_entry = set(entry) - _ENTRY_KEYS[template]
    if unknown_entry:
        raise StrategyParseError(f"Unknown fields for {template}: {sorted(unknown_entry)}.")
    unknown_portfolio = set(portfolio) - _PORTFOLIO_KEYS
    if unknown_portfolio:
        raise StrategyParseError(f"Unknown portfolio fields: {sorted(unknown_portfolio)}.")

    max_positions = int(portfolio.get("max_positions", 0))
    if max_positions < 1:
        raise ValueError("Strategy needs at least one position.")
    max_positions = min(max_positions, prices.shape[1])
    if portfolio.get("weighting", "equal") != "equal":
        raise ValueError("Only equal weighting is supported in the MVP copilot.")
    rebalance_days = int(strategy.get("rebalance_days", 0))
    if rebalance_days < 1:
        raise ValueError("Rebalance interval must be positive.")
    cost = float(strategy.get("transaction_cost_bps", 0.0))
    if not np.isfinite(cost) or not 0 <= cost < 10000:
        raise ValueError("Transaction cost must be between zero and 10,000 bps.")

    if template == "sma_cross":
        fast = int(entry.get("fast_window", 0))
        slow = int(entry.get("slow_window", 0))
        if fast < 2 or slow < 3 or fast >= slow:
            raise ValueError("SMA strategy requires 2 <= fast_window < slow_window.")
        signal_mode = entry.get("signal_mode", "trend")
        if signal_mode not in ("trend", "cross"):
            raise StrategyParseError("SMA signal_mode must be 'trend' or 'cross'.")
        strategy["entry"]["signal_mode"] = signal_mode
        warmup = slow + 1
    elif template == "rsi_oversold":
        period = int(entry.get("period", 0))
        threshold = float(entry.get("threshold", np.nan))
        if period < 2 or not 0 < threshold < 100:
            raise ValueError("RSI strategy requires period >= 2 and a threshold between 0 and 100.")
        warmup = period + 2
    else:
        lookback = int(entry.get("lookback_days", 0))
        if lookback < 2:
            raise ValueError("Momentum strategy requires at least two lookback days.")
        positive_only = entry.get("positive_only", True)
        if not isinstance(positive_only, bool):
            # `entry.get("positive_only", True)` used to be passed straight
            # into `_score_frame`'s `if entry.get("positive_only", True):`
            # check with no type guard - a JSON/CSV-sourced string "false" is
            # truthy in Python, so it silently behaved as enabled (True)
            # instead of being rejected or coerced.
            raise StrategyParseError(
                f"positive_only must be a boolean, got {positive_only!r}.")
        warmup = lookback + 1

    if len(prices) <= warmup + rebalance_days:
        raise ValueError("Not enough price history for this strategy and rebalance interval.")
    strategy["portfolio"]["max_positions"] = max_positions
    strategy["rebalance_days"] = rebalance_days
    strategy["transaction_cost_bps"] = cost
    strategy["_warmup_days"] = warmup
    return strategy


def _rsi(prices: pd.DataFrame, period: int) -> pd.DataFrame:
    """Wilder's RSI: an EMA (alpha=1/period) seeded by the first window's simple
    average, then recursed as avg[t] = (avg[t-1]*(period-1) + x[t]) / period.

    Degenerate-window conventions (undefined by Wilder's original formula):
    RSI=100 when there are gains but zero average loss (pure uptrend), RSI=50
    when there is neither gain nor loss (a flat/unchanged window) - both are
    stated choices, not universal standards.
    """
    delta = prices.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    seed_gain = gains.rolling(period).mean()
    seed_loss = losses.rolling(period).mean()

    # `diff()` leaves row 0 as NaN, so rolling(period)'s first full window lands
    # at row `period` (0-indexed) for every column - masking earlier rows to NaN
    # and setting that row to the rolling seed makes pandas' ewm treat it as the
    # recursion's first value, reproducing Wilder's formula without a Python loop.
    seeded_gain = gains.copy()
    seeded_gain.iloc[:period] = np.nan
    seeded_gain.iloc[period] = seed_gain.iloc[period]
    seeded_loss = losses.copy()
    seeded_loss.iloc[:period] = np.nan
    seeded_loss.iloc[period] = seed_loss.iloc[period]

    avg_gain = seeded_gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = seeded_loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain > 0)), 100.0)
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain == 0)), 50.0)
    return rsi


def _score_frame(prices: pd.DataFrame, strategy: dict) -> pd.DataFrame:
    entry = strategy["entry"]
    template = entry["template"]
    if template == "sma_cross":
        fast = prices.rolling(int(entry["fast_window"])).mean()
        slow = prices.rolling(int(entry["slow_window"])).mean()
        relation = fast - slow
        if entry.get("signal_mode", "trend") == "cross":
            rebalance_days = int(strategy.get("rebalance_days", 1))
            # A literal single-day crossover would almost always fall between
            # rebalance samples; treat "any upward cross in the trailing
            # rebalance window" as the event so the mode is actually usable.
            crossed_up = (relation.shift(1) <= 0) & (relation > 0)
            eligible = crossed_up.rolling(rebalance_days, min_periods=1).max().astype(bool)
            return (fast / slow - 1).where(eligible)
        return (fast / slow - 1).where(relation > 0)
    if template == "rsi_oversold":
        rsi = _rsi(prices, int(entry["period"]))
        return (100 - rsi).where(rsi < float(entry["threshold"]))
    lookback = int(entry["lookback_days"])
    momentum = prices / prices.shift(lookback) - 1
    return momentum.where(momentum > 0) if entry.get("positive_only", True) else momentum


def run_strategy_dsl_backtest(
    prices: pd.DataFrame,
    strategy: dict,
    *,
    evaluation_start: str | pd.Timestamp | None = None,
    evaluation_end: str | pd.Timestamp | None = None,
    min_evaluation_days: int = MIN_EVALUATION_DAYS,
    max_weight: float = 1.0,
    cash_weight: float = 0.0,
) -> dict:
    # Defaults (max_weight=1.0, cash_weight=0.0) reproduce this engine's
    # original always-equal-weight, never-hold-cash behavior exactly, so
    # every existing caller that doesn't pass these keeps working unchanged.
    # Real callers (app.py's Strategy Copilot page) pass the same
    # AllocationConfig.max_weight/cash_weight the rest of the app already
    # exposes in its sidebar - previously Copilot silently ignored both,
    # while its own `enforced_constraints` unconditionally claimed
    # `"cash_buffer": True` regardless of whether one was actually held.
    if not 0 < max_weight <= 1:
        raise ValueError("max_weight must be between 0 (exclusive) and 1.")
    if not 0 <= cash_weight < 1:
        raise ValueError("cash_weight must be between 0 and 1 (exclusive of 1).")
    strategy = validate_strategy_dsl(strategy, prices)
    scores = _score_frame(prices, strategy).shift(1)
    # A symbol kept via clean_gappy_columns's trailing-gap carve-out (data.py)
    # has real data only through this cutoff date - mask its score to NaN
    # from that point on so `row.head(max_positions)` below never selects it
    # again, without ever affecting selection at earlier, unaffected dates
    # (the actual lookahead-bias bug this fixes). A position still held when
    # its cutoff arrives is sold to CASH on the first post-cutoff day by
    # `liquidate_at_cutoff` in the drift step below, same as backtest.py.
    cutoff_idx = cutoff_positions(prices, eligibility_cutoffs(prices))
    for symbol, cutoff_pos in cutoff_idx.items():
        if symbol in scores.columns:
            scores.loc[scores.index[cutoff_pos + 1:], symbol] = np.nan
    warmup = int(strategy["_warmup_days"])
    # A symbol that listed inside the window carries a flat back-filled
    # placeholder before its first real print (data.clean_gappy_columns), so
    # its indicators there are fabricated (RSI 50, zero momentum, price on
    # its own SMA) and its first real `warmup` rows blend placeholder and
    # real prints. Mask its score through listing + warm-up - the same
    # seasoning the strategy already demands of the frame's own start - so
    # it is simply not a candidate until then, and never selected on a
    # placeholder. (The indicators are computed on the full frame, so the
    # cross-sectional ranking of everyone else is unaffected.)
    for symbol, listed_pos in start_positions(prices, eligibility_starts(prices)).items():
        if symbol in scores.columns:
            scores.loc[scores.index[:listed_pos + warmup + 1], symbol] = np.nan
    start = int(prices.index.searchsorted(pd.Timestamp(evaluation_start))) if evaluation_start else warmup
    end = int(prices.index.searchsorted(pd.Timestamp(evaluation_end), side="right") - 1) if evaluation_end else len(prices) - 1
    if start < warmup or start >= len(prices) or end < 0:
        raise ValueError("Evaluation start must leave enough indicator history.")
    if end - start < min_evaluation_days:
        raise ValueError(f"Select an evaluation period with at least {min_evaluation_days} daily returns.")

    symbols = list(prices.columns)
    current = pd.Series(0.0, index=symbols + ["CASH"])
    current.CASH = 1.0
    equity = gross_equity = 1.0
    # Array-based accumulation, not per-iteration DataFrame + to_dict("records") -
    # see execution.rebalance_cost_rate's docstring and README's O03 changelog
    # entry for the profiling behind this (same pattern, same ~10x measured gain,
    # duplicated from backtest.py's walk-forward loop).
    point_dates = [prices.index[start:start + 1]]
    point_equity = [np.array([1.0])]
    point_gross_equity = [np.array([1.0])]
    point_return = [np.array([np.nan])]
    rebalances, weights = [], []
    cost_bps = float(strategy["transaction_cost_bps"])
    max_positions = int(strategy["portfolio"]["max_positions"])
    rebalance_days = int(strategy["rebalance_days"])

    for idx in range(start, end, rebalance_days):
        row = scores.iloc[idx].dropna().sort_values(ascending=False)
        chosen = list(row.head(max_positions).index)
        target = pd.Series(0.0, index=symbols + ["CASH"])
        if chosen:
            # `_proportional_weights` (optimizers.py) distributes `1-cash_weight`
            # equally across `chosen` subject to `max_weight`, redistributing
            # any capped-away excess to the still-uncapped names - the same
            # capping/redistribution logic the rest of the app already uses,
            # reused here rather than reinvented. Any budget it can't place
            # (e.g. max_weight so tight that even every name at the cap can't
            # absorb 1-cash_weight) flows back to CASH, never disappears.
            picked_weights = _proportional_weights(np.ones(len(chosen)), 1.0 - cash_weight, max_weight)
            target.loc[chosen] = picked_weights
            target.CASH = 1.0 - float(picked_weights.sum())
        else:
            target.CASH = 1.0
        cost_rate = rebalance_cost_rate(target, current, cost_bps)
        before = equity
        cost_value = before * cost_rate
        next_idx = min(idx + rebalance_days, end)
        relatives = prices.iloc[idx + 1:next_idx + 1].to_numpy() / prices.iloc[idx].to_numpy()
        risky = target.reindex(symbols).to_numpy()
        liquidated: list[str] = []
        if cutoff_idx:
            # Same forced-sale-at-cutoff treatment as backtest.py's loop: the
            # value matrix is only materialized when a cutoff exists at all,
            # so the common no-cutoff path keeps its matrix-vector product.
            values = np.column_stack([relatives * risky, np.full(len(relatives), target.CASH)])
            liquidation_fees, liquidated = liquidate_at_cutoff(values, symbols, cutoff_idx, idx, cost_bps)
            growth = values.sum(axis=1)
            gross_growth = growth + liquidation_fees
            last_values = values[-1] if len(values) else None
        else:
            growth = relatives @ risky + target.CASH
            gross_growth = growth
            last_values = np.r_[relatives[-1] * risky, target.CASH] if len(growth) else None
        net_values = equity * (1 - cost_rate) * growth
        gross_values = gross_equity * gross_growth
        if liquidated:
            cost_value += equity * (1 - cost_rate) * float(liquidation_fees[-1])
        period_returns = net_values / np.r_[equity, net_values[:-1]] - 1
        dates = prices.index[idx + 1:next_idx + 1]
        point_dates.append(dates)
        point_equity.append(net_values)
        point_gross_equity.append(gross_values)
        point_return.append(period_returns)
        drifted = pd.Series(0.0, index=symbols + ["CASH"])
        if len(growth):
            drifted = pd.Series(last_values / last_values.sum(), index=symbols + ["CASH"])
            equity, gross_equity = float(net_values[-1]), float(gross_values[-1])
        turnover = float((target - current).abs().sum() / 2)
        rebalances.append({
            "date": prices.index[idx],
            "information_through": prices.index[idx - 1],
            "selected": ", ".join(chosen),
            "positions": len(chosen),
            "turnover": turnover,
            "cost": cost_rate,
            "cost_value": cost_value,
            "liquidated": ", ".join(liquidated),
            "period_return": equity / before - 1,
        })
        weights.append(target.rename(prices.index[idx]))
        current = drifted

    curve = pd.DataFrame({
        "date": np.concatenate([d.to_numpy() for d in point_dates]),
        "equity": np.concatenate(point_equity),
        "gross_equity": np.concatenate(point_gross_equity),
        "return": np.concatenate(point_return),
    }).set_index("date")
    rebalances_frame = pd.DataFrame(rebalances)
    return {
        "strategy": {key: value for key, value in strategy.items() if not key.startswith("_")},
        "equity_curve": curve,
        "rebalances": rebalances_frame,
        "weights": pd.DataFrame(weights),
        "metrics": calculate_metrics(curve, rebalances_frame),
        "latest_scores": scores.iloc[end].sort_values(ascending=False).dropna().rename_axis("symbol").reset_index(name="score"),
        "assumptions": build_assumptions(prices, start, end, cost_bps, rebalance_days),
        "benchmark": equal_weight_benchmark(prices, start, end),
        "enforced_constraints": {
            # Derived from what THIS call actually applied, not a static claim -
            # a prior version of this dict unconditionally reported
            # "cash_buffer": True even though this engine, by default, always
            # allocated 100% to picks with zero cash reserve.
            "position_cap": True, "cash_buffer": cash_weight > 0,
            "max_weight_cap": max_weight < 1.0, "equal_weight_only": max_weight >= 1.0,
            "sector_caps": False, "volatility_ceiling": False, "turnover_budget": False,
            "risk_model": None, "min_trade_benefit_gating": False,
        },
    }


def save_strategy_run(root: Path, name: str, prices, result: dict, *,
                      source_research_report_id: str | None = None) -> Path:
    """Persist a Strategy Copilot backtest, following the same artifact-envelope
    convention as save_proposal/save_research (schema_version=2 - a different
    namespace from the DSL's own internal schema_version=1 field already
    nested at result['strategy']['schema_version']). Prices are content-addressed
    and shared across saved artifacts exactly like save_proposal/save_research
    already do, so re-saving against the same price data doesn't duplicate it.

    `source_research_report_id`, when given, is the artifact id of a saved
    Research Agent report (research_agent.save_research_report) whose
    hypothesis led to this run - always stored (as a string or None, never
    omitted) so a saved run can always be traced back to the research that
    inspired it, or explicitly show it wasn't derived from a saved report."""
    name = name.strip()
    if not name:
        raise ValueError("Enter a name for this strategy run.")
    root.mkdir(parents=True, exist_ok=True)
    key = "strategy_run_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:12]
    digest = price_key(prices)
    price_path = root / "prices" / f"{digest}.parquet"
    if not price_path.exists():
        atomic_parquet(prices, price_path)
    payload = {"schema_version": 2, "kind": "strategy_run", "id": key, "name": name,
              "saved_at": datetime.now(timezone.utc), "price_hash": digest,
              "source_research_report_id": source_research_report_id,
              "prices": {"$table": f"prices/{digest}.parquet"}, "result": result}
    return write_artifact(root, key, payload)


def load_strategy_run(path: Path) -> dict:
    payload = load_artifact(path)
    if payload.get("kind") != "strategy_run":
        raise ValueError("Unsupported strategy run.")
    return payload


def strategy_runs(root: Path) -> dict[Path, str]:
    choices = {}
    for path in sorted(root.glob("strategy_run_*.json"), reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            choices[path] = f"{payload['name']} | {payload['saved_at']['$timestamp']}"
        except (ValueError, KeyError, TypeError, OSError):
            choices[path] = f"Unreadable strategy run: {path.stem}"
    return choices


MAX_SWEEP_COMBINATIONS = 24  # "bounded" per the backlog - a quick research tool, not an unbounded grid search


def sweep_strategy(
    prices: pd.DataFrame,
    template: str,
    fixed_params: dict,
    param_grid: dict[str, list],
    selection_end: str | pd.Timestamp,
    final_test_start: str | pd.Timestamp,
    final_test_end: str | pd.Timestamp | None = None,
    selection_metric: str = "Sharpe",
) -> dict:
    """Bounded parameter sweep with a chronological selection/final-test split.

    Every combination in `param_grid` (a dict of sweepable field -> candidate
    values) is run over the SELECTION window (start defaults to each
    combination's own indicator warmup, end at `selection_end`) and every
    result is reported - not just the best - to expose selection bias instead
    of hiding it. Combinations that fail validation (e.g. not enough history
    for that combination's own warmup) are recorded with an `error` column
    rather than aborting the whole sweep. The single best-by-`selection_metric`
    surviving combination (the "champion") is then re-run, once, on the
    separate FINAL TEST window (`final_test_start` to `final_test_end`,
    required to start strictly after `selection_end` so it is never used for
    selection) and that result is reported as the honest out-of-sample check.
    """
    if pd.Timestamp(final_test_start) <= pd.Timestamp(selection_end):
        raise ValueError("The final test window must start strictly after the selection window ends.")
    if not param_grid:
        raise ValueError("Provide at least one parameter to sweep.")
    names = list(param_grid.keys())
    combinations = list(itertools.product(*param_grid.values()))
    if not combinations:
        raise ValueError("Provide at least one candidate value for every swept parameter.")
    if len(combinations) > MAX_SWEEP_COMBINATIONS:
        raise ValueError(f"Sweep is bounded to {MAX_SWEEP_COMBINATIONS} combinations "
                         f"({len(combinations)} requested) - narrow the grid.")

    rows = []
    for combo in combinations:
        params = dict(zip(names, combo))
        row = dict(params)
        try:
            strategy = strategy_from_controls(template, **fixed_params, **params)
            result = run_strategy_dsl_backtest(prices, strategy, evaluation_end=selection_end)
            row.update(result["metrics"])
        except ValueError as exc:
            row["error"] = str(exc)
        rows.append(row)
    variants = pd.DataFrame(rows)

    valid = variants[variants["error"].isna()] if "error" in variants else variants
    if valid.empty:
        raise ValueError("No swept combination produced a valid backtest in the selection window.")
    if selection_metric not in valid.columns or valid[selection_metric].notna().sum() == 0:
        raise ValueError(f"No swept combination produced a finite {selection_metric} in the selection window.")
    best_index = valid[selection_metric].idxmax()
    # All current template parameters are integer-valued; plain-int cast avoids
    # leaking numpy scalar types (e.g. np.int64) into saved/displayed results.
    selected_params = {name: int(valid.loc[best_index, name]) for name in names}

    champion = strategy_from_controls(template, **fixed_params, **selected_params)
    final_test_result = run_strategy_dsl_backtest(prices, champion, evaluation_start=final_test_start,
                                                   evaluation_end=final_test_end)
    return {
        "template": template, "fixed_params": fixed_params, "param_grid": param_grid,
        "selection_end": str(pd.Timestamp(selection_end)), "final_test_start": str(pd.Timestamp(final_test_start)),
        "final_test_end": str(pd.Timestamp(final_test_end)) if final_test_end is not None else None,
        "selection_metric": selection_metric, "variants": variants,
        "selected_params": selected_params, "final_test_result": final_test_result,
    }


def save_strategy_sweep(root: Path, name: str, prices: pd.DataFrame, sweep: dict) -> Path:
    """Persist a parameter sweep (kind='strategy_sweep', schema_version=2) so it
    can be replayed exactly - the full variants table, the selected champion,
    and its separately-reported final-test result, without recomputing
    anything with knowledge the original selection didn't have."""
    name = name.strip()
    if not name:
        raise ValueError("Enter a name for this sweep.")
    root.mkdir(parents=True, exist_ok=True)
    key = "strategy_sweep_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:12]
    digest = price_key(prices)
    price_path = root / "prices" / f"{digest}.parquet"
    if not price_path.exists():
        atomic_parquet(prices, price_path)
    payload = {"schema_version": 2, "kind": "strategy_sweep", "id": key, "name": name,
              "saved_at": datetime.now(timezone.utc), "price_hash": digest,
              "prices": {"$table": f"prices/{digest}.parquet"}, "sweep": sweep}
    return write_artifact(root, key, payload)


def load_strategy_sweep(path: Path) -> dict:
    payload = load_artifact(path)
    if payload.get("kind") != "strategy_sweep":
        raise ValueError("Unsupported strategy sweep.")
    return payload


def strategy_sweeps(root: Path) -> dict[Path, str]:
    choices = {}
    for path in sorted(root.glob("strategy_sweep_*.json"), reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            choices[path] = f"{payload['name']} | {payload['saved_at']['$timestamp']}"
        except (ValueError, KeyError, TypeError, OSError):
            choices[path] = f"Unreadable strategy sweep: {path.stem}"
    return choices
