from __future__ import annotations

from dataclasses import dataclass, field, replace
from collections import OrderedDict
from pathlib import Path
from threading import Lock

import numpy as np
import pandas as pd

from .data import eligibility_starts
from .execution import start_positions
from .features import build_feature_panel, daily_returns, feature_matrix_at, forward_returns
from .models import detect_regime, forecast_confidence, ridge_daily_statistics, RidgeWindow, shrinkage_covariance
from .optimizers import AllocationConfig, allocate_portfolio, risk_checks
from .storage import CACHE_VERSION, DEFAULT_CACHE, cached_frame, price_key


# A symbol that listed inside the loaded window (data.clean_gappy_columns
# records its first real date in `.attrs["eligible_from"]`) carries a flat
# back-filled placeholder before that date. To a risk model a flat price is a
# zero-volatility, zero-correlation asset - exactly what a mean-variance
# solver would pile into - so a symbol only enters the optimizer's candidate
# set once it has this many real sessions behind the decision date. 252
# matches the covariance estimator's own window (prepare_risk_inputs uses the
# trailing 252 returns), so once eligible, a symbol's variance estimate never
# contains a single placeholder return. Until then it is held at exactly zero
# weight and listed in the decision's `ineligible` field, not silently sized.
MIN_LISTING_HISTORY_DAYS = 252


@dataclass
class DecisionContext:
    prices: pd.DataFrame
    returns: pd.DataFrame
    features: pd.DataFrame
    targets: pd.Series
    horizon: int
    ridge: RidgeWindow | None = None
    predictions: dict = field(default_factory=dict)
    risk_inputs: OrderedDict = field(default_factory=OrderedDict)
    risk_lock: object = field(default_factory=Lock)
    listing_positions: dict = field(default_factory=dict)

    def investable(self, idx: int) -> list[str]:
        """Symbols the decision at row `idx` may allocate to: everything
        except names still inside their post-listing MIN_LISTING_HISTORY_DAYS
        warm-up (see the constant above). A symbol present from the frame's
        first row is always investable here - its history is governed by
        `train_days`, exactly as before listings were supported; trailing
        cutoffs are applied post-hoc by the walk-forward loop instead, see
        backtest.py."""
        return [symbol for symbol in self.prices.columns
                if symbol not in self.listing_positions
                or self.listing_positions[symbol] + MIN_LISTING_HISTORY_DAYS <= idx]


def build_context(prices: pd.DataFrame, horizon: int, ml: bool = True,
                  cache_dir: Path | None = DEFAULT_CACHE) -> DecisionContext:
    # The edge-eligibility attrs change what build_feature_panel/forward_returns
    # produce for the very same price VALUES (a back-filled flat prefix is a
    # placeholder, not history), so they must be part of the cache key -
    # price_key hashes values and columns only.
    eligibility = (sorted((prices.attrs.get("eligible_from") or {}).items()),
                   sorted((prices.attrs.get("eligible_through") or {}).items()))
    key = price_key(prices, CACHE_VERSION, eligibility)
    label_key = price_key(prices, CACHE_VERSION, eligibility, horizon)
    returns = cached_frame(cache_dir, "returns", key, lambda: daily_returns(prices))
    features, targets, ridge = pd.DataFrame(), pd.Series(dtype=float), None
    if ml:
        features = cached_frame(cache_dir, "features", key, lambda: build_feature_panel(prices))
        targets = cached_frame(cache_dir, "targets", label_key,
                               lambda: forward_returns(prices, horizon).to_frame()).iloc[:, 0]
        statistics = cached_frame(cache_dir, "ridge", label_key,
                                  lambda: ridge_daily_statistics(features, targets))
        ridge = RidgeWindow(statistics, features.columns)
    return DecisionContext(prices, returns, features, targets, horizon, ridge,
                           listing_positions=start_positions(prices, eligibility_starts(prices)))


def _predict(context: DecisionContext, origin: int, train_days: int):
    key = (origin, train_days)
    if key not in context.predictions:
        first, last = max(126, origin - train_days), origin - 1 - context.horizon
        symbols = list(context.prices.columns)
        if last < first or context.ridge is None:
            return pd.Series(0.0, index=symbols, name="expected_return"), 0.0, 0
        context.predictions[key] = context.ridge.predict(
            context.prices.index[first], context.prices.index[last],
            feature_matrix_at(context.features, context.prices.index[origin]), symbols)
    return context.predictions[key]


def _training(context: DecisionContext, origin: int, train_days: int):
    first = max(126, origin - train_days)
    last = origin - 1 - context.horizon
    if last < first:
        return context.features.iloc[:0], context.targets.iloc[:0]
    dates = context.features.index.get_level_values("date")
    mask = (dates >= context.prices.index[first]) & (dates <= context.prices.index[last])
    features = context.features.loc[mask]
    target = context.targets.reindex(features.index)
    return features, target


def prepare_risk_inputs(context: DecisionContext, idx: int, train_days: int,
                        symbols: list[str] | None = None):
    symbols = list(context.prices.columns) if symbols is None else list(symbols)
    key = (idx, train_days, tuple(symbols))
    with context.risk_lock:
        if key not in context.risk_inputs:
            trailing = context.returns.iloc[max(0, idx - train_days):idx]
            covariance = shrinkage_covariance(trailing.tail(252), symbols)
            # The regime reads the market of names that actually traded; a
            # pre-listing placeholder's zero returns would only dampen it.
            regime, statistics = detect_regime(trailing.tail(126).loc[:, symbols] if symbols
                                               else trailing.tail(126))
            context.risk_inputs[key] = (covariance, regime, statistics)
        context.risk_inputs.move_to_end(key)
        capacity = max(1, min(32, 64 * 1024**2 // max(1, 8 * context.prices.shape[1]**2)))
        while len(context.risk_inputs) > capacity:
            context.risk_inputs.popitem(last=False)
        return context.risk_inputs[key]


def validation_summary(records: pd.DataFrame) -> dict:
    empty = {"eligible": False, "confidence": 0.0, "blocks": 0, "skill": 0.0,
             "block_win_rate": 0.0, "rank_ic": 0.0, "rmse": None, "baseline_rmse": None,
             "hit_rate": None, "reason": "Fewer than three matured validation blocks"}
    if records.empty:
        return empty
    error = (records["prediction"] - records["actual"]) ** 2
    base_error = (records["baseline"] - records["actual"]) ** 2
    rmse, baseline_rmse = float(np.sqrt(error.mean())), float(np.sqrt(base_error.mean()))
    skill = 1 - rmse / baseline_rmse if baseline_rmse > 1e-12 else 0.0
    wins, correlations = [], []
    for _, group in records.groupby("date"):
        wins.append(float(((group.prediction - group.actual) ** 2).mean()) <
                    float(((group.baseline - group.actual) ** 2).mean()))
        a, b = group.prediction.rank(), group.actual.rank()
        correlations.append(float(a.corr(b)) if a.std() > 0 and b.std() > 0 else 0.0)
    eligible = len(wins) >= 3 and skill > 0 and np.mean(wins) > .5 and np.mean(correlations) > 0
    return {"eligible": bool(eligible), "confidence": forecast_confidence(rmse, baseline_rmse, bool(eligible)),
            "blocks": len(wins), "skill": skill, "block_win_rate": float(np.mean(wins)),
            "rank_ic": float(np.mean(correlations)), "rmse": rmse, "baseline_rmse": baseline_rmse,
            "hit_rate": float((np.sign(records.prediction) == np.sign(records.actual)).mean()),
            "reason": "ML passed chronological validation" if eligible else
            (empty["reason"] if len(wins) < 3 else "ML did not consistently beat the historical-mean forecast")}


def validate_forecast(context: DecisionContext, idx: int, train_days: int):
    symbols = list(context.prices.columns)
    # Non-overlapping validation outcomes must be fully known before this decision.
    first = max(idx - train_days + 126, 126) + 40 + context.horizon
    origins = list(range(first, idx - context.horizon, context.horizon))[-6:]
    frames = []
    for origin in origins:
        prediction, baseline, days = _predict(context, origin, train_days)
        if days < 40:
            continue
        date = context.prices.index[origin]
        actual = context.targets.xs(date, level="date").reindex(symbols)
        frames.append(pd.DataFrame({"date": date, "symbol": symbols, "prediction": prediction.to_numpy(),
                                    "actual": actual.to_numpy(), "baseline": baseline}))
    records = pd.concat(frames, ignore_index=True).dropna() if frames else pd.DataFrame()
    return validation_summary(records), records


def make_decision(context: DecisionContext, idx: int, config: AllocationConfig,
                  train_days: int, current_weights: pd.Series | None = None,
                  transaction_cost_bps: float = 0.0) -> dict:
    if idx < train_days or idx >= len(context.prices):
        raise ValueError("Decision date needs the full training window and a price reference.")
    symbols = list(context.prices.columns)
    # Names still inside their post-listing warm-up are not candidates at all:
    # the covariance, the solver and the risk checks all run on `investable`
    # only, so a flat placeholder can never look like a free zero-risk asset.
    # They come back at exactly zero weight, and are named in `ineligible`.
    investable = context.investable(idx)
    ineligible = [symbol for symbol in symbols if symbol not in set(investable)]
    trailing = context.returns.iloc[max(0, idx - train_days):idx]
    covariance, regime, regime_stats = prepare_risk_inputs(context, idx, train_days, investable)
    expected = pd.Series(0.0, index=symbols, name="expected_return")
    summary = {"eligible": False, "confidence": 0.0, "blocks": 0, "reason": "Baseline selected"}
    validation = pd.DataFrame()
    effective = replace(config, forecast_horizon_days=context.horizon)
    if config.method == "ML mean-variance":
        summary, validation = validate_forecast(context, idx, train_days)
        expected, _, _ = _predict(context, idx, train_days)
        if not summary["eligible"]:
            effective = replace(effective, method="Equal weight")
    adjusted = expected * summary["confidence"] if config.method == "ML mean-variance" else expected
    if ineligible:
        # No forecast is claimed for a name the model was never allowed to
        # size - NaN, not a fabricated 0.0, in the recorded forecasts.
        expected = expected.copy()
        adjusted = adjusted.copy()
        expected.loc[ineligible] = np.nan
        adjusted.loc[ineligible] = np.nan
    if investable:
        current_investable = None
        if current_weights is not None:
            # A live holding in a not-yet-eligible name (a listing under a
            # year old entered as an existing position) cannot be sized, so
            # it is proposed for sale: to the solver that money is cash
            # already. The sell leg still counts in the post-solve turnover
            # check below, which uses the true holdings, so a tight
            # `max_turnover` reports the forced exit rather than hiding it.
            current_investable = current_weights.reindex(investable + ["CASH"]).fillna(0.0)
            current_investable.loc["CASH"] += float(current_weights.sum() - current_investable.sum())
        weights = allocate_portfolio(adjusted.loc[investable], covariance,
                                     trailing.tail(126).loc[:, investable], effective,
                                     current_investable, transaction_cost_bps)
        checks = risk_checks(weights, covariance, effective, current_weights)
    else:
        # Every name is still warming up: the only allocation is cash, and
        # there is nothing for a solver or the risk checks to look at.
        weights = pd.Series({"CASH": 1.0})
        weights.attrs["solver"] = {"mode": "cash-only", "converged": True,
                                   "fallback_reason": "no symbol has enough listed history yet"}
        checks = pd.DataFrame([{"check": "Weight total", "value": 1.0, "limit": 1.0, "passed": True},
                               {"check": "Cash reserve", "value": 1.0, "limit": float(config.cash_weight),
                                "passed": True}])
    solver = weights.attrs.get("solver", {})
    weights = weights.reindex(symbols + ["CASH"]).fillna(0.0)
    weights.attrs["solver"] = solver
    return {"date": context.prices.index[idx], "information_through": context.prices.index[idx - 1],
            "weights": weights, "expected_returns": expected, "adjusted_returns": adjusted,
            "covariance": covariance, "confidence": summary["confidence"], "regime": regime,
            "regime_stats": regime_stats, "validation": validation, "validation_summary": summary,
            "effective_method": effective.method, "checks": checks, "ineligible": ineligible,
            "reason": summary["reason"], "solver": solver}


def latest_decision(prices: pd.DataFrame, config: AllocationConfig, train_days: int,
                    current_weights: pd.Series | None = None, transaction_cost_bps: float = 0.0) -> dict:
    # The extra decision row lets shifted features include the last observed close.
    next_session = prices.index[-1] + pd.offsets.BDay(1)
    extended = pd.concat([prices, pd.DataFrame([prices.iloc[-1]], index=[next_session])])
    # concat drops `.attrs` unless every input carries identical ones - and
    # the eligibility attrs (data.clean_gappy_columns) are what keep a
    # placeholder-backed listing out of the live proposal's candidate set.
    extended.attrs = dict(prices.attrs)
    context = build_context(extended, config.forecast_horizon_days, config.method == "ML mean-variance")
    result = make_decision(context, len(prices), config, train_days, current_weights, transaction_cost_bps)
    result["as_of"] = prices.index[-1]
    return result
