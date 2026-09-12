from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .features import build_feature_panel, daily_returns, feature_matrix_at, forward_returns
from .models import detect_regime, forecast_confidence, ridge_forecast, shrinkage_covariance
from .optimizers import AllocationConfig, allocate_portfolio, risk_checks


@dataclass
class DecisionContext:
    prices: pd.DataFrame
    returns: pd.DataFrame
    features: pd.DataFrame
    targets: pd.Series
    horizon: int


def build_context(prices: pd.DataFrame, horizon: int, ml: bool = True) -> DecisionContext:
    return DecisionContext(prices, daily_returns(prices),
                           build_feature_panel(prices) if ml else pd.DataFrame(),
                           forward_returns(prices, horizon) if ml else pd.Series(dtype=float), horizon)


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
        x, y = _training(context, origin, train_days)
        joined = x.join(y).dropna()
        if joined.empty or joined.index.get_level_values("date").nunique() < 40:
            continue
        date = context.prices.index[origin]
        prediction = ridge_forecast(x, y, feature_matrix_at(context.features, date), symbols)
        actual = context.targets.xs(date, level="date").reindex(symbols)
        frames.append(pd.DataFrame({"date": date, "symbol": symbols, "prediction": prediction.to_numpy(),
                                    "actual": actual.to_numpy(), "baseline": float(joined[y.name].mean())}))
    records = pd.concat(frames, ignore_index=True).dropna() if frames else pd.DataFrame()
    return validation_summary(records), records


def make_decision(context: DecisionContext, idx: int, config: AllocationConfig,
                  train_days: int, current_weights: pd.Series | None = None,
                  transaction_cost_bps: float = 0.0) -> dict:
    if idx < train_days or idx >= len(context.prices):
        raise ValueError("Decision date needs the full training window and a price reference.")
    symbols = list(context.prices.columns)
    trailing = context.returns.iloc[max(0, idx - train_days):idx]
    covariance = shrinkage_covariance(trailing.tail(252), symbols)
    regime, regime_stats = detect_regime(trailing.tail(126))
    expected = pd.Series(0.0, index=symbols, name="expected_return")
    summary = {"eligible": False, "confidence": 0.0, "blocks": 0, "reason": "Baseline selected"}
    validation = pd.DataFrame()
    effective = replace(config, forecast_horizon_days=context.horizon)
    if config.method == "ML mean-variance":
        summary, validation = validate_forecast(context, idx, train_days)
        x, y = _training(context, idx, train_days)
        expected = ridge_forecast(x, y, feature_matrix_at(context.features, context.prices.index[idx]), symbols)
        if not summary["eligible"]:
            effective = replace(effective, method="Equal weight")
    adjusted = expected * summary["confidence"] if config.method == "ML mean-variance" else expected
    weights = allocate_portfolio(adjusted, covariance, trailing.tail(126), effective,
                                 current_weights, transaction_cost_bps)
    checks = risk_checks(weights, covariance, effective, current_weights)
    return {"date": context.prices.index[idx], "information_through": context.prices.index[idx - 1],
            "weights": weights, "expected_returns": expected, "adjusted_returns": adjusted,
            "covariance": covariance, "confidence": summary["confidence"], "regime": regime,
            "regime_stats": regime_stats, "validation": validation, "validation_summary": summary,
            "effective_method": effective.method, "checks": checks,
            "reason": summary["reason"], "solver": weights.attrs.get("solver", {})}


def latest_decision(prices: pd.DataFrame, config: AllocationConfig, train_days: int,
                    current_weights: pd.Series | None = None, transaction_cost_bps: float = 0.0) -> dict:
    # The extra decision row lets shifted features include the last observed close.
    next_session = prices.index[-1] + pd.offsets.BDay(1)
    extended = pd.concat([prices, pd.DataFrame([prices.iloc[-1]], index=[next_session])])
    context = build_context(extended, config.forecast_horizon_days, config.method == "ML mean-variance")
    result = make_decision(context, len(prices), config, train_days, current_weights, transaction_cost_bps)
    result["as_of"] = prices.index[-1]
    return result
