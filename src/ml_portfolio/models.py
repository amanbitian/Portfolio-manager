from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ForecastResult:
    expected_returns: pd.Series
    covariance: pd.DataFrame
    confidence: float
    regime: str
    diagnostics: dict[str, float]


def ridge_forecast(
    train_features: pd.DataFrame,
    train_target: pd.Series,
    current_features: pd.DataFrame,
    symbols: list[str],
    alpha: float = 8.0,
) -> pd.Series:
    joined = train_features.join(train_target, how="inner").dropna()
    if len(joined) < max(40, len(train_features.columns) * 4):
        return pd.Series(0.0, index=symbols, name="expected_return")

    x = joined[train_features.columns].to_numpy(dtype=float)
    y = joined[train_target.name].to_numpy(dtype=float)
    x = np.nan_to_num(x)
    y = np.nan_to_num(y)

    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    x_std[x_std == 0] = 1.0
    y_mean = y.mean()

    x_scaled = (x - x_mean) / x_std
    gram = x_scaled.T @ x_scaled
    penalty = alpha * np.eye(gram.shape[0])
    beta = np.linalg.pinv(gram + penalty) @ x_scaled.T @ (y - y_mean)

    aligned = current_features.reindex(symbols).fillna(0.0)
    current_x = aligned[train_features.columns].to_numpy(dtype=float)
    current_x = np.nan_to_num((current_x - x_mean) / x_std)

    predictions = y_mean + current_x @ beta
    return pd.Series(predictions, index=symbols, name="expected_return")


def shrinkage_covariance(
    returns: pd.DataFrame,
    symbols: list[str],
    shrinkage: float = 0.25,
) -> pd.DataFrame:
    window_returns = returns.reindex(columns=symbols).dropna(how="all").fillna(0.0)
    if len(window_returns) < 20:
        identity = np.eye(len(symbols)) * 0.04
        return pd.DataFrame(identity, index=symbols, columns=symbols)

    sample = window_returns.cov().to_numpy() * 252
    diag = np.diag(np.diag(sample))
    cov = (1 - shrinkage) * sample + shrinkage * diag
    cov += np.eye(len(symbols)) * 1e-6
    return pd.DataFrame(cov, index=symbols, columns=symbols)


def detect_regime(returns: pd.DataFrame) -> tuple[str, dict[str, float]]:
    market = returns.mean(axis=1).dropna()
    if len(market) < 63:
        return "insufficient-history", {"market_return_63d": 0.0, "market_vol_63d": 0.0}

    recent = market.tail(63)
    market_return = float((1 + recent).prod() - 1)
    market_vol = float(recent.std() * np.sqrt(252))

    if market_return < -0.08 and market_vol > 0.22:
        regime = "risk-off"
    elif market_return > 0.08 and market_vol < 0.24:
        regime = "risk-on"
    elif market_vol > 0.28:
        regime = "high-volatility"
    else:
        regime = "neutral"

    return regime, {"market_return_63d": market_return, "market_vol_63d": market_vol}


def forecast_confidence(rmse: float, baseline_rmse: float, eligible: bool) -> float:
    """Bounded error improvement on matched, matured validation horizons."""
    if not eligible or not np.isfinite([rmse, baseline_rmse]).all() or baseline_rmse <= 1e-12:
        return 0.0
    return float(np.clip(1 - rmse / baseline_rmse, 0, 1))
