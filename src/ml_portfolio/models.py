from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FULL_COVARIANCE_ASSET_LIMIT = 400


@dataclass(frozen=True)
class ForecastResult:
    expected_returns: pd.Series
    covariance: pd.DataFrame
    confidence: float
    regime: str
    diagnostics: dict[str, float]


def ridge_daily_statistics(features: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    joined = features.join(target, how="inner").dropna()
    rows, dates = [], []
    for date, group in joined.groupby(level="date", sort=True):
        x = group[features.columns].to_numpy(dtype=float)
        y = group[target.name].to_numpy(dtype=float)
        rows.append(np.r_[len(y), 1, x.sum(axis=0), y.sum(), (x.T @ x).ravel(), x.T @ y])
        dates.append(date)
    width = 3 + 2 * len(features.columns) + len(features.columns) ** 2
    return pd.DataFrame(rows, index=pd.DatetimeIndex(dates, name="date"),
                        columns=[f"s{i}" for i in range(width)])


class RidgeWindow:
    """Prefix sums keep each matured-window fit independent of its row count."""

    def __init__(self, statistics: pd.DataFrame, columns):
        self.dates = statistics.index
        self.columns = list(columns)
        values = statistics.to_numpy(dtype=float)
        self.prefix = np.vstack([np.zeros(values.shape[1]), values.cumsum(axis=0)])

    def predict(self, first, last, current, symbols, alpha=8.0):
        lo = self.dates.searchsorted(first)
        hi = self.dates.searchsorted(last, side="right")
        stats = self.prefix[hi] - self.prefix[lo]
        count, days = stats[:2]
        p = len(self.columns)
        zero = pd.Series(0.0, index=symbols, name="expected_return")
        baseline = stats[2+p] / count if count else 0.0
        if count < max(40, p * 4):
            return zero, float(baseline), int(days)
        mean = stats[2:2+p] / count
        xx = stats[3+p:3+p+p*p].reshape(p, p)
        xy = stats[3+p+p*p:]
        centered = xx - count * np.outer(mean, mean)
        std = np.sqrt(np.maximum(np.diag(centered) / count, 0))
        std[std < 1e-12] = 1
        gram = centered / np.outer(std, std)
        rhs = (xy - count * mean * baseline) / std
        beta = np.linalg.solve(gram + alpha * np.eye(p), rhs)
        x = current.reindex(symbols)[self.columns].fillna(0).to_numpy(dtype=float)
        prediction = baseline + ((x - mean) / std) @ beta
        return pd.Series(prediction, index=symbols, name="expected_return"), float(baseline), int(days)


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


def _factor_covariance(window_returns: pd.DataFrame, symbols: list[str], n_factors: int = 10) -> pd.DataFrame:
    """Statistical (PCA) factor-plus-specific-risk model, used above
    `full_covariance_asset_limit` where the sample covariance would be a
    classic more-assets-than-observations singular/noisy estimate.

    PCA factors are uncorrelated by construction, so the factor covariance is
    always diagonal and never needs its own sample-covariance fit - the
    reconstruction `loadings @ diag(factor_variance) @ loadings.T +
    diag(specific_variance)` is positive semidefinite by construction (sum of
    a PSD low-rank term and a PSD diagonal term), so it needs no eigenvalue
    repair beyond the same conditioning ridge the full-covariance path below
    already applies.
    """
    values = window_returns.to_numpy(dtype=float)
    t, n = values.shape
    k = max(1, min(n_factors, t - 1, n))
    centered = values - values.mean(axis=0, keepdims=True)
    # Economy SVD: cheap here since t (~252 trading days) is far smaller than n.
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    loadings = vt[:k].T                                       # n x k
    factor_variance = (s[:k] ** 2) / max(t - 1, 1) * 252       # annualized; diagonal factor covariance
    reconstructed = (loadings * factor_variance) @ loadings.T
    factor_returns = u[:, :k] * s[:k]                          # T x k
    residual = centered - factor_returns @ loadings.T
    specific_variance = np.nan_to_num(residual.var(axis=0, ddof=1) * 252, nan=0.04, posinf=0.04, neginf=0.04)
    # Fold the conditioning ridge directly into specific_variance so the dense
    # reconstruction below and the low-rank structure exposed via .attrs (for
    # the solver to use directly, without ever forming a dense n x n matrix -
    # see convex.py) stay exactly consistent with each other.
    specific_variance = np.maximum(specific_variance, 1e-6) + 1e-6
    cov = reconstructed + np.diag(specific_variance)
    cov = (cov + cov.T) / 2
    frame = pd.DataFrame(cov, index=symbols, columns=symbols)
    frame.attrs["approximation"] = "factor"
    frame.attrs["n_factors"] = int(k)
    frame.attrs["loadings"] = loadings
    frame.attrs["factor_variance"] = factor_variance
    frame.attrs["specific_variance"] = specific_variance
    return frame


def shrinkage_covariance(
    returns: pd.DataFrame,
    symbols: list[str],
    shrinkage: float = 0.25,
    full_covariance_asset_limit: int = FULL_COVARIANCE_ASSET_LIMIT,
) -> pd.DataFrame:
    window_returns = returns.reindex(columns=symbols).dropna(how="all").fillna(0.0)
    n = len(symbols)
    if len(window_returns) < 20:
        identity = np.eye(len(symbols)) * 0.04
        return pd.DataFrame(identity, index=symbols, columns=symbols)
    if n > full_covariance_asset_limit:
        frame = _factor_covariance(window_returns, symbols)
        frame.attrs["asset_limit"] = full_covariance_asset_limit
        return frame

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
