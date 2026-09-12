from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

METHODS = ("ML mean-variance", "Equal weight", "Inverse volatility", "Minimum variance")


@dataclass(frozen=True)
class AllocationConfig:
    method: str = "ML mean-variance"
    max_weight: float = 0.25
    cash_weight: float = 0.05
    risk_aversion: float = 6.0
    forecast_horizon_days: int = 21
    max_volatility: float | None = None
    max_turnover: float = 1.0
    min_trade_benefit_bps: float = 0.0
    sectors: dict[str, str] = field(default_factory=dict)
    sector_caps: dict[str, float] = field(default_factory=dict)


class AllocationError(ValueError):
    pass


def _clean_covariance(covariance: pd.DataFrame, symbols: list[str]) -> np.ndarray:
    cov = covariance.reindex(index=symbols, columns=symbols).to_numpy(dtype=float)
    if not np.isfinite(cov).all():
        raise AllocationError("Covariance must be finite and cover every asset.")
    cov = (cov + cov.T) / 2
    values, vectors = np.linalg.eigh(cov)
    if values.min() < -1e-7:
        raise AllocationError("Covariance must be positive semidefinite.")
    return (vectors * np.maximum(values, 1e-10)) @ vectors.T


def validate_config(config: AllocationConfig) -> None:
    if config.method not in METHODS:
        raise AllocationError(f"Unknown strategy: {config.method}")
    for name in ("max_weight", "cash_weight", "max_turnover"):
        value = getattr(config, name)
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise AllocationError(f"{name} must be between zero and one.")
    if not np.isfinite(config.risk_aversion) or config.risk_aversion <= 0:
        raise AllocationError("Risk aversion must be positive.")
    if config.forecast_horizon_days < 1:
        raise AllocationError("Forecast horizon must be positive.")
    if not np.isfinite(config.min_trade_benefit_bps) or config.min_trade_benefit_bps < 0:
        raise AllocationError("Minimum trade benefit must be non-negative.")
    if config.max_volatility is not None and (
        not np.isfinite(config.max_volatility) or config.max_volatility <= 0
    ):
        raise AllocationError("Volatility limit must be positive.")
    if any(not np.isfinite(v) or not 0 <= v <= 1 for v in config.sector_caps.values()):
        raise AllocationError("Sector limits must be between zero and one.")


def normalize_holdings(current: pd.Series | None, symbols: list[str]) -> pd.Series:
    index = symbols + ["CASH"]
    if current is None:
        return pd.Series([0.0] * len(symbols) + [1.0], index=index)
    if not current.index.is_unique or set(current.index) - set(index):
        raise AllocationError("Holdings contain duplicate or unknown assets.")
    weights = current.reindex(index).fillna(0.0).astype(float)
    if not np.isfinite(weights).all() or (weights < 0).any() or abs(weights.sum() - 1) > 1e-7:
        raise AllocationError("Current weights must be non-negative and sum to one.")
    return weights


def risk_checks(weights: pd.Series, covariance: pd.DataFrame, config: AllocationConfig,
                current: pd.Series | None = None) -> pd.DataFrame:
    symbols = list(covariance.index)
    cov = _clean_covariance(covariance, symbols)
    w = weights.reindex(symbols).fillna(0.0).to_numpy()
    rows = []

    def add(name, value, limit, passed):
        rows.append({"check": name, "value": float(value), "limit": float(limit), "passed": bool(passed)})

    add("Weight total", weights.sum(), 1, abs(weights.sum() - 1) <= 1e-6)
    add("Long only", weights.min(), 0, weights.min() >= -1e-7)
    add("Largest position", w.max(), config.max_weight, w.max() <= config.max_weight + 1e-6)
    cash = float(weights.get("CASH", 0))
    add("Cash reserve", cash, config.cash_weight, cash >= config.cash_weight - 1e-6)
    vol = float(np.sqrt(max(0, w @ cov @ w)))
    if config.max_volatility is not None:
        add("Annual volatility", vol, config.max_volatility, vol <= config.max_volatility + 1e-6)
    if current is not None:
        old = normalize_holdings(current, symbols)
        turnover = float((weights.reindex(old.index).fillna(0) - old).abs().sum() / 2)
        add("One-way turnover", turnover, config.max_turnover, turnover <= config.max_turnover + 1e-6)
    for sector, cap in config.sector_caps.items():
        exposure = sum(w[i] for i, s in enumerate(symbols) if config.sectors.get(s, "Unassigned") == sector)
        add(f"Sector: {sector}", exposure, cap, exposure <= cap + 1e-6)
    return pd.DataFrame(rows)


def _proportional_weights(raw: np.ndarray, total: float, cap: float) -> np.ndarray:
    weights = np.zeros(len(raw))
    active = np.ones(len(raw), dtype=bool)
    for _ in range(len(raw) + 1):
        if not active.any():
            break
        values = raw[active]
        values = values if values.sum() > 0 else np.ones(len(values))
        proposed = (total - weights.sum()) * values / values.sum()
        indices = np.flatnonzero(active)
        capped = proposed > cap
        if not capped.any():
            weights[indices] = proposed
            break
        weights[indices[capped]] = cap
        active[indices[capped]] = False
    return weights


def allocate_portfolio(expected_returns: pd.Series, covariance: pd.DataFrame,
                       trailing_returns: pd.DataFrame, config: AllocationConfig,
                       current_weights: pd.Series | None = None,
                       transaction_cost_bps: float = 0.0) -> pd.Series:
    validate_config(config)
    symbols = list(expected_returns.index)
    if not symbols or not expected_returns.index.is_unique or "CASH" in symbols:
        raise AllocationError("Allocation requires unique risky asset symbols.")
    if not np.isfinite(transaction_cost_bps) or not 0 <= transaction_cost_bps < 10000:
        raise AllocationError("Trading cost must be between zero and 10,000 bps.")
    mu = expected_returns.to_numpy(dtype=float)
    if not np.isfinite(mu).all():
        raise AllocationError("Expected returns must be finite.")
    n = len(symbols)
    old = normalize_holdings(current_weights, symbols)
    cov = _clean_covariance(covariance, symbols)
    target = min(1 - config.cash_weight, n * config.max_weight)
    raw = np.ones(n)
    if config.method == "Inverse volatility":
        vol = trailing_returns.reindex(columns=symbols).std().to_numpy()
        raw = np.divide(1, vol, out=np.zeros(n), where=np.isfinite(vol) & (vol > 0))
    desired = _proportional_weights(raw, target, config.max_weight)

    # Auxiliary absolute changes cover risky positions and residual cash.
    mapping = np.vstack([np.eye(n), -np.ones(n)])
    offset = np.r_[np.zeros(n), 1.0] - old.to_numpy()
    constraints = [
        {"type": "ineq", "fun": lambda x: target - x[:n].sum(),
         "jac": lambda x: np.r_[-np.ones(n), np.zeros(n + 1)]},
        {"type": "ineq", "fun": lambda x: x[n:] - (mapping @ x[:n] + offset),
         "jac": lambda x: np.hstack([-mapping, np.eye(n + 1)])},
        {"type": "ineq", "fun": lambda x: x[n:] + (mapping @ x[:n] + offset),
         "jac": lambda x: np.hstack([mapping, np.eye(n + 1)])},
        {"type": "ineq", "fun": lambda x: 2 * config.max_turnover - x[n:].sum(),
         "jac": lambda x: np.r_[np.zeros(n), -np.ones(n + 1)]},
    ]
    if config.max_volatility is not None:
        variance_limit = config.max_volatility ** 2
        constraints.append({"type": "ineq", "fun": lambda x: 1 - x[:n] @ cov @ x[:n] / variance_limit,
                            "jac": lambda x: np.r_[-2 * cov @ x[:n] / variance_limit, np.zeros(n + 1)]})
    for sector, cap in config.sector_caps.items():
        mask = np.array([config.sectors.get(s, "Unassigned") == sector for s in symbols], dtype=float)
        constraints.append({"type": "ineq", "fun": lambda x, m=mask, c=cap: c - m @ x[:n],
                            "jac": lambda x, m=mask: np.r_[-m, np.zeros(n + 1)]})
    horizon_cov = cov * config.forecast_horizon_days / 252
    fee = transaction_cost_bps / 10000
    penalty = fee + config.min_trade_benefit_bps / 10000
    ml = config.method == "ML mean-variance"
    minimum_variance = config.method == "Minimum variance"

    def objective(x):
        w = x[:n]
        base = (.5 * config.risk_aversion * w @ horizon_cov @ w - (mu @ w if ml else 0)) if (ml or minimum_variance) else .5 * np.sum((w - desired) ** 2)
        return base + (penalty if ml else fee) * x[n:n+n].sum()

    def gradient(x):
        grad = config.risk_aversion * horizon_cov @ x[:n] - (mu if ml else 0) if (ml or minimum_variance) else x[:n] - desired
        return np.r_[grad, np.full(n, penalty if ml else fee), 0.0]

    initial = np.minimum(old.iloc[:n].to_numpy(), config.max_weight)
    x0 = np.r_[initial, np.abs(mapping @ initial + offset)]
    if minimum_variance:
        # Keep the largest feasible risky budget, then minimize variance within it.
        budget_result = minimize(lambda x: -x[:n].sum(), x0,
                                 jac=lambda x: np.r_[-np.ones(n), np.zeros(n + 1)], method="SLSQP",
                                 bounds=[(0, config.max_weight)] * n + [(0, 2)] * (n + 1),
                                 constraints=constraints, options={"ftol": 1e-11, "maxiter": 500})
        # Reaching the proven upper bound certifies this linear budget objective.
        full_budget_feasible = (budget_result.x[:n].sum() >= target - 1e-7
                                and all(np.min(c["fun"](budget_result.x)) >= -1e-7 for c in constraints))
        if not budget_result.success and not full_budget_feasible:
            raise AllocationError(f"Risk and turnover limits are incompatible: {budget_result.message}")
        # Budget slack avoids a singular feasible set tangent to the risk ceiling.
        budget = max(0.0, float(budget_result.x[:n].sum()) - 1e-7)
        constraints.append({"type": "ineq", "fun": lambda x: x[:n].sum() - budget,
                            "jac": lambda x: np.r_[np.ones(n), np.zeros(n + 1)]})
        x0 = budget_result.x
        x0[n:] = np.abs(mapping @ x0[:n] + offset)
    objective_scale = max(float(np.linalg.eigvalsh(horizon_cov).max()) * config.risk_aversion,
                          float(np.abs(mu).max()), penalty, 1e-5) if (ml or minimum_variance) else 1.0
    result = minimize(lambda x: objective(x) / objective_scale, x0,
                      jac=lambda x: gradient(x) / objective_scale, method="SLSQP",
                      bounds=[(0, config.max_weight)] * n + [(0, 2)] * (n + 1),
                      constraints=constraints, options={"ftol": 1e-10, "maxiter": 500})
    risky = np.clip(result.x[:n], 0, config.max_weight)
    if risky.sum() > target:
        risky *= target / risky.sum()
    weights = pd.Series(np.r_[risky, max(0, 1 - risky.sum())], index=symbols + ["CASH"])
    checks = risk_checks(weights, covariance, config, old)
    if not result.success or not checks["passed"].all():
        raise AllocationError("No accepted allocation for these limits. Check turnover against required risk reductions. "
                              f"Solver: {result.message}")
    weights.attrs["solver"] = {"converged": True, "iterations": int(result.nit), "objective": float(result.fun)}
    return weights
