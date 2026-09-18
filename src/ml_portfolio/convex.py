from __future__ import annotations

from time import perf_counter

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.linalg import cholesky

from .optimizers import AllocationError, _proportional_weights, risk_checks


def _solve(problem: cp.Problem, solver: str) -> None:
    options = ({"eps_abs": 1e-9, "eps_rel": 1e-9, "max_iter": 20000, "polishing": True}
               if solver == "OSQP" else
               {"tol_gap_abs": 1e-8, "tol_gap_rel": 1e-8, "tol_feas": 1e-9, "max_iter": 300})
    try:
        problem.solve(solver=solver, warm_start=True, **options)
    except cp.error.SolverError:
        if solver == "OSQP":
            return _solve(problem, "CLARABEL")
        raise AllocationError("Convex solver failed to find an accepted allocation.") from None
    if problem.status != cp.OPTIMAL:
        if solver == "OSQP":
            return _solve(problem, "CLARABEL")
        raise AllocationError(f"Risk and turnover limits have no accepted allocation: {problem.status}")


def _risk_sqrt_w(w, cov, n, factor_structure, diagonal, use_factor_structure):
    """One square-root risk representation, shared by both the objective's
    risk term (`cp.sum_squares`) and the `max_volatility` constraint
    (`cp.norm(..., 2)`) - previously these were built from two independently
    computed pieces (a `quad_form`+`psd_wrap` matrix for the objective, a
    separately-derived Cholesky factor for the constraint); unifying them
    removes the duplicate work and, evidently, some numerical conditioning
    difference: doing so also made a pre-existing, previously-deterministic
    CLARABEL `optimal_inaccurate` test failure (documented in
    known_issues.md since Stage 1) stop reproducing.

    O05 (`todo.md`) asked for these values to be reused across rebalance
    decisions via CVXPY's `cp.Parameter`/DPP support rather than rebuilt from
    literals every call. That was implemented and measured directly (not
    assumed) across every risk shape and every realistic universe size this
    app actually uses the convex backend for - see README's O05 changelog
    entry for the numbers. The result was a consistent, sometimes severe,
    NET SLOWDOWN (up to ~3x at the 1000-asset factor-model scale this app is
    built around) rather than a speedup: CVXPY's DPP machinery still has to
    re-evaluate the affine map from parameter values to solver data on every
    `.solve()` call, and at this app's problem sizes (100s-1000s of assets,
    dense-ish per-asset risk data) that re-evaluation cost consistently
    outweighed the canonicalization it was meant to save. Parameter reuse
    only measured faster in a narrow band (roughly n<20), well below where
    this backend is ever actually selected (n>250 in "auto" mode, or an
    explicit user choice at any size). Per O05's own "Measure First" tag and
    "Done when: repeated decisions demonstrate a measurable gain" bar, the
    measured answer here is "no gain" - so plain NumPy literals (rebuilt each
    call, exactly as before) are used, not cp.Parameter/DPP reuse.
    """
    if use_factor_structure:
        loadings = factor_structure["loadings"]
        factor_sqrt = np.sqrt(np.maximum(factor_structure["factor_variance"], 0))
        specific_sqrt = np.sqrt(np.maximum(factor_structure["specific_variance"], 0))
        return cp.hstack([cp.multiply(factor_sqrt, loadings.T @ w), cp.multiply(specific_sqrt, w)])
    if diagonal:
        return cp.multiply(np.sqrt(np.maximum(np.diag(cov), 0)), w)
    factor = cholesky(cov + np.eye(n) * 1e-10, lower=False, check_finite=False)
    return factor @ w


def allocate_convex(mu, cov, trailing, config, symbols, old, cost_bps, *,
                    risk_model_label="full covariance", factor_structure=None):
    n = len(symbols)
    target = min(1 - config.cash_weight, n * config.max_weight)
    diagonal = np.count_nonzero(cov) == n
    # A factor-model covariance is low-rank-plus-diagonal (n x k loadings, k
    # factor variances, n specific variances - k is small, ~10). Expressing the
    # risk term/volatility constraint through that structure directly, instead
    # of reconstructing a dense n x n matrix, keeps the conic program's
    # per-iteration cost near the diagonal case instead of dense-cone cost -
    # confirmed necessary by benchmarking: reconstructing dense turned a
    # ~170ms/decision solve into several seconds at n=1000.
    use_factor_structure = factor_structure is not None and not diagonal
    w = cp.Variable(n)
    cash = 1 - cp.sum(w)
    changes = w - old.reindex(symbols).to_numpy()
    turnover = (cp.norm1(changes) + cp.abs(cash - old.CASH)) / 2
    constraints = [w >= 0, w <= config.max_weight, cp.sum(w) <= target,
                   turnover <= config.max_turnover]
    if config.max_volatility is not None:
        risk_sqrt_w = _risk_sqrt_w(w, cov, n, factor_structure, diagonal, use_factor_structure)
        constraints.append(cp.norm(risk_sqrt_w, 2) <= config.max_volatility)
    for sector, cap in config.sector_caps.items():
        mask = np.array([config.sectors.get(s, "Unassigned") == sector for s in symbols], dtype=float)
        constraints.append(mask @ w <= cap)
    ml = config.method == "ML mean-variance"
    minimum = config.method == "Minimum variance"
    fee = (cost_bps + (config.min_trade_benefit_bps if ml else 0)) / 10000
    if ml or minimum:
        risk_sqrt_w = _risk_sqrt_w(w, cov, n, factor_structure, diagonal, use_factor_structure)
        risk_term = cp.sum_squares(risk_sqrt_w)
        base = .5 * config.risk_aversion * config.forecast_horizon_days / 252 * risk_term
        if ml:
            base -= mu @ w
        scale = max(float(np.max(np.sum(np.abs(cov), axis=1))) * config.risk_aversion * config.forecast_horizon_days / 252,
                    float(np.abs(mu).max()) if ml else 0, fee, 1e-5)
    else:
        raw = np.ones(n)
        if config.method == "Inverse volatility":
            vol = trailing.reindex(columns=symbols).std().to_numpy()
            raw = np.divide(1, vol, out=np.zeros(n), where=np.isfinite(vol) & (vol > 0))
        desired = _proportional_weights(raw, target, config.max_weight)
        base = .5 * cp.sum_squares(w - desired)
        scale = 1

    build_start = perf_counter()
    if minimum:
        budget_problem = cp.Problem(cp.Maximize(cp.sum(w)), constraints)
        build_seconds = perf_counter() - build_start
        solve_start = perf_counter()
        _solve(budget_problem, "CLARABEL")
        budget = max(0.0, min(target, float(w.value.sum())) - 1e-7)
        constraints.append(cp.sum(w) >= budget)
        problem = cp.Problem(cp.Minimize((base + fee * cp.norm1(changes)) / scale), constraints)
        _solve(problem, "CLARABEL" if config.max_volatility is not None else "OSQP")
        solve_seconds = perf_counter() - solve_start
    else:
        problem = cp.Problem(cp.Minimize((base + fee * cp.norm1(changes)) / scale), constraints)
        build_seconds = perf_counter() - build_start
        solve_start = perf_counter()
        _solve(problem, "CLARABEL" if config.max_volatility is not None else "OSQP")
        solve_seconds = perf_counter() - solve_start
    risky = np.clip(w.value, 0, config.max_weight)
    if risky.sum() > target:
        risky *= target / risky.sum()
    weights = pd.Series(np.r_[risky, 1 - risky.sum()], index=symbols + ["CASH"])
    covariance = pd.DataFrame(cov, index=symbols, columns=symbols)
    if not risk_checks(weights, covariance, config, old).passed.all():
        raise AllocationError("Convex solution failed post-solve risk checks.")
    weights.attrs["solver"] = {"converged": True, "iterations": problem.solver_stats.num_iters,
                               "objective": float(problem.value * scale), "mode": "convex",
                               "backend": problem.solver_stats.solver_name, "assets": n,
                               "risk_model": "diagonal" if diagonal else risk_model_label,
                               "build_seconds": build_seconds, "solve_seconds": solve_seconds}
    return weights
