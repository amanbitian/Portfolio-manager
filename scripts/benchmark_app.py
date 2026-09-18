"""Reproducible application benchmarks; synthetic data, no services required.

Covers context/feature build, ridge fitting, strategy comparisons (serial,
shared, parallel), purchase valuation and persistence, exports, a two-session
concurrency proxy, and headless UI-rendering cost - see README.md and
todo.md (O01) for what each phase measures and its known limits.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import multiprocessing
import platform
import sys
import tempfile
import threading
import tracemalloc
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.append(str(ROOT / ".app-deps"))

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ml_portfolio.backtest import BacktestConfig, run_strategy_comparison, run_walk_forward_backtest
from ml_portfolio.data import make_synthetic_prices
from ml_portfolio.decisions import _predict, _training, build_context
from ml_portfolio.features import feature_matrix_at
from ml_portfolio.holdings import load_purchase_portfolio, save_purchase_portfolio, value_purchases
from ml_portfolio.models import ridge_forecast
from ml_portfolio.optimizers import METHODS, AllocationConfig
from ml_portfolio.storage import price_key

try:
    import psutil
except ImportError:
    psutil = None

DEPENDENCIES = ["pandas", "numpy", "streamlit", "scipy", "cvxpy", "pyarrow", "threadpoolctl"]


def measure_repeated(timings: dict, name: str, function, repeats: int, setup=None):
    """Run `function` `repeats` times, reporting median/p95 wall time.

    `setup()`, when given, runs before every repeat (e.g. to give a "cold"
    phase a fresh cache directory each time); its return value is ignored.
    Returns the last call's return value so the pipeline can keep going.
    """
    samples = []
    value = None
    for _ in range(max(1, repeats)):
        if setup is not None:
            setup()
        start = perf_counter()
        value = function()
        samples.append(perf_counter() - start)
    arr = np.array(samples)
    timings[name] = {
        "median_seconds": round(float(np.median(arr)), 4),
        "p95_seconds": round(float(np.percentile(arr, 95)), 4),
        "samples": [round(s, 4) for s in samples],
    }
    return value


def dependency_versions() -> dict:
    versions = {}
    for package in DEPENDENCIES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def hardware_info() -> dict:
    info = {"platform": platform.platform(), "processor": platform.processor(),
            "cpu_count": __import__("os").cpu_count()}
    if psutil is not None:
        info["total_memory_mb"] = round(psutil.virtual_memory().total / 1024**2, 1)
    return info


def solver_diagnostics(rebalances: pd.DataFrame) -> dict:
    if rebalances.empty or "solver_backend" not in rebalances:
        return {}
    fallbacks = rebalances["solver_fallback_reason"].notna().sum() if "solver_fallback_reason" in rebalances else 0
    return {
        "backend_counts": rebalances["solver_backend"].value_counts(dropna=False).to_dict(),
        "risk_model_counts": rebalances["solver_risk_model"].value_counts(dropna=False).to_dict(),
        "fallback_count": int(fallbacks),
        "mean_iterations": round(float(rebalances["solver_iterations"].dropna().mean()), 2)
        if rebalances["solver_iterations"].notna().any() else None,
        "mean_build_seconds": round(float(rebalances["solver_build_seconds"].dropna().mean()), 5)
        if "solver_build_seconds" in rebalances and rebalances["solver_build_seconds"].notna().any() else None,
        "mean_solve_seconds": round(float(rebalances["solver_solve_seconds"].dropna().mean()), 5)
        if "solver_solve_seconds" in rebalances and rebalances["solver_solve_seconds"].notna().any() else None,
    }


def _run_comparison_worker(prices: pd.DataFrame, config: AllocationConfig, backtest: BacktestConfig,
                           root: str, max_workers: int, out: dict, key: str) -> None:
    start = perf_counter()
    run_strategy_comparison(prices, config, backtest, max_workers, Path(root))
    out[key] = perf_counter() - start


def two_thread_proxy(prices, config, backtest, root, max_workers) -> dict:
    """Two comparisons launched as threads in this process. `_comparison_lock`
    in backtest.py serializes them intentionally - this measures the resulting
    stall, the realistic number for a single-process Streamlit deployment."""
    def worker():
        run_strategy_comparison(prices, config, backtest, max_workers, root)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    wall_start = perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall_seconds = perf_counter() - wall_start
    return {"wall_seconds": round(wall_seconds, 4), "note": "serialized by _comparison_lock (expected)"}


def two_process_proxy(prices, config, backtest, root, max_workers) -> dict:
    """Two comparisons in separate OS processes sharing a cache directory -
    the best-case parallelism number if this app were ever deployed
    multi-process, distinct from the in-process serialization above."""
    manager = multiprocessing.Manager()
    out = manager.dict()
    processes = [
        multiprocessing.Process(target=_run_comparison_worker,
                                args=(prices, config, backtest, str(root), max_workers, out, key))
        for key in ("a", "b")
    ]
    wall_start = perf_counter()
    try:
        for p in processes:
            p.start()
        for p in processes:
            p.join(timeout=300)
        wall_seconds = perf_counter() - wall_start
        if any(p.exitcode != 0 for p in processes):
            return {"error": "one or more worker processes failed", "wall_seconds": round(wall_seconds, 4)}
        return {"wall_seconds": round(wall_seconds, 4), "per_process_seconds": dict(out)}
    except Exception as exc:  # pragma: no cover - platform-dependent multiprocessing issues
        return {"error": str(exc)}


def purchase_round_trip(prices: pd.DataFrame, root: Path) -> dict:
    symbols = list(prices.columns[: min(20, prices.shape[1])])
    purchases = pd.DataFrame({
        "symbol": symbols, "instrument_type": "Stock", "purchase_date": prices.index[0],
        "quantity": 10, "purchase_price": prices.iloc[0][symbols].to_numpy(),
    })
    latest_prices, as_of = prices.iloc[-1], prices.index[-1]
    timings: dict = {}
    measure_repeated(timings, "value_purchases_seconds",
                     lambda: value_purchases(purchases, latest_prices, as_of), 3)
    path = measure_repeated(timings, "save_purchase_portfolio_seconds",
                            lambda: save_purchase_portfolio(root, f"bench-{uuid4().hex[:8]}", purchases,
                                                            1000.0, latest_prices, as_of), 3)
    measure_repeated(timings, "load_purchase_portfolio_seconds", lambda: load_purchase_portfolio(path), 3)
    return timings


def ui_rendering_phase(repeats: int) -> dict:
    """Headless server-side render cost via streamlit.testing.v1.AppTest - no
    browser, so real DOM paint/JS/network time is out of scope (that would
    need browser automation); this covers script-execution/widget-tree cost,
    the dominant server-side share of "UI rendering" time."""
    try:
        from streamlit.testing.v1 import AppTest
    except ImportError:
        return {"error": "streamlit.testing.v1.AppTest unavailable in this Streamlit version"}
    timings: dict = {}
    workspace_names = ["Portfolio", "Research", "Research Agent", "Strategy Copilot", "Saved Decisions"]

    def render(name):
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
        at.run()
        if name != workspace_names[0]:
            at.sidebar.radio[0].set_value(name).run()
        if at.exception:
            raise RuntimeError(f"AppTest exception on {name}: {[str(e) for e in at.exception]}")

    for name in workspace_names:
        measure_repeated(timings, name, lambda n=name: render(n), repeats)
    return timings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stocks", type=int, default=1000)
    parser.add_argument("--rows", type=int, default=650)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--ui-repeats", type=int, default=1,
                        help="AppTest reruns per workspace (slower; default keeps the run quick).")
    parser.add_argument("--skip-two-process", action="store_true",
                        help="Skip the multiprocessing concurrency proxy (useful on constrained CI).")
    args = parser.parse_args()

    tracemalloc.start()
    symbols = [f"S{i:04d}" for i in range(args.stocks)]
    prices = make_synthetic_prices(symbols=symbols, periods=args.rows, seed=args.seed,
                                   end_date="2026-01-02")
    timings = {
        "stocks": args.stocks, "rows": args.rows, "seed": args.seed, "repeats": args.repeats,
        "dataset_version": price_key(prices),
        "dependencies": dependency_versions(),
        "hardware": hardware_info(),
    }

    config = AllocationConfig(max_weight=.02, max_volatility=.2)
    backtest = BacktestConfig(train_days=504)

    with tempfile.TemporaryDirectory() as temp, threadpool_limits(limits=1):
        root = Path(temp)

        # Cold-context timing needs a genuinely empty cache directory every repeat.
        cold_dirs = []

        def fresh_cold_dir():
            cold_dirs.append(tempfile.mkdtemp(dir=temp))

        measure_repeated(timings, "context_cold_seconds",
                         lambda: build_context(prices, 21, cache_dir=Path(cold_dirs[-1])),
                         args.repeats, setup=fresh_cold_dir)

        context = build_context(prices, 21, cache_dir=root)
        measure_repeated(timings, "context_warm_seconds", lambda: build_context(prices, 21, cache_dir=root),
                         args.repeats)
        origins = list(range(350, 500, 21))

        def original_fits():
            for origin in origins:
                x, y = _training(context, origin, 504)
                ridge_forecast(x, y, feature_matrix_at(context.features, prices.index[origin]), list(prices.columns))

        measure_repeated(timings, "ridge_original_seconds", original_fits, args.repeats)
        measure_repeated(timings, "ridge_statistics_seconds",
                         lambda: [_predict(context, origin, 504) for origin in origins], args.repeats)

        import ml_portfolio.convex  # noqa: F401  (import the convex runtime before timing it)
        independent = measure_repeated(timings, "independent_serial_seconds", lambda: {
            method: run_walk_forward_backtest(prices, replace(config, method=method), backtest, cache_dir=root)
            for method in METHODS}, args.repeats)
        shared = measure_repeated(timings, "shared_serial_seconds",
                                  lambda: run_strategy_comparison(prices, config, backtest, 1, root), args.repeats)
        parallel = measure_repeated(timings, "shared_parallel_seconds",
                                    lambda: run_strategy_comparison(prices, config, backtest, 4, root), args.repeats)
        for method in METHODS:
            np.testing.assert_allclose(parallel[method]["equity_curve"], independent[method]["equity_curve"], atol=1e-9)
            np.testing.assert_allclose(parallel[method]["weights"], shared[method]["weights"], atol=1e-9)
        timings["results_match"] = True
        timings["cache_megabytes"] = round(sum(p.stat().st_size for p in root.glob("*/*.parquet")) / 1024**2, 2)

        timings["solver_diagnostics"] = {
            method: solver_diagnostics(independent[method]["rebalances"]) for method in METHODS
        }

        timings["purchase_workflow"] = purchase_round_trip(prices, root)
        measure_repeated(timings, "export_csv_seconds",
                         lambda: independent[METHODS[0]]["rebalances"].to_csv(index=False), args.repeats)

        timings["concurrency"] = {"two_threads": two_thread_proxy(prices, config, backtest, root, 4)}
        if not args.skip_two_process:
            timings["concurrency"]["two_processes"] = two_process_proxy(prices, config, backtest, root, 1)

    timings["ui_rendering"] = ui_rendering_phase(args.ui_repeats)

    python_current, python_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    timings["memory"] = {"python_peak_mb": round(python_peak / 1024**2, 2)}
    if psutil is not None:
        timings["memory"]["process_peak_mb"] = round(psutil.Process().memory_info().rss / 1024**2, 2)
    else:
        timings["memory"]["process_peak_mb"] = None
        timings["memory"]["note"] = "install psutil for OS-level RSS; python_peak_mb only tracks Python allocations"

    print(json.dumps(timings, indent=2, default=str))


if __name__ == "__main__":
    main()
