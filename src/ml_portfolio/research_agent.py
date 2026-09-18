from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from .data import listed_only
from .storage import atomic_parquet, load_artifact, price_key, write_artifact

LOOKBACKS = {"1M": 21, "3M": 63, "6M": 126, "1Y": 252}


def _check_prices(prices: pd.DataFrame) -> None:
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("Research requires a DatetimeIndex.")
    if prices.empty or prices.shape[1] < 2:
        raise ValueError("Research requires at least two assets.")
    if not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError("Research requires unique, increasing dates.")
    if not prices.columns.is_unique:
        raise ValueError("Research requires unique symbols.")
    values = prices.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Research prices must be finite and positive.")


def _score_higher(value: float, low: float, high: float) -> float:
    if not np.isfinite(value):
        return 50.0
    return float(np.clip((value - low) / (high - low), 0, 1) * 100)


def _score_lower(value: float, low: float, high: float) -> float:
    return 100.0 - _score_higher(value, low, high)


def _max_drawdown(series: pd.Series) -> float:
    return float((series / series.cummax() - 1).min())


def _trailing_return(series: pd.Series, days: int) -> float:
    """Return over the trailing `days` window, or NaN if there isn't enough
    history to measure it. Previously silently substituted the full-history
    return instead - a report row labeled "1Y return" could show, say, a
    100-day return under that label with no indication the window was
    actually shorter, misrepresenting how much risk/return history backs it.
    NaN cells (edge placeholders masked by data.listed_only) don't count as
    history either: a name listed 100 sessions ago has no 1Y return, even
    when the loaded frame is years long.
    """
    series = series.dropna()
    if len(series) <= days:
        return float("nan")
    return float(series.iloc[-1] / series.iloc[-days - 1] - 1)


def build_price_research_report(
    prices: pd.DataFrame,
    symbol: str,
    peers: list[str] | None = None,
    question: str = "",
) -> dict:
    """Create a deterministic price-led research brief for one asset.

    This is intentionally not an LLM research report. It creates structured facts,
    scores, and a draft thesis from the loaded price/NAV matrix so a future local
    research agent can consume clean evidence rather than raw tables.
    """
    _check_prices(prices)
    symbol = str(symbol).strip().upper()
    if symbol not in prices.columns:
        raise ValueError(f"Unknown symbol: {symbol}")
    peers = [str(peer).strip().upper() for peer in (peers or []) if str(peer).strip()]
    peers = [peer for peer in dict.fromkeys(peers) if peer != symbol]
    unknown = sorted(set(peers) - set(prices.columns))
    if unknown:
        raise ValueError(f"Unknown peer symbols: {', '.join(unknown)}")

    # Only a symbol's real prints: the loader back-fills a mid-window listing
    # flat before its first real date (and forward-fills after a cutoff), and
    # every statistic below - trailing returns, volatility, drawdown, SMAs,
    # the 1Y-return scenario distribution - would otherwise be measured on a
    # fabricated flat line and look reassuringly calm.
    prices = listed_only(prices)
    returns = prices.pct_change().dropna(how="all")
    asset = prices[symbol].dropna()
    if asset.empty:
        raise ValueError(f"No real price history for {symbol} in the loaded window.")
    asset_returns = returns[symbol].dropna()
    universe_return = returns.mean(axis=1).reindex(asset_returns.index)
    latest = float(asset.iloc[-1])
    cumulative = float(asset.iloc[-1] / asset.iloc[0] - 1)
    annual_vol = float(asset_returns.std(ddof=1) * np.sqrt(252)) if len(asset_returns) > 1 else float("nan")
    sharpe = float(asset_returns.mean() * 252 / annual_vol) if annual_vol > 0 else float("nan")
    drawdown = _max_drawdown(asset)
    beta = float(asset_returns.cov(universe_return) / universe_return.var()) if universe_return.var() > 0 else float("nan")

    trailing = {name: _trailing_return(asset, days) for name, days in LOOKBACKS.items()}
    trailing_frame = pd.DataFrame({name: prices.apply(_trailing_return, days=days) for name, days in LOOKBACKS.items()})
    ranks = trailing_frame.rank(ascending=False, pct=True)
    momentum_rank = float(1 - ranks.loc[symbol, "6M"]) if "6M" in ranks else 0.5

    sma_50 = float(asset.rolling(50).mean().iloc[-1]) if len(asset) >= 50 else float("nan")
    sma_200 = float(asset.rolling(200).mean().iloc[-1]) if len(asset) >= 200 else float("nan")
    trend_state = "above both 50-day and 200-day trend lines"
    if np.isfinite(sma_50) and latest < sma_50:
        trend_state = "below its 50-day trend line"
    if np.isfinite(sma_200) and latest < sma_200:
        trend_state = "below its 200-day trend line"

    peer_symbols = peers or [col for col in prices.columns if col != symbol][: min(5, prices.shape[1] - 1)]
    peer_rows = []
    for peer in [symbol] + peer_symbols:
        row = {
            "symbol": peer,
            "latest_price": float(prices[peer].dropna().iloc[-1]),
            "return_3m": _trailing_return(prices[peer], 63),
            "return_6m": _trailing_return(prices[peer], 126),
            "return_1y": _trailing_return(prices[peer], 252),
            "annualized_volatility": float(returns[peer].std(ddof=1) * np.sqrt(252)),
            "max_drawdown": _max_drawdown(prices[peer].dropna()),
        }
        peer_rows.append(row)
    peer_frame = pd.DataFrame(peer_rows)

    peer_median_6m = float(peer_frame.loc[peer_frame.symbol != symbol, "return_6m"].median()) if len(peer_frame) > 1 else 0.0
    relative_6m = trailing["6M"] - peer_median_6m
    scores = pd.DataFrame(
        [
            {"dimension": "Momentum", "score": _score_higher(trailing["6M"], -0.15, 0.30)},
            {"dimension": "Trend", "score": _score_higher((latest / sma_200 - 1) if np.isfinite(sma_200) else trailing["3M"], -0.10, 0.20)},
            {"dimension": "Risk stability", "score": 0.55 * _score_lower(annual_vol, 0.10, 0.45) + 0.45 * _score_higher(drawdown, -0.45, -0.08)},
            {"dimension": "Peer relative", "score": _score_higher(relative_6m, -0.15, 0.15)},
        ]
    )
    overall = float(np.average(scores.score, weights=[0.35, 0.25, 0.25, 0.15]))
    scores = pd.concat([pd.DataFrame([{"dimension": "Overall", "score": overall}]), scores], ignore_index=True)

    best_peer = peer_frame.sort_values("return_6m", ascending=False).iloc[0].symbol
    risk_note = "risk looks contained versus recent history" if annual_vol < 0.25 and drawdown > -0.25 else "risk deserves attention because volatility or drawdown is elevated"
    thesis = (
        f"{symbol} scores {overall:.0f}/100 on this price-led research pass. "
        f"It is {trend_state}, has returned {trailing['6M']:.1%} over roughly six months, "
        f"and its 6M return is {relative_6m:.1%} versus the selected peer median. "
        f"{best_peer} currently leads the comparison on 6M momentum. The main caveat is that "
        f"{risk_note}. This is a deterministic market-data brief, not a full valuation or filings view."
    )
    if question.strip():
        thesis += f"\n\nQuestion focus: {question.strip()}"

    summary = pd.DataFrame(
        [
            {"metric": "Latest price", "value": latest},
            {"metric": "Full-period return", "value": cumulative},
            {"metric": "1M return", "value": trailing["1M"]},
            {"metric": "3M return", "value": trailing["3M"]},
            {"metric": "6M return", "value": trailing["6M"]},
            {"metric": "1Y return", "value": trailing["1Y"]},
            {"metric": "Annualized volatility", "value": annual_vol},
            {"metric": "Sharpe, zero rf", "value": sharpe},
            {"metric": "Max drawdown", "value": drawdown},
            {"metric": "Beta to equal-weight universe", "value": beta},
        ]
    )
    evidence = [
        "All metrics use the loaded price/NAV matrix only.",
        "Trailing returns and trend signals are measured as of the latest loaded date.",
        "Peer relative score compares six-month return against the selected peer median.",
        "No fundamentals, filings, news, dividends, taxes, or distribution data are included yet.",
    ]
    hypothesis = {
        "title": f"{symbol} price momentum versus selected peers",
        "strategy_prompt": (
            f"Test a momentum strategy that ranks the loaded universe by 6-month price strength, "
            f"holds the top names including candidates like {symbol}, rebalances monthly, and includes costs."
        ),
    }
    formulas = {
        "momentum": {"metric": "6M trailing return", "low": -0.15, "high": 0.30},
        "trend": {"metric": "price vs 200-day SMA (falls back to 3M return if SMA unavailable)",
                 "low": -0.10, "high": 0.20},
        "risk_stability": {"volatility_weight": 0.55, "volatility_low": 0.10, "volatility_high": 0.45,
                           "drawdown_weight": 0.45, "drawdown_low": -0.45, "drawdown_high": -0.08},
        "peer_relative": {"metric": "6M return minus peer median 6M return", "low": -0.15, "high": 0.15},
        "overall_weights": {"Momentum": 0.35, "Trend": 0.25, "Risk stability": 0.25, "Peer relative": 0.15},
    }
    rolling_1y = asset.pct_change(252).dropna()
    if len(rolling_1y) >= 20:
        scenario_range = {
            "bear_10th_percentile": float(rolling_1y.quantile(0.10)),
            "base_median": float(rolling_1y.quantile(0.50)),
            "bull_90th_percentile": float(rolling_1y.quantile(0.90)),
            "sample_size": int(len(rolling_1y)),
            "note": "Historical distribution of trailing 1-year returns for this symbol over the "
                   "loaded price history - a historical range, not a forecast.",
        }
    else:
        scenario_range = {"sample_size": int(len(rolling_1y)),
                          "note": "Not enough price history for a historical scenario range."}
    return {
        "symbol": symbol,
        "as_of": prices.index[-1],
        "summary": summary,
        "scores": scores,
        "peers": peer_frame,
        "thesis": thesis,
        "evidence": evidence,
        "hypothesis": hypothesis,
        "formulas": formulas,
        "scenario_range": scenario_range,
    }


def save_research_report(root: Path, name: str, prices, report: dict) -> Path:
    """Persist a Research Agent brief, following the same artifact-envelope
    convention as save_strategy_run/save_proposal/save_research (schema_version=2)."""
    name = name.strip()
    if not name:
        raise ValueError("Enter a name for this research report.")
    root.mkdir(parents=True, exist_ok=True)
    key = "research_report_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:12]
    digest = price_key(prices)
    price_path = root / "prices" / f"{digest}.parquet"
    if not price_path.exists():
        atomic_parquet(prices, price_path)
    payload = {"schema_version": 2, "kind": "research_report", "id": key, "name": name,
              "saved_at": datetime.now(timezone.utc), "price_hash": digest,
              "prices": {"$table": f"prices/{digest}.parquet"}, "report": report}
    return write_artifact(root, key, payload)


def load_research_report(path: Path) -> dict:
    payload = load_artifact(path)
    if payload.get("kind") != "research_report":
        raise ValueError("Unsupported research report.")
    return payload


def research_reports(root: Path) -> dict[Path, str]:
    choices = {}
    for path in sorted(root.glob("research_report_*.json"), reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            choices[path] = f"{payload['name']} | {payload['saved_at']['$timestamp']}"
        except (ValueError, KeyError, TypeError, OSError):
            choices[path] = f"Unreadable research report: {path.stem}"
    return choices
