from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from .decisions import latest_decision
from .execution import rebalance_cost_rate
from .optimizers import AllocationConfig, risk_checks


def build_proposal(prices: pd.DataFrame, quantities: pd.Series, cash: float,
                   config: AllocationConfig, train_days: int, cost_bps: float) -> dict:
    if not quantities.index.is_unique or set(quantities.index) - set(prices.columns):
        raise ValueError("Holdings contain unknown or duplicate symbols.")
    quantities = quantities.reindex(prices.columns).fillna(0).astype(float)
    if not np.isfinite(quantities).all() or (quantities < 0).any() or not np.equal(quantities, np.floor(quantities)).all():
        raise ValueError("Holdings must be non-negative whole-share quantities.")
    if not np.isfinite(cash) or cash < 0:
        raise ValueError("Available cash must be non-negative.")
    last = prices.iloc[-1].astype(float)
    if not np.isfinite(last).all() or (last <= 0).any():
        raise ValueError("Proposal valuation needs positive, finite latest prices.")
    values = quantities * last
    capital = float(values.sum() + cash)
    if capital <= 0:
        raise ValueError("Enter cash or existing holdings before calculating a proposal.")
    current = values / capital
    current.loc["CASH"] = cash / capital
    decision = latest_decision(prices, config, train_days, current, cost_bps)
    weights = decision["weights"]
    fee_rate = rebalance_cost_rate(weights, current, cost_bps)
    target_quantity = np.floor(weights.reindex(last.index).clip(lower=0) * capital * (1 - fee_rate) / last).astype(int)
    delta = target_quantity - quantities
    notional = delta * last
    fees = notional.abs() * cost_bps / 10000
    remaining_cash = float(cash - notional.sum() - fees.sum())
    after_value = target_quantity * last
    after_capital = float(after_value.sum() + remaining_cash)
    if after_capital <= 0:
        raise ValueError("Fees would exhaust the portfolio.")
    rounded = after_value / after_capital
    rounded.loc["CASH"] = remaining_cash / after_capital
    checks = risk_checks(rounded, decision["covariance"], config, current)
    checks = pd.concat([checks, pd.DataFrame([{"check": "Cash funding", "value": remaining_cash,
                                             "limit": 0.0, "passed": remaining_cash >= -1e-7}])], ignore_index=True)
    reasons = []
    for symbol in last.index:
        if delta[symbol] == 0:
            reasons.append("No whole-share change")
        elif decision["effective_method"] == "ML mean-variance":
            reasons.append("Validated forecast; costs and portfolio limits applied")
        else:
            reasons.append(f"{decision['effective_method']} target; portfolio limits applied")
    trades = pd.DataFrame({"symbol": last.index, "reference_price": last.to_numpy(),
                           "current_quantity": quantities.to_numpy().astype(int),
                           "target_quantity": target_quantity.to_numpy(), "trade_quantity": delta.to_numpy().astype(int),
                           "action": np.where(delta > 0, "Buy", np.where(delta < 0, "Sell", "Hold")),
                           "trade_value": notional.to_numpy(), "estimated_fee": fees.to_numpy(),
                           "current_weight": current.reindex(last.index).to_numpy(),
                           "target_weight": weights.reindex(last.index).to_numpy(),
                           "rounded_weight": rounded.reindex(last.index).to_numpy(), "reason": reasons})
    return {"decision": decision, "trades": trades, "checks": checks, "accepted": bool(checks.passed.all()),
            "capital": capital, "estimated_fees": float(fees.sum()), "cash_after": remaining_cash,
            "capital_after": after_capital, "current_weights": current, "rounded_weights": rounded}


def _json_value(value):
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, pd.DataFrame):
        return _json_value(value.to_dict("records"))
    if isinstance(value, pd.Series):
        return _json_value(value.to_dict())
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_proposal(root: Path, proposal: dict, prices: pd.DataFrame,
                  config: AllocationConfig, train_days: int, cost_bps: float) -> Path:
    if not proposal["accepted"]:
        raise ValueError("A proposal with failed risk checks cannot be saved as accepted.")
    root.mkdir(parents=True, exist_ok=True)
    key = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:12]
    payload = {"schema_version": 1, "id": key, "saved_at": datetime.now(timezone.utc),
               "allocation_config": asdict(config), "train_days": train_days, "cost_bps": cost_bps,
               "price_hash": hashlib.sha256(pd.util.hash_pandas_object(prices, index=True).values.tobytes()
                                             + repr(list(prices.columns)).encode()).hexdigest(),
               "prices": prices.rename_axis("date").reset_index(), "proposal": proposal}
    path = root / f"{key}.json"
    with path.open("x", encoding="utf-8") as handle:
        json.dump(_json_value(payload), handle, allow_nan=False, indent=2)
    return path


def saved_proposals(root: Path, latest_prices: pd.Series, as_of: pd.Timestamp,
                    price_history: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    for path in sorted(root.glob("*.json"), reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            proposal = payload["proposal"]
            decision = proposal["decision"]
            trades = pd.DataFrame(proposal["trades"])
            symbols = trades.symbol.tolist()
            comparable = set(symbols).issubset(latest_prices.index) and as_of >= pd.Timestamp(decision["as_of"])
            if comparable and price_history is not None:
                original_date = pd.Timestamp(decision["as_of"])
                comparable = original_date in price_history.index
                if comparable:
                    comparable = bool(np.allclose(price_history.loc[original_date, symbols].to_numpy(),
                                                   trades.reference_price.to_numpy(), rtol=1e-8, atol=1e-8))
            value = float((trades.target_quantity.to_numpy() * latest_prices.reindex(symbols).to_numpy()).sum()
                          + proposal["cash_after"]) if comparable else np.nan
            rows.append({"id": payload["id"], "saved_at": payload["saved_at"], "as_of": decision["as_of"],
                         "method": decision["effective_method"], "initial_capital": proposal["capital"],
                         "estimated_fees": proposal["estimated_fees"], "marked_value": value,
                         "hold_return_after_fees": value / proposal["capital"] - 1 if comparable else np.nan,
                         "status": "Hypothetical hold" if comparable else "Not comparable"})
        except (ValueError, KeyError, TypeError, OSError):
            rows.append({"id": path.stem, "status": "Unreadable snapshot"})
    return pd.DataFrame(rows)
