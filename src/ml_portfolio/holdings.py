from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from .storage import load_artifact, write_artifact

INSTRUMENT_TYPES = ("Stock", "Mutual fund")
PURCHASE_COLUMNS = ["symbol", "instrument_type", "purchase_date", "quantity", "purchase_price"]


def parse_calendar_dates(values, as_of) -> pd.DatetimeIndex:
    """Parse and validate a column of calendar dates: reject numeric timestamps,
    timezone-aware values, unparsable values, and dates after `as_of`. Shared by
    the open-lot purchase book and the transaction ledger so there is one date
    contract, not two."""
    try:
        if any(isinstance(value, (int, float, np.number)) for value in values):
            raise ValueError("Numeric timestamps are not calendar dates.")
        dates = pd.DatetimeIndex([pd.Timestamp(value) for value in values])
        if dates.tz is not None:
            raise ValueError("Dates must be calendar dates without a timezone.")
        dates = dates.normalize()
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("Enter a valid calendar date for every row.") from exc
    valuation_date = pd.Timestamp(as_of).normalize()
    if dates.isna().any() or (dates > valuation_date).any():
        raise ValueError("Dates must be on or before the latest valuation date.")
    return dates


def value_purchases(purchases: pd.DataFrame, latest_prices: pd.Series, as_of) -> dict:
    if not set(PURCHASE_COLUMNS).issubset(purchases.columns):
        raise ValueError("Purchases need symbol, instrument type, purchase date, units, and purchase price.")
    lots = purchases[PURCHASE_COLUMNS].dropna(how="all").copy()
    if lots.empty:
        raise ValueError("Add at least one purchase.")
    lots["symbol"] = lots.symbol.astype("string").str.strip().str.upper()
    if lots.symbol.isna().any() or not lots.symbol.isin(latest_prices.index).all():
        raise ValueError("Every purchase must match a symbol in the loaded price or NAV data.")
    if not lots.instrument_type.isin(INSTRUMENT_TYPES).all():
        raise ValueError("Choose Stock or Mutual fund for every purchase.")
    if (lots.groupby("symbol").instrument_type.nunique() > 1).any():
        raise ValueError("A symbol cannot have more than one instrument type.")
    for column in ("quantity", "purchase_price"):
        lots[column] = pd.to_numeric(lots[column], errors="coerce")
        if not np.isfinite(lots[column]).all() or (lots[column] <= 0).any():
            raise ValueError("Purchase units and purchase prices must be positive, finite numbers.")
    stock_units = lots.loc[lots.instrument_type == "Stock", "quantity"]
    if not np.equal(stock_units, np.floor(stock_units)).all():
        raise ValueError("Stock purchases require whole shares; mutual funds allow fractional units.")
    lots["purchase_date"] = parse_calendar_dates(lots.purchase_date, as_of)
    valuation_date = pd.Timestamp(as_of).normalize()
    latest = pd.to_numeric(latest_prices.reindex(lots.symbol), errors="coerce").to_numpy()
    if not np.isfinite(latest).all() or (latest <= 0).any():
        raise ValueError("Purchased assets require a positive latest price or NAV.")
    lots["latest_price"] = latest
    lots["invested_cost"] = lots.quantity * lots.purchase_price
    lots["current_value"] = lots.quantity * lots.latest_price
    if not np.isfinite(lots[["invested_cost", "current_value"]]).all().all():
        raise ValueError("Purchase amounts are too large.")
    lots["unrealized_gain"] = lots.current_value - lots.invested_cost
    lots["return_pct"] = lots.unrealized_gain / lots.invested_cost
    lots["holding_days"] = (valuation_date - lots.purchase_date).dt.days
    lots = lots.sort_values(["purchase_date", "symbol"], kind="stable").reset_index(drop=True)
    positions = lots.groupby("symbol", sort=False).agg(
        instrument_type=("instrument_type", "first"), quantity=("quantity", "sum"),
        invested_cost=("invested_cost", "sum"), current_value=("current_value", "sum"),
        latest_price=("latest_price", "first"), first_purchase=("purchase_date", "min"),
        last_purchase=("purchase_date", "max"), lots=("symbol", "size"))
    positions["average_purchase_price"] = positions.invested_cost / positions.quantity
    positions["unrealized_gain"] = positions.current_value - positions.invested_cost
    positions["return_pct"] = positions.unrealized_gain / positions.invested_cost
    cost, value = float(positions.invested_cost.sum()), float(positions.current_value.sum())
    if not np.isfinite([cost, value]).all():
        raise ValueError("Total purchase amounts are too large.")
    return {"lots": lots, "positions": positions, "invested_cost": cost, "current_value": value,
            "unrealized_gain": value - cost, "return_pct": value / cost - 1, "as_of": valuation_date}


def save_purchase_portfolio(root: Path, name: str, purchases: pd.DataFrame, cash: float,
                            latest_prices: pd.Series, as_of) -> Path:
    name = name.strip()
    if not name:
        raise ValueError("Enter a portfolio name.")
    if not np.isfinite(cash) or cash < 0:
        raise ValueError("Cash must be finite and non-negative.")
    book = value_purchases(purchases, latest_prices, as_of)
    key = "portfolio_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:12]
    return write_artifact(root, key, {"schema_version": 2, "kind": "purchase_portfolio", "name": name,
                                    "saved_at": datetime.now(timezone.utc), "cash": cash, "book": book})


def load_purchase_portfolio(path: Path) -> dict:
    payload = load_artifact(path)
    if payload.get("schema_version") != 2 or payload.get("kind") != "purchase_portfolio":
        raise ValueError("Unsupported purchase portfolio.")
    return payload


def purchase_portfolios(root: Path) -> dict[Path, str]:
    choices = {}
    for path in sorted(root.glob("portfolio_*.json"), reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            choices[path] = f"{payload['name']} | {payload['saved_at']['$timestamp']}"
        except (ValueError, KeyError, TypeError, OSError):
            choices[path] = f"Unreadable portfolio: {path.stem}"
    return choices
