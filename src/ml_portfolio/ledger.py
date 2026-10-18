from __future__ import annotations

import hashlib
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .holdings import INSTRUMENT_TYPES, parse_calendar_dates
from .storage import load_artifact, write_artifact

TRANSACTION_COLUMNS = ["transaction_id", "action", "symbol", "instrument_type",
                       "date", "quantity", "price", "amount", "notes"]
TRANSACTION_ACTIONS = ("buy", "sell", "deposit", "withdrawal", "fee", "distribution")
_SECURITY_ACTIONS = ("buy", "sell")
_CASH_ONLY_ACTIONS = ("deposit", "withdrawal", "fee", "distribution")
SCHEMA_VERSION = 1  # transaction_ledger's own version namespace - independent of
                    # purchase_portfolio's schema_version=2 and the strategy DSL's
                    # schema_version=1 (three separate namespaces, same convention
                    # already used elsewhere in this codebase).


def transaction_id(action, symbol, date, quantity, price, amount, notes="") -> str:
    """Content-derived id: identical inputs always produce the same id, which is
    what makes duplicate-import detection possible without a separate index.

    `notes` is normalized with `pd.isna` (not `notes or ""`): a genuinely empty
    notes field is Python `None`/`""` when entered directly in the UI, but
    becomes float `NaN` after a CSV export/import round trip - and `NaN` is
    *truthy* in Python (`nan or "" == nan`), so the old `notes or ""` check
    hashed a real `"nan"` string in that case, giving the same transaction a
    different id purely because of how it arrived and breaking dedup on
    reimport.
    """
    payload = repr((str(action), None if symbol is None else str(symbol),
                    str(pd.Timestamp(date).normalize()),
                    None if quantity is None or pd.isna(quantity) else float(quantity),
                    None if price is None or pd.isna(price) else float(price),
                    None if amount is None or pd.isna(amount) else float(amount),
                    "" if pd.isna(notes) else str(notes)))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _validate_transactions(transactions: pd.DataFrame, as_of) -> pd.DataFrame:
    if not set(["action", "date"]).issubset(transactions.columns):
        raise ValueError("Transactions need at least an action and a date.")
    rows = transactions.reindex(columns=TRANSACTION_COLUMNS).dropna(how="all").copy()
    if rows.empty:
        raise ValueError("Add at least one transaction.")
    if not rows.action.isin(TRANSACTION_ACTIONS).all():
        raise ValueError(f"Action must be one of {TRANSACTION_ACTIONS}.")
    rows["date"] = parse_calendar_dates(rows.date, as_of)

    security = rows.action.isin(_SECURITY_ACTIONS)
    cash_only = rows.action.isin(_CASH_ONLY_ACTIONS)

    sec_rows = rows[security]
    if security.any():
        symbols = sec_rows.symbol.astype("string").str.strip().str.upper()
        if symbols.isna().any():
            raise ValueError("Buy/sell transactions need a symbol.")
        rows.loc[security, "symbol"] = symbols
        if not sec_rows.instrument_type.isin(INSTRUMENT_TYPES).all():
            raise ValueError("Choose Stock or Mutual fund for every buy/sell.")
        if (rows.loc[security].groupby("symbol").instrument_type.nunique() > 1).any():
            raise ValueError("A symbol cannot have more than one instrument type.")
        for column in ("quantity", "price"):
            values = pd.to_numeric(sec_rows[column], errors="coerce")
            if not np.isfinite(values).all() or (values <= 0).any():
                raise ValueError("Buy/sell transactions need positive, finite quantity and price.")
            rows.loc[security, column] = values.to_numpy()
        stock_mask = security & (rows.instrument_type == "Stock")
        stock_qty = rows.loc[stock_mask, "quantity"]
        if not np.equal(stock_qty, np.floor(stock_qty)).all():
            raise ValueError("Stock buy/sell transactions require whole shares.")

    if cash_only.any():
        amounts = pd.to_numeric(rows.loc[cash_only, "amount"], errors="coerce")
        if not np.isfinite(amounts).all() or (amounts <= 0).any():
            raise ValueError("Deposits, withdrawals, fees, and distributions need a positive amount.")
        rows.loc[cash_only, "amount"] = amounts.to_numpy()

    # `TRANSACTION_COLUMNS` starts with "transaction_id", so `reindex` above
    # already preserved one if the incoming data supplied it (e.g. a
    # broker-provided reference id) - only compute a fresh id where one is
    # genuinely missing, instead of unconditionally overwriting and discarding
    # that external identity.
    computed = [
        transaction_id(row.action, row.symbol, row.date, row.quantity, row.price, row.amount, row.notes)
        for row in rows.itertuples()
    ]
    has_incoming_id = rows["transaction_id"].notna() & (rows["transaction_id"].astype(str).str.strip() != "")
    rows["transaction_id"] = np.where(has_incoming_id, rows["transaction_id"].astype(str), computed)
    sorted_rows = rows.sort_values(["date", "action"], kind="stable").reset_index(drop=True)
    _check_chronological_cash(sorted_rows)
    return sorted_rows


def _check_chronological_cash(rows: pd.DataFrame) -> None:
    """Reject a transaction history that goes cash-negative at ANY point in
    time, not just in its final balance. `portfolio.build_proposal` already
    rejects negative cash for a brand-new proposal (portfolio.py) - a saved
    ledger that was never actually affordable (e.g. a buy recorded before its
    funding deposit) shouldn't be treated as valid either, since nothing else
    in this codebase models borrowing/margin. `rows` must already be sorted
    chronologically (as `_validate_transactions` always passes it here).

    Only applies once a `deposit` is present: a buy-only history (no deposits
    at all) is a long-standing, intentionally supported convention - matching
    the older open-lot `value_purchases` mode - where cash origin is never
    tracked and `cash_balance` is simply negative-by-construction (total cost
    basis), not a real funding constraint. A deposit signals the user IS
    tracking real cash, and that's when going negative becomes a genuine
    inconsistency worth rejecting.
    """
    if not (rows.action == "deposit").any():
        return
    deltas = pd.Series(0.0, index=rows.index)
    deposit, withdrawal = rows.action == "deposit", rows.action == "withdrawal"
    fee, distribution = rows.action == "fee", rows.action == "distribution"
    buy, sell = rows.action == "buy", rows.action == "sell"
    deltas[deposit] = rows.loc[deposit, "amount"]
    deltas[withdrawal] = -rows.loc[withdrawal, "amount"]
    deltas[fee] = -rows.loc[fee, "amount"]
    deltas[distribution] = rows.loc[distribution, "amount"]
    deltas[buy] = -(rows.loc[buy, "quantity"] * rows.loc[buy, "price"])
    deltas[sell] = rows.loc[sell, "quantity"] * rows.loc[sell, "price"]
    running = deltas.cumsum()
    negative = running < -1e-6
    if negative.any():
        row = rows.loc[negative.idxmax()]
        target = f" of {row.symbol}" if pd.notna(row.symbol) else ""
        raise ValueError(f"Cash goes negative on {row.date.date()} ({row.action}{target}, "
                         f"running balance {running[negative].iloc[0]:,.2f}); "
                         "record deposits before spending them - this app doesn't model borrowing.")


def append_transactions(existing: pd.DataFrame | None, new: pd.DataFrame, as_of) -> tuple[pd.DataFrame, list[str]]:
    """Validate `new`, assign stable ids, and append rows whose id doesn't
    already exist in `existing` - identical content always produces the same
    id, so re-importing the same rows is a no-op instead of a duplicate.

    Also dedupes `new` against *itself* first: a duplicated row within the
    same incoming batch (e.g. an accidentally double-pasted CSV row) must not
    both be appended - the original code only ever compared incoming rows
    against `existing`, so two identical rows arriving together both went in.
    """
    validated = _validate_transactions(new, as_of)
    within_batch_duplicates = validated.loc[validated["transaction_id"].duplicated(), "transaction_id"].tolist()
    validated = validated.drop_duplicates(subset="transaction_id", keep="first")
    if existing is None or existing.empty:
        return validated.reset_index(drop=True), within_batch_duplicates
    existing_ids = set(existing["transaction_id"])
    duplicate_ids = within_batch_duplicates + [tid for tid in validated["transaction_id"] if tid in existing_ids]
    fresh = validated[~validated["transaction_id"].isin(existing_ids)]
    combined = pd.concat([existing, fresh], ignore_index=True).sort_values(
        ["date", "action"], kind="stable").reset_index(drop=True)
    return combined, duplicate_ids


def apply_fifo(transactions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match sells against buys oldest-first per symbol. Returns (open_lots, realized).
    Raises ValueError if a sell exceeds the open quantity available for its symbol."""
    open_lots_rows, realized_rows = [], []
    for symbol, group in transactions[transactions.action.isin(_SECURITY_ACTIONS)].groupby("symbol"):
        instrument_type = group.instrument_type.iloc[0]
        queue = deque()
        for row in group.sort_values(["date", "action"], kind="stable").itertuples():
            if row.action == "buy":
                queue.append([row.date, float(row.quantity), float(row.price)])
            else:  # sell
                remaining = float(row.quantity)
                if remaining > sum(lot[1] for lot in queue) + 1e-9:
                    raise ValueError(f"Sell exceeds available quantity for {symbol}.")
                while remaining > 1e-12:
                    lot = queue[0]
                    matched = min(remaining, lot[1])
                    cost = matched * lot[2]
                    proceeds = matched * float(row.price)
                    realized_rows.append({
                        "symbol": symbol, "instrument_type": instrument_type,
                        "buy_date": lot[0], "sell_date": row.date, "quantity": matched,
                        "cost": cost, "proceeds": proceeds, "realized_gain": proceeds - cost,
                        "holding_days": (row.date - lot[0]).days,
                    })
                    lot[1] -= matched
                    remaining -= matched
                    if lot[1] <= 1e-12:
                        queue.popleft()
        for date, quantity, price in queue:
            if quantity > 1e-12:
                open_lots_rows.append({"symbol": symbol, "instrument_type": instrument_type,
                                       "purchase_date": date, "quantity": quantity, "purchase_price": price})

    open_lots = pd.DataFrame(open_lots_rows, columns=["symbol", "instrument_type", "purchase_date",
                                                      "quantity", "purchase_price"])
    realized = pd.DataFrame(realized_rows, columns=["symbol", "instrument_type", "buy_date", "sell_date",
                                                    "quantity", "cost", "proceeds", "realized_gain",
                                                    "holding_days"])
    return open_lots, realized


def cash_balance(transactions: pd.DataFrame) -> float:
    """deposits - withdrawals - buy cost + sell proceeds + distributions - fees."""
    buys = transactions[transactions.action == "buy"]
    sells = transactions[transactions.action == "sell"]
    flows = {
        "deposit": transactions.loc[transactions.action == "deposit", "amount"].sum(),
        "withdrawal": -transactions.loc[transactions.action == "withdrawal", "amount"].sum(),
        "fee": -transactions.loc[transactions.action == "fee", "amount"].sum(),
        "distribution": transactions.loc[transactions.action == "distribution", "amount"].sum(),
        "buy": -(buys.quantity * buys.price).sum(),
        "sell": (sells.quantity * sells.price).sum(),
    }
    return float(sum(flows.values()))


def _net_present_value(rate: float, flows: list[tuple[pd.Timestamp, float]]) -> float:
    origin = flows[0][0]
    return sum(cash / (1 + rate) ** ((date - origin).days / 365.0) for date, cash in flows)


def compute_xirr(cash_flows: list[tuple[pd.Timestamp, float]]) -> dict:
    """`cash_flows` should already include the terminal value as a final positive
    flow. Scans a coarse rate grid for sign changes in NPV(rate) rather than
    assuming a single well-behaved root exists, and reports both a missing root
    and a possible second root instead of silently returning one number."""
    flows = sorted(cash_flows, key=lambda item: item[0])
    if len({round(rate, 6) for _, rate in flows}) == 1 or all(cash >= 0 for _, cash in flows) or \
       all(cash <= 0 for _, cash in flows):
        return {"xirr": None, "converged": False, "multiple_roots_possible": False,
                "note": "All cash flows have the same sign; no rate of return is defined."}

    # Fine enough to separate roots ~0.1-0.2% apart (a real reproduced case:
    # two roots at 10.2%/10.4% previously fell inside one ~1%-wide cell with
    # matching-sign endpoints, so the bracket scan below saw no sign change at
    # all and reported "no root found" instead of finding either root).
    # `_net_present_value` is cheap - a few thousand extra grid points here is
    # still sub-second even for a large ledger, and this isn't a hot loop.
    grid = np.concatenate([np.linspace(-0.99, 1.0, 4000), np.linspace(1.0, 10.0, 1800)])
    values = [_net_present_value(rate, flows) for rate in grid]
    brackets = [(grid[i], grid[i + 1]) for i in range(len(grid) - 1)
               if np.sign(values[i]) != np.sign(values[i + 1]) and np.sign(values[i]) != 0]
    if not brackets:
        return {"xirr": None, "converged": False, "multiple_roots_possible": False,
                "note": "No root found for annual rates between -99% and 1000%."}
    root = brentq(_net_present_value, brackets[0][0], brackets[0][1], args=(flows,))
    return {"xirr": float(root), "converged": True, "multiple_roots_possible": len(brackets) > 1,
            "note": "Multiple sign changes found; XIRR may not be unique for this cash-flow pattern."
                   if len(brackets) > 1 else ""}


def modified_dietz_return(cash_flows: list[tuple[pd.Timestamp, float]], start_value: float,
                          end_value: float, start_date, end_date) -> float:
    """Cash-flow-weighted return approximation - NOT a true daily-linked
    time-weighted return (that would need a full valuation history at every
    cash-flow date). Weight of each flow = fraction of the period it was invested."""
    start_date, end_date = pd.Timestamp(start_date), pd.Timestamp(end_date)
    total_days = max((end_date - start_date).days, 1)
    net_flow = sum(cash for _, cash in cash_flows)
    weighted = sum(cash * (end_date - pd.Timestamp(date)).days / total_days for date, cash in cash_flows)
    denominator = start_value + weighted
    if denominator == 0:
        return float("nan")
    return (end_value - start_value - net_flow) / denominator


def value_ledger(transactions: pd.DataFrame, latest_prices: pd.Series, as_of) -> dict:
    if transactions.empty:
        raise ValueError("Add at least one transaction.")
    open_lots, realized = apply_fifo(transactions)
    valuation_date = pd.Timestamp(as_of).normalize()
    if not open_lots.empty:
        missing = set(open_lots.symbol) - set(latest_prices.index)
        if missing:
            raise ValueError(f"No price/NAV available for: {sorted(missing)}.")
        latest = pd.to_numeric(latest_prices.reindex(open_lots.symbol), errors="coerce").to_numpy()
        if not np.isfinite(latest).all() or (latest <= 0).any():
            raise ValueError("Held assets require a positive latest price or NAV.")
        open_lots["latest_price"] = latest
        open_lots["current_value"] = open_lots.quantity * open_lots.latest_price
        open_lots["invested_cost"] = open_lots.quantity * open_lots.purchase_price
        open_lots["unrealized_gain"] = open_lots.current_value - open_lots.invested_cost
        open_lots["holding_days"] = (valuation_date - open_lots.purchase_date).dt.days
        positions = open_lots.groupby("symbol", sort=False).agg(
            instrument_type=("instrument_type", "first"), quantity=("quantity", "sum"),
            invested_cost=("invested_cost", "sum"), current_value=("current_value", "sum"),
            latest_price=("latest_price", "first"), first_purchase=("purchase_date", "min"),
            last_purchase=("purchase_date", "max"), lots=("symbol", "size"))
        positions["average_purchase_price"] = positions.invested_cost / positions.quantity
        positions["unrealized_gain"] = positions.current_value - positions.invested_cost
    else:
        positions = pd.DataFrame(columns=["instrument_type", "quantity", "invested_cost", "current_value",
                                          "latest_price", "first_purchase", "last_purchase", "lots",
                                          "average_purchase_price", "unrealized_gain"])

    cash = cash_balance(transactions)
    holdings_value = float(positions.current_value.sum()) if not positions.empty else 0.0
    realized_gain = float(realized.realized_gain.sum()) if not realized.empty else 0.0

    external_flows = []
    for row in transactions.itertuples():
        if row.action == "deposit":
            external_flows.append((row.date, -float(row.amount)))
        elif row.action == "withdrawal":
            external_flows.append((row.date, float(row.amount)))
    external_flows.append((valuation_date, holdings_value + cash))
    xirr = compute_xirr(external_flows)

    net_deposits = sum(-cash for _, cash in external_flows[:-1])
    start_date = min((t.date for t in transactions.itertuples()), default=valuation_date)
    # modified_dietz_return uses the standard contribution-positive convention,
    # the opposite of XIRR's investor-outlay-negative convention used above.
    dietz_flows = [(date, -cash) for date, cash in external_flows[:-1]]
    dietz = modified_dietz_return(dietz_flows, 0.0, holdings_value + cash, start_date, valuation_date)

    return {
        "open_lots": open_lots, "positions": positions, "realized": realized,
        "cash_balance": cash, "holdings_value": holdings_value, "total_value": holdings_value + cash,
        "realized_gain": realized_gain,
        "unrealized_gain": float(positions.unrealized_gain.sum()) if not positions.empty else 0.0,
        "net_deposits": net_deposits, "xirr": xirr, "modified_dietz_return": dietz, "as_of": valuation_date,
    }


def save_transaction_ledger(root: Path, name: str, transactions: pd.DataFrame,
                            latest_prices: pd.Series, as_of) -> Path:
    name = name.strip()
    if not name:
        raise ValueError("Enter a ledger name.")
    book = value_ledger(transactions, latest_prices, as_of)
    key = "ledger_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:12]
    return write_artifact(root, key, {"schema_version": SCHEMA_VERSION, "kind": "transaction_ledger",
                                     "name": name, "saved_at": datetime.now(timezone.utc),
                                     "transactions": transactions, "book": book})


def load_transaction_ledger(path: Path) -> dict:
    payload = load_artifact(path)
    if payload.get("kind") != "transaction_ledger":
        raise ValueError("Unsupported transaction ledger.")
    return payload


PAPER_ACCOUNT_KEY = "paper_trading_account"


def load_paper_account(root: Path) -> pd.DataFrame | None:
    """Fixed-key artifact (not timestamped like `save_transaction_ledger`) so
    the paper trading page always reads/writes the same single account."""
    path = root / f"{PAPER_ACCOUNT_KEY}.json"
    if not path.exists():
        return None
    payload = load_artifact(path)
    if payload.get("kind") != "paper_trading_account":
        raise ValueError("Unsupported paper trading account.")
    return payload["transactions"]


def save_paper_account(root: Path, transactions: pd.DataFrame) -> Path:
    return write_artifact(root, PAPER_ACCOUNT_KEY, {"schema_version": SCHEMA_VERSION,
                                                     "kind": "paper_trading_account",
                                                     "saved_at": datetime.now(timezone.utc),
                                                     "transactions": transactions})


def transaction_ledgers(root: Path) -> dict[Path, str]:
    import json
    choices = {}
    for path in sorted(root.glob("ledger_*.json"), reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            choices[path] = f"{payload['name']} | {payload['saved_at']['$timestamp']}"
        except (ValueError, KeyError, TypeError, OSError):
            choices[path] = f"Unreadable ledger: {path.stem}"
    return choices
