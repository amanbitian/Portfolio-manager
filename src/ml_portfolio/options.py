"""Deterministic option strategy analytics used by the Options Lab.

The module deliberately has no market-data or broker dependency.  Prices and
Greeks are Black--Scholes estimates from the assumptions supplied by the user;
that keeps the sandbox useful while making it impossible to mistake the output
for an executable quote.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log, sqrt

import numpy as np
import pandas as pd
from scipy.stats import norm


OPTION_TYPES = ("Call", "Put")
SIDES = ("Buy", "Sell")
STRATEGIES = (
    "Long Straddle",
    "Long Strangle",
    "Iron Condor",
    "Call Butterfly",
    "Batman (double butterfly)",
    "Custom",
)


@dataclass(frozen=True)
class OptionAssumptions:
    spot: float
    days_to_expiry: int
    volatility: float
    rate: float = 0.06
    dividend_yield: float = 0.0
    lot_size: int = 1
    strategy_lots: int = 1

    @property
    def years(self) -> float:
        return max(self.days_to_expiry, 0) / 365.0

    @property
    def units(self) -> int:
        return self.lot_size * self.strategy_lots


def _validate_option_inputs(spot: float, strike: float, years: float, volatility: float) -> None:
    if spot <= 0 or strike <= 0:
        raise ValueError("Spot and strike must be positive.")
    if years < 0:
        raise ValueError("Time to expiry cannot be negative.")
    if volatility < 0:
        raise ValueError("Volatility cannot be negative.")


def black_scholes(
    spot: float,
    strike: float,
    years: float,
    volatility: float,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
    option_type: str = "Call",
) -> dict[str, float]:
    """Return price and standard Greeks for one European option unit.

    Theta is returned per calendar day and vega per one volatility point (a
    change of 1%, not a change of 100%).
    """
    _validate_option_inputs(spot, strike, years, volatility)
    if option_type not in OPTION_TYPES:
        raise ValueError(f"Option type must be one of {OPTION_TYPES}.")
    if years == 0 or volatility == 0:
        intrinsic = max(spot - strike, 0.0) if option_type == "Call" else max(strike - spot, 0.0)
        if option_type == "Call":
            delta = 1.0 if spot > strike else (0.5 if spot == strike else 0.0)
        else:
            delta = -1.0 if spot < strike else (-0.5 if spot == strike else 0.0)
        return {"price": intrinsic, "delta": delta, "gamma": 0.0, "theta": 0.0, "vega": 0.0}

    root_t = sqrt(years)
    d1 = (log(spot / strike) + (rate - dividend_yield + 0.5 * volatility**2) * years) / (volatility * root_t)
    d2 = d1 - volatility * root_t
    spot_discount = exp(-dividend_yield * years)
    strike_discount = exp(-rate * years)
    density = norm.pdf(d1)
    gamma = spot_discount * density / (spot * volatility * root_t)
    vega = spot * spot_discount * density * root_t / 100.0
    common_theta = -(spot * spot_discount * density * volatility) / (2.0 * root_t)
    if option_type == "Call":
        price = spot * spot_discount * norm.cdf(d1) - strike * strike_discount * norm.cdf(d2)
        delta = spot_discount * norm.cdf(d1)
        theta = common_theta - rate * strike * strike_discount * norm.cdf(d2) \
            + dividend_yield * spot * spot_discount * norm.cdf(d1)
    else:
        price = strike * strike_discount * norm.cdf(-d2) - spot * spot_discount * norm.cdf(-d1)
        delta = spot_discount * (norm.cdf(d1) - 1.0)
        theta = common_theta + rate * strike * strike_discount * norm.cdf(-d2) \
            - dividend_yield * spot * spot_discount * norm.cdf(-d1)
    return {"price": float(price), "delta": float(delta), "gamma": float(gamma),
            "theta": float(theta / 365.0), "vega": float(vega)}


def strategy_legs(strategy: str, spot: float, spacing: float) -> pd.DataFrame:
    """Create an editable strategy template with strikes rounded to spacing."""
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}")
    if spot <= 0 or spacing <= 0:
        raise ValueError("Spot and strike spacing must be positive.")
    at_the_money = round(spot / spacing) * spacing
    templates = {
        "Long Straddle": [("Buy", "Call", at_the_money, 1), ("Buy", "Put", at_the_money, 1)],
        "Long Strangle": [("Buy", "Put", at_the_money - spacing, 1),
                          ("Buy", "Call", at_the_money + spacing, 1)],
        "Iron Condor": [("Buy", "Put", at_the_money - 2 * spacing, 1),
                        ("Sell", "Put", at_the_money - spacing, 1),
                        ("Sell", "Call", at_the_money + spacing, 1),
                        ("Buy", "Call", at_the_money + 2 * spacing, 1)],
        "Call Butterfly": [("Buy", "Call", at_the_money - spacing, 1),
                           ("Sell", "Call", at_the_money, 2),
                           ("Buy", "Call", at_the_money + spacing, 1)],
        # Two adjacent long call butterflies.  The expiry payoff has the two
        # characteristic peaks that give this structure its informal name.
        "Batman (double butterfly)": [("Buy", "Call", at_the_money - 2 * spacing, 1),
                                      ("Sell", "Call", at_the_money - spacing, 2),
                                      ("Buy", "Call", at_the_money, 2),
                                      ("Sell", "Call", at_the_money + spacing, 2),
                                      ("Buy", "Call", at_the_money + 2 * spacing, 1)],
        "Custom": [("Buy", "Call", at_the_money, 1)],
    }
    return pd.DataFrame(templates[strategy], columns=["side", "option_type", "strike", "quantity"])


def validate_legs(legs: pd.DataFrame) -> pd.DataFrame:
    required = ["side", "option_type", "strike", "quantity"]
    missing = [column for column in required if column not in legs]
    if missing:
        raise ValueError(f"Missing leg columns: {', '.join(missing)}")
    cleaned = legs.loc[:, required].copy()
    if cleaned.empty:
        raise ValueError("Add at least one option leg.")
    if not cleaned.side.isin(SIDES).all():
        raise ValueError("Each side must be Buy or Sell.")
    if not cleaned.option_type.isin(OPTION_TYPES).all():
        raise ValueError("Each option type must be Call or Put.")
    cleaned["strike"] = pd.to_numeric(cleaned.strike, errors="coerce")
    cleaned["quantity"] = pd.to_numeric(cleaned.quantity, errors="coerce")
    if cleaned[["strike", "quantity"]].isna().any().any():
        raise ValueError("Strike and quantity must be numbers.")
    if (cleaned.strike <= 0).any() or (cleaned.quantity <= 0).any():
        raise ValueError("Strike and quantity must be positive.")
    return cleaned


def price_legs(legs: pd.DataFrame, assumptions: OptionAssumptions, *, days_remaining: int | None = None,
               spot: float | None = None, volatility: float | None = None) -> pd.DataFrame:
    """Price every leg and return signed position values and Greeks."""
    legs = validate_legs(legs)
    scenario_spot = assumptions.spot if spot is None else float(spot)
    scenario_vol = assumptions.volatility if volatility is None else max(float(volatility), 0.0)
    scenario_days = assumptions.days_to_expiry if days_remaining is None else max(int(days_remaining), 0)
    rows = []
    for row in legs.itertuples(index=False):
        result = black_scholes(scenario_spot, float(row.strike), scenario_days / 365.0, scenario_vol,
                               assumptions.rate, assumptions.dividend_yield, row.option_type)
        sign = 1.0 if row.side == "Buy" else -1.0
        scale = sign * float(row.quantity) * assumptions.units
        rows.append({**row._asdict(), "model_price": result["price"],
                     "position_value": result["price"] * scale,
                     "delta": result["delta"] * scale, "gamma": result["gamma"] * scale,
                     "theta_per_day": result["theta"] * scale, "vega_per_point": result["vega"] * scale})
    return pd.DataFrame(rows)


def expiry_payoff(legs: pd.DataFrame, assumptions: OptionAssumptions,
                  terminal_spots: np.ndarray | None = None) -> pd.DataFrame:
    """Calculate strategy P&L at expiry, including the modelled entry premium."""
    legs = validate_legs(legs)
    if terminal_spots is None:
        strikes = legs.strike.to_numpy(float)
        upper = max(assumptions.spot, strikes.max()) * 1.5
        # Zero is a valid terminal boundary and is required to measure the full
        # risk of an uncovered short put. Keeping it in the grid also avoids a
        # deceptively truncated payoff chart for custom strategies.
        terminal_spots = np.unique(np.r_[np.linspace(0.0, upper, 1001), strikes, assumptions.spot])
    terminal_spots = np.asarray(terminal_spots, dtype=float)
    initial_value = price_legs(legs, assumptions).position_value.sum()
    pnl = np.full(terminal_spots.shape, -initial_value, dtype=float)
    for row in legs.itertuples(index=False):
        intrinsic = (np.maximum(terminal_spots - float(row.strike), 0.0) if row.option_type == "Call"
                     else np.maximum(float(row.strike) - terminal_spots, 0.0))
        sign = 1.0 if row.side == "Buy" else -1.0
        pnl += intrinsic * sign * float(row.quantity) * assumptions.units
    return pd.DataFrame({"terminal_spot": terminal_spots, "pnl": pnl}).sort_values("terminal_spot")


def _breakevens(payoff: pd.DataFrame) -> list[float]:
    spots = payoff.terminal_spot.to_numpy(float)
    pnl = payoff.pnl.to_numpy(float)
    roots: list[float] = []
    exact = np.flatnonzero(np.isclose(pnl, 0.0, atol=1e-8))
    roots.extend(spots[exact].tolist())
    crossings = np.flatnonzero(pnl[:-1] * pnl[1:] < 0)
    for index in crossings:
        x0, x1 = spots[index:index + 2]
        y0, y1 = pnl[index:index + 2]
        roots.append(float(x0 - y0 * (x1 - x0) / (y1 - y0)))
    return sorted({round(root, 4) for root in roots})


def _tail_slopes(legs: pd.DataFrame, units: int) -> tuple[float, float]:
    # For S below every positive strike puts have slope -1; above every strike
    # calls have slope +1.  A non-zero high tail means profit/loss is unbounded.
    signed = np.where(legs.side.eq("Buy"), 1.0, -1.0) * legs.quantity.to_numpy(float) * units
    low = float((-signed[legs.option_type.eq("Put")]).sum())
    high = float(signed[legs.option_type.eq("Call")].sum())
    return low, high


def strategy_summary(legs: pd.DataFrame, assumptions: OptionAssumptions) -> dict[str, object]:
    """Aggregate premium, risk, probability, expectation, and Greeks."""
    legs = validate_legs(legs)
    priced = price_legs(legs, assumptions)
    payoff = expiry_payoff(legs, assumptions)
    _low_slope, high_slope = _tail_slopes(legs, assumptions.units)
    max_profit: float | None = float(payoff.pnl.max())
    max_loss: float | None = float(payoff.pnl.min())
    # The underlying is bounded at zero, so only the high-price tail can be
    # truly unbounded. Puts can still have a large, but finite, zero-price
    # payoff.
    if high_slope > 0:
        max_profit = None
    if high_slope < 0:
        max_loss = None

    # Integrate the positive-payoff states under the same risk-neutral
    # lognormal distribution used by Black--Scholes.  This is a model estimate,
    # not an empirical forecast.
    sigma_t = assumptions.volatility * sqrt(max(assumptions.years, 1e-12))
    if sigma_t == 0:
        weights = np.zeros(len(payoff))
        weights[int(np.abs(payoff.terminal_spot.to_numpy() - assumptions.spot).argmin())] = 1.0
    else:
        spots = payoff.terminal_spot.to_numpy(float)
        edges = np.r_[0.0, (spots[:-1] + spots[1:]) / 2.0, np.inf]
        finite_edges = edges[1:-1]
        z = (np.log(finite_edges / assumptions.spot)
             - (assumptions.rate - assumptions.dividend_yield - 0.5 * assumptions.volatility**2)
             * assumptions.years) / sigma_t
        cdf = np.r_[0.0, norm.cdf(z), 1.0]
        weights = np.diff(cdf)
    probability_profit = float(weights[payoff.pnl.to_numpy() > 0].sum())
    expected_pnl = float(np.dot(weights, payoff.pnl.to_numpy()))
    net_premium = float(-priced.position_value.sum())  # positive means credit received
    return {
        "net_premium": net_premium,
        "max_profit": max_profit,
        "max_loss": max_loss,
        "breakevens": _breakevens(payoff),
        "probability_profit": probability_profit,
        "expected_pnl": expected_pnl,
        "delta": float(priced.delta.sum()),
        "gamma": float(priced.gamma.sum()),
        "theta_per_day": float(priced.theta_per_day.sum()),
        "vega_per_point": float(priced.vega_per_point.sum()),
        "entry_debit": float(priced.position_value.sum()),
        "priced_legs": priced,
        "payoff": payoff,
    }


def scenario_matrix(legs: pd.DataFrame, assumptions: OptionAssumptions,
                    spot_moves: tuple[float, ...] = (-0.10, -0.05, 0.0, 0.05, 0.10),
                    vol_moves: tuple[float, ...] = (-0.05, 0.0, 0.05),
                    days_remaining: int | None = None) -> pd.DataFrame:
    """Return mark-to-model P&L for a grid of spot and absolute IV changes."""
    entry_value = float(price_legs(legs, assumptions).position_value.sum())
    remaining = max(assumptions.days_to_expiry // 2, 0) if days_remaining is None else days_remaining
    values: dict[str, list[float]] = {}
    for vol_move in vol_moves:
        column = f"IV {vol_move:+.0%}"
        values[column] = []
        for spot_move in spot_moves:
            scenario = price_legs(legs, assumptions, days_remaining=remaining,
                                  spot=assumptions.spot * (1.0 + spot_move),
                                  volatility=max(assumptions.volatility + vol_move, 0.0001))
            values[column].append(float(scenario.position_value.sum() - entry_value))
    return pd.DataFrame(values, index=pd.Index([f"Spot {move:+.0%}" for move in spot_moves], name="Scenario"))
