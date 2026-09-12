"""Corporate actions (dividends + splits) from yfinance, with a cumulative
back-adjustment factor for building split/dividend-adjusted return series later.

yfinance only exposes dividends and split ratios. Bonus / rights issues would need
NSE's corporate-actions feed (bot-protected; add later with a primed session).
"""

from __future__ import annotations

import logging

import pandas as pd

from ._yf import throttle, ticker

LOG = logging.getLogger("data_pipeline.corporate_actions")


def fetch_actions(yf_ticker: str, symbol: str) -> pd.DataFrame:
    tk = ticker(yf_ticker)
    try:
        throttle()
        actions = tk.actions
    except Exception as exc:  # noqa: BLE001
        LOG.debug("%s actions raised %s", symbol, exc)
        actions = None
    if actions is None or actions.empty:
        return pd.DataFrame(
            columns=["symbol", "ex_date", "action_type", "ratio", "amount", "source"]
        )

    actions = actions.reset_index().rename(columns={"Date": "ex_date"})
    rows: list[dict] = []
    for _, r in actions.iterrows():
        ex = pd.Timestamp(r["ex_date"]).tz_localize(None).normalize()
        div = float(r.get("Dividends", 0.0) or 0.0)
        split = float(r.get("Stock Splits", 0.0) or 0.0)
        if div > 0:
            rows.append(
                {
                    "symbol": symbol,
                    "ex_date": ex,
                    "action_type": "dividend",
                    "ratio": pd.NA,
                    "amount": div,
                    "source": "yfinance",
                }
            )
        if split and split != 0.0:
            rows.append(
                {
                    "symbol": symbol,
                    "ex_date": ex,
                    "action_type": "split",
                    "ratio": split,  # e.g. 2.0 => 1 share becomes 2
                    "amount": pd.NA,
                    "source": "yfinance",
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("ex_date").reset_index(drop=True)

    # Back-adjustment factor: price on/before an ex-date multiplied by the product of
    # all *later* split ratios (dividends left out - keep it a pure price factor).
    splits = df[df["action_type"] == "split"][["ex_date", "ratio"]]
    df["split_adj_factor"] = 1.0
    for _, s in splits.iterrows():
        df.loc[df["ex_date"] < s["ex_date"], "split_adj_factor"] *= float(s["ratio"])
    return df
