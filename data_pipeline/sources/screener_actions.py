"""Derive bonus/split corporate actions from Screener's own balance-sheet history.

yfinance's ``.actions`` misses a lot of Indian bonus issues. Screener's balance sheet
carries ``No. of Equity Shares`` and ``Face value`` every year, so we can detect the
events directly:

- **bonus**: a jump in ``No. of Equity Shares`` (>=15%) *corroborated* by a populated
  ``New Bonus Shares`` cell in that same period. Screener repeats/carries that column
  forward in later years without a real event, so the corroboration is paired with an
  actual share-count jump - a lone populated cell with no jump is not treated as a bonus.
- **split**: a drop in ``Face value`` between consecutive annual periods.

``ex_date`` is approximated as the fiscal period-end the event landed in - Screener
gives no exact date, so this is good for multi-year price adjustment, not for tight
event-window studies. Rows are tagged ``source="screener_derived"``.
"""

from __future__ import annotations

import pandas as pd

JUMP_THRESHOLD = 1.15  # >=15% share-count growth, corroborated by New Bonus Shares


def _pivot(balance_sheet: pd.DataFrame) -> pd.DataFrame:
    bs = balance_sheet
    if "freq" in bs.columns:
        bs = bs[bs["freq"] == "A"]
    return bs.pivot_table(index="period_end", columns="line_item", values="value", aggfunc="last").sort_index()


def derive_actions(balance_sheet: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if balance_sheet is None or balance_sheet.empty:
        return pd.DataFrame()
    piv = _pivot(balance_sheet)
    cols = {c.lower(): c for c in piv.columns}
    shares_col = next((cols[k] for k in cols if "no. of equity shares" in k or "no of equity shares" in k), None)
    bonus_col = next((cols[k] for k in cols if "new bonus" in k), None)
    face_col = next((cols[k] for k in cols if "face value" in k), None)

    rows: list[dict] = []

    # Build split ratios first so a share-count jump that's fully explained by a
    # same-period split isn't *also* counted as a bonus.
    split_ratio_by_date: dict = {}
    if face_col is not None:
        fv = piv[face_col].dropna()
        for i in range(1, len(fv)):
            d, prev, cur = fv.index[i], fv.iloc[i - 1], fv.iloc[i]
            if cur > 0 and cur < prev - 0.01:
                ratio = prev / cur
                split_ratio_by_date[d] = ratio
                rows.append(
                    {
                        "symbol": symbol,
                        "ex_date": pd.Timestamp(d).normalize(),
                        "action_type": "split",
                        "ratio": round(float(ratio), 4),
                        "amount": pd.NA,
                        "source": "screener_derived",
                    }
                )

    if shares_col is not None:
        shares = piv[shares_col].dropna()
        for i in range(1, len(shares)):
            d, prev, cur = shares.index[i], shares.iloc[i - 1], shares.iloc[i]
            if prev <= 0:
                continue
            ratio = cur / prev
            if ratio < JUMP_THRESHOLD:
                continue
            bonus_value = piv.loc[d, bonus_col] if bonus_col is not None else None
            # "New Bonus Shares" is a stale running total Screener carries forward
            # even in periods with no bonus, and a non-bonus period can still report
            # it as an explicit 0 - either way, only a *positive* value is evidence.
            corroborated = bonus_value is not None and pd.notna(bonus_value) and float(bonus_value) > 0
            if not corroborated:
                continue
            split_ratio = split_ratio_by_date.get(d)
            if split_ratio is not None and abs(split_ratio - ratio) <= 0.05 * ratio:
                # the same-period split already accounts for this share-count jump.
                continue
            rows.append(
                {
                    "symbol": symbol,
                    "ex_date": pd.Timestamp(d).normalize(),
                    "action_type": "bonus",
                    "ratio": round(float(ratio), 4),
                    "amount": pd.NA,
                    "source": "screener_derived",
                }
            )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("ex_date").reset_index(drop=True)


def merge_actions(
    existing: pd.DataFrame, derived: pd.DataFrame, *, tolerance_days: int = 400
) -> pd.DataFrame:
    """Combine yfinance-sourced actions with Screener-derived ones.

    A derived candidate is dropped if a split/bonus already exists in ``existing``
    within ``tolerance_days`` (avoids double-counting the same event from two
    sources). The default is wide (>1 year) because Screener's approximated ex_date
    is a *fiscal year-end*, which can land up to ~11 months after the real ex-date -
    a same-year real match can otherwise look >200 days apart.
    ``split_adj_factor`` is recomputed across the merged, deduped set.
    """
    if derived.empty:
        return existing

    if existing is None or existing.empty:
        keep = derived
        base = pd.DataFrame()
    else:
        known = existing.loc[
            existing["action_type"].isin(["split", "bonus"]), "ex_date"
        ]
        known = pd.to_datetime(known)

        def _is_dupe(d: pd.Timestamp) -> bool:
            return any(abs((pd.Timestamp(d) - e).days) <= tolerance_days for e in known)

        keep = derived[~derived["ex_date"].map(_is_dupe)]
        base = existing

    combined = pd.concat([base, keep], ignore_index=True) if not base.empty else pd.concat(
        [pd.DataFrame(columns=derived.columns), keep], ignore_index=True
    )
    if combined.empty:
        return combined
    combined = combined.sort_values("ex_date").reset_index(drop=True)

    splits = combined[combined["action_type"].isin(["split", "bonus"])][["ex_date", "ratio"]].dropna()
    combined["split_adj_factor"] = 1.0
    for _, s in splits.iterrows():
        combined.loc[combined["ex_date"] < s["ex_date"], "split_adj_factor"] *= float(s["ratio"])
    return combined
