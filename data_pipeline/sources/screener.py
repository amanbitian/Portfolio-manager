"""Parser for screener.in Excel exports (the "Export to Excel" button on a company page).

The workbook has one sheet - usually "Data Sheet" - with sections stacked vertically:

    PROFIT & LOSS
    Report Date        Mar-2013   Mar-2014   ...
    Sales              ...
    ...
    QUARTERS
    Report Date        Jun-2022   Sep-2022   ...
    ...
    BALANCE SHEET
    ...
    CASH FLOW:
    ...
    PRICE:
    Date               ...
    Price              ...

Layout varies (banks differ, row sets change across export versions), so this parser
is keyword-tolerant: it locates sections by header text, finds the date row inside each,
and treats every following labelled numeric row as data until the next section.

Output: one long-form DataFrame per section with columns
``[line_item, period_end, value]``.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

LOG = logging.getLogger("data_pipeline.screener")

# section header text (upper, punctuation-stripped) -> our short name
SECTION_ALIASES: dict[str, str] = {
    "PROFIT LOSS": "pnl",
    "PROFIT AND LOSS": "pnl",
    "QUARTERS": "quarters",
    "QUARTERLY RESULTS": "quarters",
    "BALANCE SHEET": "balance_sheet",
    "CASH FLOW": "cash_flow",
    "PRICE": "price",
    "DERIVED": "derived",
}

_DATE_LABELS = {"report date", "date", "narration"}
_NON_DATA_LABELS = {"", "nan", "adjusted equity shares in cr"}
# Some sections (notably "PRICE:") carry their values directly on the header row itself,
# reusing the date columns set by the most recent "Report Date" row rather than having
# their own. Display name used for that inline row's line_item.
_SECTION_DISPLAY = {"price": "Price", "derived": "Derived"}


def _norm_header(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^A-Z ]", " ", str(text).upper())).strip()


def _to_period(value: object) -> pd.Timestamp | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    ts = pd.to_datetime(value, errors="coerce", dayfirst=False)
    if ts is pd.NaT or pd.isna(ts):
        # screener sometimes uses "Mar-2014" style
        ts = pd.to_datetime(str(value), errors="coerce", format="%b-%Y")
    return None if pd.isna(ts) else pd.Timestamp(ts).normalize()


def _load_sheet(path: Path) -> pd.DataFrame:
    xls = pd.ExcelFile(path)
    for name in xls.sheet_names:
        if name.strip().lower() in {"data sheet", "data"}:
            return pd.read_excel(xls, sheet_name=name, header=None, dtype=object)
    return pd.read_excel(xls, sheet_name=0, header=None, dtype=object)


def parse_workbook(path: Path) -> dict[str, pd.DataFrame]:
    grid = _load_sheet(path)
    n_rows, n_cols = grid.shape
    sections: dict[str, pd.DataFrame] = {}

    current: str | None = None
    date_map: dict[int, pd.Timestamp] = {}
    rows: list[dict] = []

    def flush() -> None:
        nonlocal rows
        if current and rows:
            df = pd.DataFrame(rows)
            df = df.dropna(subset=["value"]).reset_index(drop=True)
            if not df.empty:
                sections[current] = pd.concat(
                    [sections.get(current, pd.DataFrame()), df], ignore_index=True
                )
        rows = []

    for r in range(n_rows):
        label_cell = grid.iat[r, 0]
        label = "" if label_cell is None else str(label_cell).strip()
        header = _norm_header(label)

        # section header? match the whole normalized label, not a substring - a data
        # row like "Cash Flow from Operating Activities" also contains "CASH FLOW"
        # and must not be mistaken for the "CASH FLOW:" section header itself.
        matched = SECTION_ALIASES.get(header)
        if matched:
            n_vals = sum(pd.notna(grid.iat[r, c]) for c in range(1, n_cols))
            flush()
            if n_vals > 1 and date_map:
                # values live on the header row itself (e.g. "PRICE:" year-end prices) -
                # reuse whichever date row was last seen (sections without their own
                # "Report Date" row, like PRICE, are keyed off the preceding section's).
                inline = [
                    {
                        "line_item": _SECTION_DISPLAY.get(matched, matched.title()),
                        "period_end": period,
                        "value": float(val),
                    }
                    for c, period in date_map.items()
                    if c < n_cols and pd.notna(val := pd.to_numeric(grid.iat[r, c], errors="coerce"))
                ]
                if inline:
                    sections[matched] = pd.concat(
                        [sections.get(matched, pd.DataFrame()), pd.DataFrame(inline)],
                        ignore_index=True,
                    )
            current = matched
            continue

        if current is None:
            continue

        low = label.lower().rstrip(":")
        if low in _DATE_LABELS or (not date_map and low.startswith("report date")):
            date_map = {}
            for c in range(1, n_cols):
                p = _to_period(grid.iat[r, c])
                if p is not None:
                    date_map[c] = p
            continue

        if not date_map or low in _NON_DATA_LABELS:
            continue

        for c, period in date_map.items():
            val = pd.to_numeric(grid.iat[r, c], errors="coerce") if c < n_cols else None
            if pd.notna(val):
                rows.append({"line_item": label, "period_end": period, "value": float(val)})

    flush()
    return sections


def symbol_from_path(path: Path) -> str:
    """`RELIANCE.xlsx` / `Reliance Industries.xlsx` -> best-effort NSE symbol."""
    stem = path.stem.strip()
    # a bare ticker export is already the symbol
    if re.fullmatch(r"[A-Za-z0-9&.\-]{1,20}", stem):
        return stem.upper()
    # otherwise take the first token, uppercase, strip non-alnum
    token = re.split(r"[\s_\-]", stem)[0]
    return re.sub(r"[^A-Z0-9]", "", token.upper())
