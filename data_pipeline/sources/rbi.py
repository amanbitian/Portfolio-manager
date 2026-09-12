"""India macro - best effort.

RBI's DBIE has no clean public JSON API and India's e-Sankhyiki API needs a data.gov.in
key. For now we ship a small hand-maintained policy repo-rate history (the single most
useful India rate feature) and document the richer sources for later.

e-Sankhyiki:  https://esankhyiki.mospi.gov.in/api-docs
RBI DBIE:     https://data.rbi.org.in  (portal; scrape per-table CSVs)
data.gov.in:  https://api.data.gov.in  (needs api_key -> DATA_GOV_IN_KEY)
"""

from __future__ import annotations

import pandas as pd

# RBI policy repo rate, effective date -> rate %. Source: RBI press releases.
# Extend as new decisions are announced.
_REPO_RATE_HISTORY: list[tuple[str, float]] = [
    ("2000-01-01", 8.00),
    ("2004-10-27", 6.00),
    ("2005-10-26", 6.25),
    ("2006-06-09", 6.75),
    ("2007-03-31", 7.75),
    ("2008-06-25", 8.50),
    ("2008-10-20", 8.00),
    ("2008-12-08", 6.50),
    ("2009-04-21", 4.75),
    ("2010-03-19", 5.00),
    ("2010-11-02", 6.25),
    ("2011-10-25", 8.50),
    ("2012-04-17", 8.00),
    ("2013-05-03", 7.25),
    ("2014-01-28", 8.00),
    ("2015-01-15", 7.75),
    ("2015-09-29", 6.75),
    ("2016-04-05", 6.50),
    ("2017-08-02", 6.00),
    ("2018-06-06", 6.25),
    ("2018-08-01", 6.50),
    ("2019-02-07", 6.25),
    ("2019-06-06", 5.75),
    ("2019-10-04", 5.15),
    ("2020-03-27", 4.40),
    ("2020-05-22", 4.00),
    ("2022-05-04", 4.40),
    ("2022-06-08", 4.90),
    ("2022-08-05", 5.40),
    ("2022-09-30", 5.90),
    ("2022-12-07", 6.25),
    ("2023-02-08", 6.50),
    ("2024-10-09", 6.50),
    ("2025-02-07", 6.25),
    ("2025-04-09", 6.00),
    ("2025-06-06", 5.50),
]


def repo_rate_history() -> pd.DataFrame:
    df = pd.DataFrame(_REPO_RATE_HISTORY, columns=["date", "value"])
    df["date"] = pd.to_datetime(df["date"])
    df["series_id"] = "RBI:REPO_RATE"
    df["label"] = "RBI policy repo rate %"
    df["unit"] = "percent"
    df["source"] = "rbi_manual"
    return df[["series_id", "label", "date", "value", "unit", "source"]]
