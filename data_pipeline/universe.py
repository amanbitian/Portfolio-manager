"""Build the equity universe: constituents + ISIN + Upstox instrument_key.

Fallback chain for the constituent list (first that yields rows wins):

1. ``reference/universe_manual.csv`` you drop in yourself
   (columns: ``symbol,isin`` required; ``name,industry`` optional).
   This is how you load the *true* Nifty 1000 - download it from
   https://www.niftyindices.com  ->  Nifty 1000  ->  "Download (.csv)".
2. NSE archives ``ind_nifty1000list.csv`` (not currently published, kept for the day it is).
3. NSE archives ``ind_niftytotalmarket_list.csv`` - the Nifty Total Market (~750 names),
   the broadest list NSE publishes as a free CSV. Used with ``source`` flagged so it is
   obvious the universe is 750, not 1000.

Every constituent is then joined by ISIN to the Upstox NSE instrument dump to obtain the
``instrument_key`` needed by the historical-candle API.
"""

from __future__ import annotations

import gzip
import io
import logging
from datetime import datetime, timezone

import pandas as pd

from .config import PipelineConfig
from .http import HttpClient
from .io import write_parquet

LOG = logging.getLogger("data_pipeline.universe")

NSE_ARCHIVE = "https://nsearchives.nseindia.com/content/indices/{name}.csv"
UPSTOX_NSE_INSTRUMENTS = (
    "https://assets.upstox.com/market-quote/instruments/exchange/NSE.csv.gz"
)

_NSE_COLS = {
    "Company Name": "name",
    "Industry": "industry",
    "Symbol": "symbol",
    "ISIN Code": "isin",
}


def _from_manual_csv(cfg: PipelineConfig) -> pd.DataFrame | None:
    path = cfg.reference_dir / "universe_manual.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "symbol" not in df.columns or "isin" not in df.columns:
        LOG.warning("%s present but missing symbol/isin columns; ignoring", path)
        return None
    for opt in ("name", "industry"):
        if opt not in df.columns:
            df[opt] = pd.NA
    df["source"] = "manual_csv"
    LOG.info("universe: %d rows from %s", len(df), path)
    return df[["symbol", "isin", "name", "industry", "source"]]


def _from_nse_archive(client: HttpClient, name: str, source: str) -> pd.DataFrame | None:
    try:
        resp = client.get(
            NSE_ARCHIVE.format(name=name), host_key="nse", expect_json=False
        )
    except Exception as exc:  # noqa: BLE001 - fallback chain, log and move on
        LOG.warning("universe: %s fetch failed (%s)", name, exc)
        return None
    ctype = resp.headers.get("content-type", "")
    if "csv" not in ctype:
        LOG.warning("universe: %s returned %s, not csv", name, ctype)
        return None
    df = pd.read_csv(io.StringIO(resp.text))
    df = df.rename(columns=_NSE_COLS)
    if "symbol" not in df.columns or "isin" not in df.columns:
        return None
    df["source"] = source
    LOG.info("universe: %d rows from NSE %s", len(df), name)
    return df[["symbol", "isin", "name", "industry", "source"]]


def _load_constituents(cfg: PipelineConfig, client: HttpClient) -> pd.DataFrame:
    manual = _from_manual_csv(cfg)
    if manual is not None and not manual.empty:
        return manual

    n1000 = _from_nse_archive(client, "ind_nifty1000list", "nse_nifty1000")
    if n1000 is not None and not n1000.empty:
        return n1000

    tm = _from_nse_archive(
        client, "ind_niftytotalmarket_list", "nse_nifty_total_market_750"
    )
    if tm is not None and not tm.empty:
        LOG.warning(
            "universe: using Nifty Total Market (~750) - the true Nifty 1000 is not a "
            "free NSE CSV. Drop reference/universe_manual.csv to override."
        )
        return tm

    raise RuntimeError(
        "Could not obtain a constituent list from any source. "
        "Create reference/universe_manual.csv with columns symbol,isin."
    )


def _upstox_instruments(cfg: PipelineConfig, client: HttpClient) -> pd.DataFrame:
    resp = client.get(UPSTOX_NSE_INSTRUMENTS, host_key="assets", expect_json=False)
    raw = gzip.decompress(resp.content).decode("utf-8")
    inst = pd.read_csv(io.StringIO(raw))
    write_parquet(
        inst,
        cfg.reference_dir / "upstox_instruments_nse.parquet",
        merge=False,
    )
    eq = inst[inst["instrument_type"].str.upper() == "EQUITY"].copy()
    # instrument_key like "NSE_EQ|INE002A01018"; the token after "|" is the ISIN.
    eq["isin"] = eq["instrument_key"].str.split("|").str[-1]
    eq = eq[eq["isin"].str.startswith("INE", na=False)]
    return eq[["isin", "instrument_key", "tradingsymbol", "name"]].rename(
        columns={"tradingsymbol": "upstox_tradingsymbol", "name": "upstox_name"}
    )


def build_universe(cfg: PipelineConfig) -> pd.DataFrame:
    cfg.ensure_dirs()
    client = HttpClient(cfg)

    cons = _load_constituents(cfg, client)
    cons["symbol"] = cons["symbol"].astype(str).str.strip().str.upper()
    cons["isin"] = cons["isin"].astype(str).str.strip().str.upper()
    cons = cons.drop_duplicates(subset=["symbol"])

    # NSE seeds its index CSVs with DUMMY* placeholder rows - drop them.
    dummies = cons["symbol"].str.startswith("DUMMY")
    if dummies.any():
        LOG.info("universe: dropping %d DUMMY placeholder rows", int(dummies.sum()))
        cons = cons[~dummies].reset_index(drop=True)

    inst = _upstox_instruments(cfg, client)
    merged = cons.merge(inst, on="isin", how="left")

    unmatched = merged[merged["instrument_key"].isna()]
    if not unmatched.empty:
        LOG.warning(
            "universe: %d/%d symbols have no Upstox instrument_key (dropped from price "
            "ingestion): %s",
            len(unmatched),
            len(merged),
            ", ".join(unmatched["symbol"].head(30)),
        )
        (cfg.log_dir / "universe_unmatched.csv").write_text(
            unmatched.to_csv(index=False), encoding="utf-8"
        )

    merged["yf_ticker"] = merged["symbol"] + ".NS"
    merged["ingested_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    cols = [
        "symbol",
        "isin",
        "name",
        "industry",
        "instrument_key",
        "yf_ticker",
        "upstox_tradingsymbol",
        "source",
        "ingested_at",
    ]
    out = merged[cols].sort_values("symbol").reset_index(drop=True)
    write_parquet(out, cfg.reference_dir / "universe.parquet", merge=False)
    LOG.info(
        "universe: wrote %d rows (%d with instrument_key) -> %s",
        len(out),
        int(out["instrument_key"].notna().sum()),
        cfg.reference_dir / "universe.parquet",
    )
    return out


def load_universe(cfg: PipelineConfig, *, only_priceable: bool = False) -> pd.DataFrame:
    path = cfg.reference_dir / "universe.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing - run `py -3 -m data_pipeline.run universe` first."
        )
    df = pd.read_parquet(path)
    if only_priceable:
        df = df[df["instrument_key"].notna()].reset_index(drop=True)
    return df
