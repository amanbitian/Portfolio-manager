"""Build a compact, browsable summary of the ingested mutual fund NAV history.

`ingest/mutual_funds.py` pulls raw daily NAV history per scheme into
~38k individual Parquet files (`fundamentals/mutual_funds/nav/{scheme_code}.parquet`)
- fine for the ingest, useless to search or rank directly: opening every file to
answer "what are the top-performing funds" takes minutes, not the sub-second
response an interactive app needs. This module reads that raw history ONCE
(threaded - I/O bound, one small file per scheme) and derives one row per
scheme with the fields a browse/search/rank UI actually needs: latest NAV,
trailing 1y/3y returns, and the category/fund-house metadata that only lives
inside each NAV file (the scheme list itself carries no category).

    py -3 -m data_pipeline.run mutual-fund-catalog
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .config import PipelineConfig
from .io import write_parquet

LOG = logging.getLogger("data_pipeline.mutual_fund_catalog")

CATALOG_COLUMNS = [
    "scheme_code", "scheme_name", "fund_house", "scheme_type", "scheme_category",
    "latest_date", "latest_nav", "first_date", "data_points",
    "return_1y", "return_3y_cagr",
]


def _trailing_return(nav: pd.Series, dates: pd.DatetimeIndex, years: float) -> float | None:
    """Nearest NAV on/before `latest - years` years, matched by calendar days
    (not row count) since publication gaps vary a lot scheme to scheme."""
    target = dates[-1] - pd.Timedelta(days=int(years * 365.25))
    if target < dates[0]:
        return None
    idx = dates.searchsorted(target, side="right") - 1
    if idx < 0:
        return None
    base = nav.iloc[idx]
    if base <= 0:
        return None
    return float(nav.iloc[-1] / base - 1)


def _summarize_one(path: Path) -> dict | None:
    df = pd.read_parquet(path, columns=["date", "nav", "scheme_code", "fund_house", "scheme_category",
                                        "scheme_type"])
    if df.empty:
        return None
    df = df.sort_values("date")
    dates = pd.DatetimeIndex(df["date"])
    nav = df["nav"].astype(float)
    years_span = (dates[-1] - dates[0]).days / 365.25
    return_1y = _trailing_return(nav, dates, 1.0) if years_span >= 1.0 else None
    return_3y = _trailing_return(nav, dates, 3.0) if years_span >= 3.0 else None
    return {
        "scheme_code": int(df["scheme_code"].iloc[-1]),
        "fund_house": df["fund_house"].iloc[-1],
        "scheme_type": df["scheme_type"].iloc[-1],
        "scheme_category": df["scheme_category"].iloc[-1],
        "latest_date": dates[-1],
        "latest_nav": float(nav.iloc[-1]),
        "first_date": dates[0],
        "data_points": len(df),
        "return_1y": return_1y,
        "return_3y_cagr": (1 + return_3y) ** (1 / 3) - 1 if return_3y is not None else None,
    }


def build_mutual_fund_catalog(cfg: PipelineConfig, *, workers: int = 64) -> pd.DataFrame:
    cfg.ensure_dirs()
    schemes_path = cfg.reference_dir / "mutual_fund_schemes.parquet"
    if not schemes_path.exists():
        raise FileNotFoundError(
            f"{schemes_path} missing - run `py -3 -m data_pipeline.run mutual-funds` first."
        )
    schemes = pd.read_parquet(schemes_path, columns=["scheme_code", "scheme_name"])
    nav_dir = cfg.fundamentals_dir / "mutual_funds" / "nav"
    paths = {int(p.stem): p for p in nav_dir.glob("*.parquet")}
    LOG.info("mutual-fund-catalog: %d schemes listed, %d have NAV files", len(schemes), len(paths))

    rows: list[dict] = []
    failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_summarize_one, path): code for code, path in paths.items()}
        with tqdm(total=len(futures), desc="mutual-fund-catalog", unit="scheme") as bar:
            for fut in as_completed(futures):
                bar.update(1)
                try:
                    row = fut.result()
                except Exception as exc:  # noqa: BLE001 - one bad file must not abort the build
                    LOG.warning("mutual-fund-catalog: %s failed: %s", futures[fut], exc)
                    failed += 1
                    continue
                if row is not None:
                    rows.append(row)

    summary = pd.DataFrame(rows)
    catalog = schemes.merge(summary, on="scheme_code", how="inner")
    catalog = catalog[CATALOG_COLUMNS].sort_values("scheme_code").reset_index(drop=True)
    write_parquet(catalog, cfg.reference_dir / "mutual_fund_catalog.parquet", merge=False)
    LOG.info("mutual-fund-catalog: wrote %d rows (%d read failures) -> %s",
             len(catalog), failed, cfg.reference_dir / "mutual_fund_catalog.parquet")
    return catalog


def load_mutual_fund_catalog(cfg: PipelineConfig) -> pd.DataFrame:
    path = cfg.reference_dir / "mutual_fund_catalog.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing - run `py -3 -m data_pipeline.run mutual-fund-catalog` first."
        )
    return pd.read_parquet(path)
