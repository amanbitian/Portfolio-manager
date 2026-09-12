"""Cross-source data sanity checker.

Runs a battery of checks over the Parquet lake and writes a report to
``_reports/sanity_<date>.{parquet,md}`` plus a console summary. Screener.in data (when
present under ``fundamentals/screener/``) is used as an independent cross-check against
the Upstox prices and yfinance fundamentals.

    py -3 -m data_pipeline.run sanity                 # whole universe
    py -3 -m data_pipeline.run sanity --symbols RELIANCE,INFY
    py -3 -m data_pipeline.run sanity --limit 50

Severities: ERROR (data is wrong / unusable), WARN (suspicious, needs a look),
INFO (worth knowing, not a problem).
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .io import read_parquet_dir
from .universe import load_universe

LOG = logging.getLogger("data_pipeline.sanity")

# Screener reports money in Rs Crore; yfinance in absolute Rs. 1 crore = 1e7.
SCREENER_MONEY_SCALE = 1e7

# canonical concept -> candidate line-item substrings (lower-case) per source
_CONCEPTS: dict[str, dict[str, list[str]]] = {
    "revenue": {
        "yf": ["total revenue", "operating revenue"],
        "screener": ["sales", "revenue", "interest earned"],
    },
    "net_income": {
        "yf": ["net income", "net income common stockholders"],
        "screener": ["net profit"],
    },
    "operating_cash_flow": {
        "yf": ["operating cash flow", "cash from operating"],
        "screener": ["cash from operating activity"],
    },
    "total_assets": {
        "yf": ["total assets"],
        "screener": ["total"],
    },
}


@dataclass
class Finding:
    dataset: str
    symbol: str
    check: str
    severity: str
    detail: str
    value: float | None = None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _safe_read(path: Path) -> pd.DataFrame:
    for _ in range(4):
        try:
            return pd.read_parquet(path)
        except (OSError, PermissionError):
            pass
    return pd.DataFrame()


def _fiscal_year(ts: pd.Timestamp) -> int:
    ts = pd.Timestamp(ts)
    return ts.year if ts.month >= 4 else ts.year - 1


def _series_for(df: pd.DataFrame, concept: str, source: str, *, freq: str = "A") -> pd.Series:
    """period_end(fiscal year int) -> value for a concept, from a long-form frame.

    Tries each candidate line-item name in priority order, exact match first, and only
    falls back to substring match if no candidate matches exactly - otherwise a name
    like "net income" also captures "Net Income From Continuing Operation ...".
    """
    if df.empty:
        return pd.Series(dtype=float)
    work = df[df["freq"] == freq] if "freq" in df.columns else df
    if work.empty:
        return pd.Series(dtype=float)
    label = work["line_item"].astype(str).str.strip().str.lower()

    for matcher in (lambda n, x: n == x, lambda n, x: n in x):
        for name in _CONCEPTS[concept][source]:
            hit = work[label.map(lambda x, n=name: matcher(n, x))]
            if not hit.empty:
                fy = hit["period_end"].map(_fiscal_year)
                return hit.assign(fy=fy).groupby("fy")["value"].last()
    return pd.Series(dtype=float)


def _rel_diff(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1.0)
    return abs(a - b) / denom


# ---------------------------------------------------------------------------
# individual checks
# ---------------------------------------------------------------------------
def check_coverage(cfg: PipelineConfig, uni: pd.DataFrame, has_screener: bool) -> list[Finding]:
    out: list[Finding] = []
    for sym in uni["symbol"]:
        if not (cfg.ohlcv_dir("1day") / f"symbol={sym}").exists():
            out.append(Finding("coverage", sym, "no_daily_prices", "ERROR",
                               "no market/ohlcv_1day partition"))
        if not (cfg.corp_actions_dir / f"{sym}.parquet").exists():
            out.append(Finding("coverage", sym, "no_corp_actions", "INFO",
                               "no corporate_actions file"))
        if not (cfg.fundamentals_dir / "income_stmt" / f"{sym}.parquet").exists():
            out.append(Finding("coverage", sym, "no_yf_fundamentals", "WARN",
                               "no yfinance income statement"))
        if has_screener and not (cfg.screener_parquet_dir / "pnl" / f"{sym}.parquet").exists():
            out.append(Finding("coverage", sym, "no_screener", "INFO",
                               "no screener export"))
    return out


def check_prices(cfg: PipelineConfig, sym: str, corp: pd.DataFrame, today: date) -> list[Finding]:
    part = cfg.ohlcv_dir("1day") / f"symbol={sym}"
    if not part.exists():
        return []
    df = read_parquet_dir(part)
    out: list[Finding] = []
    if df.empty:
        return [Finding("price_1day", sym, "empty", "ERROR", "0 rows")]

    df = df.sort_values("ts_utc").reset_index(drop=True)
    dts = pd.to_datetime(df["ts_ist"]).dt.tz_localize(None).dt.normalize()
    px = df[["open", "high", "low", "close"]].astype(float)

    n_bad_px = int((px <= 0).any(axis=1).sum())
    if n_bad_px:
        out.append(Finding("price_1day", sym, "nonpositive_price", "ERROR",
                           f"{n_bad_px} bars with a <=0 OHLC", n_bad_px))

    tol = px["close"].abs() * 1e-6 + 1e-6
    bad_ohlc = (
        (px["high"] < px["low"] - tol)
        | (px["high"] < px[["open", "close"]].max(axis=1) - tol)
        | (px["low"] > px[["open", "close"]].min(axis=1) + tol)
    )
    if bad_ohlc.any():
        out.append(Finding("price_1day", sym, "inconsistent_ohlc", "ERROR",
                           f"{int(bad_ohlc.sum())} bars where high/low don't bound open/close",
                           int(bad_ohlc.sum())))

    dup = int(df["ts_utc"].duplicated().sum())
    if dup:
        out.append(Finding("price_1day", sym, "duplicate_timestamps", "ERROR",
                           f"{dup} duplicate bar timestamps", dup))

    ret = px["close"].pct_change()
    ca_dates: list[pd.Timestamp] = []
    if not corp.empty and "ex_date" in corp.columns:
        ca_dates = [
            pd.Timestamp(d).tz_localize(None).normalize()
            for d in pd.to_datetime(corp["ex_date"], errors="coerce").dropna()
        ]
    jump_dates = list(dts[ret.abs() > 0.5])
    unexpl = [
        d for d in jump_dates
        if not any(abs((d - c).days) <= 3 for c in ca_dates)
    ]
    if unexpl:
        sample = ", ".join(str(d.date()) for d in unexpl[:3])
        out.append(Finding("price_1day", sym, "unexplained_jump", "WARN",
                           f"{len(unexpl)} daily moves >50% with no corp action nearby ({sample})",
                           len(unexpl)))

    gaps = dts.diff().dt.days
    big = gaps[gaps > 10]
    if len(big) > 1:  # ignore a single leading gap
        worst = int(gaps.max())
        when = dts.iloc[int(gaps.idxmax())].date()
        out.append(Finding("price_1day", sym, "history_gap", "WARN",
                           f"{len(big)} gaps >10 days; largest {worst}d ending {when}", worst))

    same = (px["close"].diff() == 0).astype(int)
    run = same.groupby((same == 0).cumsum()).cumsum().max()
    if run and run >= 15:
        out.append(Finding("price_1day", sym, "flatline", "WARN",
                           f"{int(run)} consecutive identical closes (possible suspension)",
                           int(run)))

    last = dts.iloc[-1].date()
    if (today - last).days > 10:
        out.append(Finding("price_1day", sym, "stale", "WARN",
                           f"last bar {last} ({(today - last).days}d ago)"))

    first = dts.iloc[0].date()
    if first > date(2015, 1, 1):
        out.append(Finding("price_1day", sym, "short_history", "INFO",
                           f"history starts {first}"))
    return out


def check_fundamentals_internal(yf_all: pd.DataFrame, sym: str) -> list[Finding]:
    out: list[Finding] = []
    inc = yf_all[(yf_all["symbol"] == sym) & (yf_all["statement"] == "income_stmt")]
    annual = inc[inc["freq"] == "A"] if "freq" in inc.columns else inc
    if annual.empty:
        out.append(Finding("fundamentals_yf", sym, "no_annual", "WARN",
                           "no annual income statement rows"))
        return out

    rev = _series_for(annual, "revenue", "yf")
    if (rev <= 0).any():
        yrs = ", ".join(str(y) for y in rev.index[rev <= 0][:3])
        out.append(Finding("fundamentals_yf", sym, "nonpositive_revenue", "ERROR",
                           f"revenue <= 0 for FY {yrs}"))

    latest = pd.Timestamp(annual["period_end"].max())
    months = (pd.Timestamp.now() - latest).days / 30.4
    if months > 18:
        out.append(Finding("fundamentals_yf", sym, "stale", "WARN",
                           f"latest annual period {latest.date()} ({months:.0f} months old)"))
    return out


def check_screener_internal(scr: dict[str, pd.DataFrame], sym: str) -> list[Finding]:
    out: list[Finding] = []
    cf = scr.get("cash_flow", pd.DataFrame())
    if not cf.empty:
        piv = cf.pivot_table(index="period_end", columns="line_item", values="value", aggfunc="last")
        cols = {c.lower(): c for c in piv.columns}
        need = ["cash from operating activity", "cash from investing activity",
                "cash from financing activity", "net cash flow"]
        if all(any(n in k for k in cols) for n in need):
            def col(sub): return piv[next(cols[k] for k in cols if sub in k)]
            recon = col("operating") + col("investing") + col("financing") - col("net cash flow")
            bad = recon.abs() > (col("net cash flow").abs().clip(lower=1) * 0.02 + 1)
            if bad.any():
                out.append(Finding("screener", sym, "cashflow_recon", "WARN",
                                   f"CFO+CFI+CFF != NetCashFlow for {int(bad.sum())} year(s)"))

    pnl = scr.get("pnl", pd.DataFrame())
    if not pnl.empty:
        sales = _series_for(pnl.assign(freq="A"), "revenue", "screener")
        if (sales <= 0).any():
            out.append(Finding("screener", sym, "nonpositive_sales", "ERROR",
                               "Screener Sales <= 0 in some year"))
    return out


def check_xcheck_fundamentals(
    yf_all: pd.DataFrame, scr: dict[str, pd.DataFrame], sym: str, industry: str = ""
) -> list[Finding]:
    pnl = scr.get("pnl", pd.DataFrame())
    if pnl.empty:
        return []
    # "Revenue" is defined differently for banks/NBFCs (interest income vs total income vs
    # financing profit) between yfinance and Screener - a 20-40% gap there is not an error.
    is_financial = bool(re.search(r"financ|bank|insuranc", str(industry), re.I))
    concepts = ("net_income",) if is_financial else ("revenue", "net_income")
    inc = yf_all[(yf_all["symbol"] == sym) & (yf_all["statement"] == "income_stmt")]
    inc = inc[inc["freq"] == "A"] if "freq" in inc.columns else inc
    out: list[Finding] = []

    # First: is yfinance consistently ~1/FX of Screener? (INFY/WIT etc. report in USD.)
    rev_scr = _series_for(pnl.assign(freq="A"), "revenue", "screener") * SCREENER_MONEY_SCALE
    rev_yf = _series_for(inc, "revenue", "yf")
    common = rev_yf.index.intersection(rev_scr.index)
    if len(common) >= 2:
        ratios = [float(rev_scr[y]) / float(rev_yf[y]) for y in common
                  if rev_yf[y] and abs(float(rev_yf[y])) > 0]
        if ratios:
            med = float(np.median(ratios))
            spread = float(np.std(ratios)) / med if med else 1.0
            if 55 <= med <= 100 and spread < 0.15:
                out.append(Finding("xcheck_fundamentals", sym, "currency_mismatch", "ERROR",
                                   f"yfinance fundamentals look USD-denominated: Screener/yfinance "
                                   f"revenue ratio ~{med:.0f}x across {len(ratios)} years "
                                   f"(fix: these need x{med:.0f} or a different source)",
                                   round(med, 1)))
                return out  # every concept mismatch below would just be this

    for concept in concepts:
        a = _series_for(inc, concept, "yf")
        b = _series_for(pnl.assign(freq="A"), concept, "screener") * SCREENER_MONEY_SCALE
        common = a.index.intersection(b.index)
        if len(common) < 2:
            continue
        diffs = {y: _rel_diff(float(a[y]), float(b[y])) for y in common}
        worst_y = max(diffs, key=diffs.get)
        worst = diffs[worst_y]
        sev = "ERROR" if worst > 0.30 else "WARN" if worst > 0.10 else None
        if sev:
            out.append(Finding("xcheck_fundamentals", sym, f"{concept}_mismatch", sev,
                               f"yfinance vs Screener {concept} differ {worst:.0%} in FY{worst_y} "
                               f"(yf={float(a[worst_y]):,.0f}, scr={float(b[worst_y]):,.0f})",
                               round(worst, 3)))
    return out


def check_xcheck_price(cfg: PipelineConfig, scr: dict[str, pd.DataFrame], sym: str) -> list[Finding]:
    price = scr.get("price", pd.DataFrame())
    part = cfg.ohlcv_dir("1day") / f"symbol={sym}"
    if price.empty or not part.exists():
        return []
    prow = price[price["line_item"].astype(str).str.lower().str.contains("price")]
    if prow.empty:
        return []
    s_prices = prow.groupby("period_end")["value"].last().sort_index()

    day = read_parquet_dir(part)
    if day.empty:
        return []
    day = day.assign(d=pd.to_datetime(day["ts_ist"]).dt.tz_localize(None)).sort_values("d")
    dser = day.set_index("d")["close"].astype(float)

    rels = []
    for d, sp in s_prices.items():
        if sp is None or sp <= 0:
            continue
        idx = dser.index.searchsorted(pd.Timestamp(d))
        for cand in (idx, idx - 1):
            if 0 <= cand < len(dser) and abs((dser.index[cand] - pd.Timestamp(d)).days) <= 6:
                rels.append(_rel_diff(float(dser.iloc[cand]), float(sp)))
                break
    if len(rels) < 5:
        return []
    med = float(np.median(rels))
    sev = "ERROR" if med > 0.25 else "WARN" if med > 0.05 else None
    if sev:
        return [Finding("xcheck_price", sym, "price_basis_mismatch", sev,
                        f"Upstox vs Screener monthly close differ (median {med:.1%}, "
                        f"n={len(rels)}) - likely a split/adjustment mismatch", round(med, 3))]
    return []


def check_corp_action_coverage(scr: dict[str, pd.DataFrame], corp: pd.DataFrame, sym: str) -> list[Finding]:
    """Flag a Screener-implied bonus/split with no matching action in our data.

    Uses the same detection (share-count jump corroborated by the bonus flag, or a
    face-value drop) and the same wide dedup tolerance as `corp-actions-screener`, so
    this check only ever flags what that ingest job would *not* already have merged in
    - run it and this warning should already be gone for most symbols.
    """
    from .sources.screener_actions import derive_actions, merge_actions

    bs = scr.get("balance_sheet", pd.DataFrame())
    if bs.empty:
        return []
    derived = derive_actions(bs, sym)
    if derived.empty:
        return []
    still_missing = merge_actions(corp, derived)
    # merge_actions returns corp's rows plus any derived rows that weren't already
    # covered; isolate just those survivors (source == screener_derived and not in corp).
    if corp is None or corp.empty:
        new_rows = derived
    else:
        new_rows = still_missing[
            (still_missing.get("source") == "screener_derived")
            & (~still_missing["ex_date"].isin(pd.to_datetime(corp["ex_date"])))
        ]
    if new_rows.empty:
        return []
    years = sorted({pd.Timestamp(d).year for d in new_rows["ex_date"]})
    return [Finding("xcheck_corp_actions", sym, "possible_missing_action", "WARN",
                    f"Screener shows a bonus / face-value split around FY "
                    f"{', '.join(map(str, years))} with no matching action in our data")]


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------
def _load_screener(cfg: PipelineConfig, sym: str) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    base = cfg.screener_parquet_dir
    if not base.exists():
        return out
    for section in ("pnl", "quarters", "balance_sheet", "cash_flow", "price", "derived"):
        p = base / section / f"{sym}.parquet"
        if p.exists():
            out[section] = _safe_read(p)
    return out


def run(
    cfg: PipelineConfig,
    *,
    symbols: list[str] | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    cfg.ensure_dirs()
    uni = load_universe(cfg)
    if symbols:
        uni = uni[uni["symbol"].isin({s.upper() for s in symbols})]
    if limit:
        uni = uni.head(limit)

    has_screener = cfg.screener_parquet_dir.exists() and any(
        cfg.screener_parquet_dir.rglob("*.parquet")
    )
    today = cfg.end
    findings: list[Finding] = []

    findings += check_coverage(cfg, uni, has_screener)

    yf_income_root = cfg.fundamentals_dir / "income_stmt"
    yf_balance_root = cfg.fundamentals_dir / "balance_sheet"
    industry_of = dict(zip(uni["symbol"], uni.get("industry", pd.Series(dtype=str))))

    for i, sym in enumerate(uni["symbol"], 1):
        corp = _safe_read(cfg.corp_actions_dir / f"{sym}.parquet")
        findings += check_prices(cfg, sym, corp, today)

        yf_all = pd.concat(
            [_safe_read(yf_income_root / f"{sym}.parquet"),
             _safe_read(yf_balance_root / f"{sym}.parquet")],
            ignore_index=True,
        )
        if not yf_all.empty:
            findings += check_fundamentals_internal(yf_all, sym)

        scr = _load_screener(cfg, sym)
        if scr:
            findings += check_screener_internal(scr, sym)
            findings += check_xcheck_fundamentals(yf_all, scr, sym, industry_of.get(sym, ""))
            findings += check_xcheck_price(cfg, scr, sym)
            findings += check_corp_action_coverage(scr, corp, sym)

        if i % 100 == 0:
            LOG.info("sanity: %d/%d symbols", i, len(uni))

    report = pd.DataFrame([asdict(f) for f in findings])
    _write_report(cfg, report, len(uni), has_screener)
    return report


def _write_report(cfg: PipelineConfig, report: pd.DataFrame, n_symbols: int, has_screener: bool) -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pq = cfg.report_dir / f"sanity_{stamp}.parquet"
    md = cfg.report_dir / f"sanity_{stamp}.md"

    order = {"ERROR": 0, "WARN": 1, "INFO": 2}
    if not report.empty:
        report = report.sort_values(
            by=["severity", "dataset", "symbol"],
            key=lambda c: c.map(order) if c.name == "severity" else c,
        ).reset_index(drop=True)
        report.to_parquet(pq, index=False)

    counts = report["severity"].value_counts().to_dict() if not report.empty else {}
    lines = [
        f"# Data sanity report - {datetime.now():%Y-%m-%d %H:%M}",
        "",
        f"- symbols checked: **{n_symbols}**",
        f"- Screener cross-check: **{'on' if has_screener else 'off (no exports found)'}**",
        f"- findings: **{len(report)}**  "
        f"(ERROR {counts.get('ERROR', 0)}, WARN {counts.get('WARN', 0)}, INFO {counts.get('INFO', 0)})",
        "",
    ]
    if not report.empty:
        by_check = (
            report.groupby(["severity", "dataset", "check"])
            .size().reset_index(name="n")
            .sort_values(["severity", "n"], ascending=[True, False],
                         key=lambda c: c.map(order) if c.name == "severity" else c)
        )
        lines += ["## By check", "", "| sev | dataset | check | count |", "|---|---|---|---|"]
        for _, r in by_check.iterrows():
            lines.append(f"| {r['severity']} | {r['dataset']} | {r['check']} | {r['n']} |")
        lines += ["", "## ERROR + WARN detail (first 200)", "",
                  "| sev | symbol | check | detail |", "|---|---|---|---|"]
        for _, r in report[report["severity"] != "INFO"].head(200).iterrows():
            lines.append(f"| {r['severity']} | {r['symbol']} | {r['check']} | {r['detail']} |")
    md.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines[:8]))
    print(f"\nfull report: {md}")
    if not report.empty:
        print(f"parquet:     {pq}")
