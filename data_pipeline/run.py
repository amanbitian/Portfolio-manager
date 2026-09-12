"""CLI entry point for the data pipeline.

    py -3 -m data_pipeline.run universe
    py -3 -m data_pipeline.run market --interval 1day
    py -3 -m data_pipeline.run market --interval 1min --limit 10
    py -3 -m data_pipeline.run corp-actions
    py -3 -m data_pipeline.run fundamentals
    py -3 -m data_pipeline.run macro
    py -3 -m data_pipeline.run news --limit 10
    py -3 -m data_pipeline.run status
    py -3 -m data_pipeline.run all
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime

from .checkpoint import Manifest
from .config import PipelineConfig, load_config
from .io import dir_size_mb


def _setup_logging(cfg: PipelineConfig, name: str) -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)  # flush progress in background runs
    except (AttributeError, ValueError):
        pass
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    logfile = cfg.log_dir / f"{name}-{datetime.now():%Y%m%d}.log"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    handlers.append(logging.FileHandler(logfile, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )
    logging.getLogger("yfinance").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def _split_symbols(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [s.strip().upper() for s in value.split(",") if s.strip()]


def cmd_status(cfg: PipelineConfig) -> None:
    print(f"data root: {cfg.data_root}")
    print(f"{'dataset':<26}{'done':>8}{'failed':>8}{'rows':>14}   size")
    print("-" * 74)
    manifests = sorted(cfg.manifest_dir.glob("*.json")) if cfg.manifest_dir.exists() else []
    for mf in manifests:
        s = Manifest(mf).summary()
        print(f"{mf.stem:<26}{s['done']:>8}{s['failed']:>8}{s['rows']:>14,}")
    print("-" * 74)
    # article counts for news datasets that track state in the output, not a manifest
    import pandas as _pd

    def _count(files: list) -> int:
        total = 0
        for p in files:
            try:
                total += len(_pd.read_parquet(p, columns=["url"]))
            except Exception:  # noqa: BLE001
                pass
        return total

    cc = cfg.news_dir / "commoncrawl"
    for label, files in (
        ("news/rss articles", list((cfg.news_dir / "rss").rglob("*.parquet"))),
        ("news/cc catalogued", list((cc / "_catalog").glob("*.parquet"))),
        ("news/cc fetched", list(cc.rglob("part-*.parquet"))),
    ):
        if files:
            print(f"{label:<26}{'':>8}{'':>8}{_count(files):>14,}")
    print("-" * 74)
    for label, path in (
        ("reference", cfg.reference_dir),
        ("market/ohlcv_1day", cfg.ohlcv_dir("1day")),
        ("market/ohlcv_1min", cfg.ohlcv_dir("1min")),
        ("corporate_actions", cfg.corp_actions_dir),
        ("fundamentals", cfg.fundamentals_dir),
        ("macro", cfg.macro_dir),
        ("news", cfg.news_dir),
    ):
        print(f"{label:<40}{dir_size_mb(path):>10.1f} MB")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="data_pipeline.run", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("universe", help="build reference/universe.parquet")

    p_market = sub.add_parser("market", help="OHLCV via Upstox")
    p_market.add_argument("--interval", choices=["1day", "1min"], required=True)
    p_market.add_argument("--symbols", help="comma-separated subset")
    p_market.add_argument("--limit", type=int, help="first N symbols only")

    for name in ("corp-actions", "corp-actions-screener", "fundamentals", "screener", "sanity"):
        p = sub.add_parser(name)
        p.add_argument("--symbols")
        p.add_argument("--limit", type=int)

    p_scrdl = sub.add_parser(
        "screener-download", help="download screener.in Excel exports (needs SCREENER_SESSIONID)"
    )
    p_scrdl.add_argument("--symbols")
    p_scrdl.add_argument("--limit", type=int)
    p_scrdl.add_argument("--standalone", action="store_true",
                         help="standalone financials instead of consolidated")
    p_scrdl.add_argument("--force", action="store_true", help="re-download existing files")

    p_macro = sub.add_parser("macro", help="World Bank + FRED + RBI repo rate")
    p_macro.add_argument(
        "--force", action="store_true", help="re-fetch series already marked done"
    )

    p_news = sub.add_parser("news", help="news: GDELT (historical) and/or RSS (ongoing)")
    p_news.add_argument("--source", choices=["gdelt", "rss", "all"], default="all")
    p_news.add_argument("--symbols")
    p_news.add_argument("--limit", type=int)
    p_news.add_argument("--years", type=int, help="GDELT backfill window")

    p_cc = sub.add_parser("news-cc", help="historical news from Common Crawl (2017+)")
    p_cc.add_argument("--phase", choices=["catalog", "fetch", "both"], default="both")
    p_cc.add_argument("--since", type=int, default=2017, help="earliest crawl year")
    p_cc.add_argument("--domains", help="comma list: et,bs,mint,hbl,mc,fe,bt")
    p_cc.add_argument("--limit", type=int, help="max articles to fetch this run")
    p_cc.add_argument("--workers", type=int, default=6, help="parallel fetch workers")

    sub.add_parser("status", help="manifest + disk summary")
    sub.add_parser("all", help="universe -> 1day -> corp-actions -> fundamentals -> macro -> news pilot")

    args = parser.parse_args(argv)
    cfg = load_config()
    _setup_logging(cfg, args.cmd.replace("-", "_"))
    log = logging.getLogger("data_pipeline.run")

    if args.cmd == "universe":
        from .universe import build_universe

        build_universe(cfg)
    elif args.cmd == "market":
        from .ingest import market

        market.run(
            cfg,
            args.interval,
            symbols=_split_symbols(args.symbols),
            limit=args.limit,
        )
    elif args.cmd == "corp-actions":
        from .ingest import corp_actions

        corp_actions.run(cfg, symbols=_split_symbols(args.symbols), limit=args.limit)
    elif args.cmd == "corp-actions-screener":
        from .ingest import corp_actions_screener

        corp_actions_screener.run(cfg, symbols=_split_symbols(args.symbols), limit=args.limit)
    elif args.cmd == "fundamentals":
        from .ingest import fundamentals

        fundamentals.run(cfg, symbols=_split_symbols(args.symbols), limit=args.limit)
    elif args.cmd == "screener":
        from .ingest import screener

        screener.run(cfg, symbols=_split_symbols(args.symbols), limit=args.limit)
    elif args.cmd == "screener-download":
        from .tools import screener_download

        screener_download.run(
            cfg,
            symbols=_split_symbols(args.symbols),
            limit=args.limit,
            standalone=args.standalone,
            force=args.force,
        )
    elif args.cmd == "sanity":
        from . import sanity

        sanity.run(cfg, symbols=_split_symbols(args.symbols), limit=args.limit)
    elif args.cmd == "macro":
        from .ingest import macro

        macro.run(cfg, force=args.force)
    elif args.cmd == "news":
        if args.source in ("rss", "all"):
            from .ingest import news_rss

            news_rss.run(cfg)
        if args.source in ("gdelt", "all"):
            from .ingest import news

            news.run(
                cfg,
                symbols=_split_symbols(args.symbols),
                limit=args.limit,
                years=args.years,
            )
    elif args.cmd == "news-cc":
        from .ingest import news_cc

        domains = (
            [d.strip().lower() for d in args.domains.split(",") if d.strip()]
            if args.domains
            else None
        )
        news_cc.run(
            cfg,
            phase=args.phase,
            since=args.since,
            domains=domains,
            limit=args.limit,
            workers=args.workers,
        )
    elif args.cmd == "status":
        cmd_status(cfg)
    elif args.cmd == "all":
        from .ingest import corp_actions, fundamentals, macro, market, news
        from .universe import build_universe

        build_universe(cfg)
        market.run(cfg, "1day")
        corp_actions.run(cfg)
        fundamentals.run(cfg)
        macro.run(cfg)
        news.run(cfg, limit=10)
        log.info(
            "all: pilot done. Launch the full 1-minute backfill with:\n"
            "  py -3 -m data_pipeline.run market --interval 1min"
        )
        cmd_status(cfg)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
