from __future__ import annotations

from dataclasses import asdict
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .storage import atomic_parquet, price_key, read_table, write_artifact


def research_overview(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 2 or payload.get("kind") != "research":
        raise ValueError("Unsupported research artifact.")
    return {method: {"metrics": result["metrics"],
                     "equity_curve": read_table(path.parent, result["equity_curve"], columns=["equity"])}
            for method, result in payload["results"].items()}


def save_research(root: Path, prices, config, backtest, results) -> Path:
    key = "research_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:12]
    digest = price_key(prices)
    price_path = root / "prices" / f"{digest}.parquet"
    if not price_path.exists():
        atomic_parquet(prices, price_path)
    payload = {"schema_version": 2, "kind": "research", "id": key,
               "saved_at": datetime.now(timezone.utc), "allocation_config": asdict(config),
               "backtest_config": asdict(backtest), "price_hash": digest,
               "prices": {"$table": f"prices/{digest}.parquet"}, "results": results}
    return write_artifact(root, key, payload)
