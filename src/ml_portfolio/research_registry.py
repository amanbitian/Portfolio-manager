from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

# Every artifact kind that represents a piece of research, as opposed to
# operational state (a real paper account, a purchase portfolio, a
# transaction ledger) that happens to share the same schema_version=2
# artifact envelope (storage.write_artifact). Legacy proposal files predate
# the "kind" field entirely and default to "proposal" (see portfolio.py's
# saved_proposals), so that default is mirrored here.
EXPERIMENT_KINDS = frozenset({"proposal", "research", "research_report", "strategy_run", "strategy_sweep",
                              "robustness_report"})

VALID_STATUSES = ("unlabeled", "supported", "falsified", "inconclusive")

_UNSET = object()


def default_research_meta() -> dict:
    return {"research_question": "", "hypothesis": "", "status": "unlabeled",
            "conclusion": "", "parent_id": None}


def get_research_meta(path: Path) -> dict:
    """Read just the `research` metadata block from a saved artifact's
    manifest, without resolving its parquet tables (load_artifact does that
    and is unnecessary here)."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    meta = default_research_meta()
    meta.update(payload.get("research") or {})
    return meta


def set_research_meta(path: Path, *, research_question=_UNSET, hypothesis=_UNSET,
                      status=_UNSET, conclusion=_UNSET, parent_id=_UNSET) -> dict:
    """Attach or update the research metadata block on an already-saved
    artifact, in place. Every field defaults to leaving the existing value
    (or the schema default, for a first tag) untouched - callers pass only
    the fields they mean to change, including an explicit "" to clear a
    text field. Written atomically so a crash mid-write can never corrupt
    the manifest the way a bare `write_text` could.
    """
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    meta = default_research_meta()
    meta.update(payload.get("research") or {})
    updates = {"research_question": research_question, "hypothesis": hypothesis,
               "status": status, "conclusion": conclusion, "parent_id": parent_id}
    for key, value in updates.items():
        if value is not _UNSET:
            meta[key] = value
    if meta["status"] not in VALID_STATUSES:
        raise ValueError(f"status must be one of {VALID_STATUSES}, got {meta['status']!r}")
    payload["research"] = meta
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(temporary, path)
    return meta


def list_experiments(root: Path) -> pd.DataFrame:
    """Every saved research-kind artifact under `root`, newest first, with
    its research metadata flattened alongside the artifact envelope fields.
    Operational kinds (paper account, purchase portfolios, ledgers) are
    excluded; unreadable manifests are skipped rather than raising, since
    this powers a listing view, not a strict data-integrity check."""
    rows = []
    for path in root.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != 2:
                continue
            kind = payload.get("kind", "proposal")
            if kind not in EXPERIMENT_KINDS:
                continue
            meta = default_research_meta()
            meta.update(payload.get("research") or {})
            rows.append({
                "path": str(path), "id": payload.get("id", path.stem), "kind": kind,
                "name": payload.get("name", ""), "saved_at": payload["saved_at"]["$timestamp"],
                "research_question": meta["research_question"], "hypothesis": meta["hypothesis"],
                "status": meta["status"], "conclusion": meta["conclusion"], "parent_id": meta["parent_id"],
                "source_research_report_id": payload.get("source_research_report_id"),
            })
        except (ValueError, KeyError, TypeError, OSError):
            continue
    frame = pd.DataFrame(rows, columns=["path", "id", "kind", "name", "saved_at", "research_question",
                                        "hypothesis", "status", "conclusion", "parent_id",
                                        "source_research_report_id"])
    if frame.empty:
        return frame
    return frame.sort_values("saved_at", ascending=False, ignore_index=True)


def registry_summary(root: Path) -> dict:
    """Counts for the "12 hypotheses: 4 supported, 5 falsified, 3
    inconclusive" style overview a research registry view wants."""
    return summarize_experiments(list_experiments(root))


def summarize_experiments(experiments: pd.DataFrame) -> dict:
    """Same counts as `registry_summary`, from an already-loaded frame - lets
    a caller that already has one (e.g. after its own cached `list_experiments`
    call) avoid re-scanning `root` a second time in the same pass."""
    if experiments.empty:
        return {"total": 0, "by_status": {status: 0 for status in VALID_STATUSES}, "by_kind": {}}
    return {
        "total": int(len(experiments)),
        "by_status": {status: int((experiments.status == status).sum()) for status in VALID_STATUSES},
        "by_kind": experiments.kind.value_counts().to_dict(),
    }
