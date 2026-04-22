"""Query file discovery + parsing for dropdowns."""

from __future__ import annotations

from pathlib import Path
from typing import List

import yaml
from fastapi import APIRouter, HTTPException

from webapp.backend.schemas import QueryEntry, QueryFile

router = APIRouter(tags=["queries"])

_REPO_ROOT = Path(__file__).resolve().parents[3]
_QUERIES_DIR = _REPO_ROOT / "data" / "queries"


def _resolve(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = _REPO_ROOT / path
    return p


@router.get("/query-files", response_model=List[str])
def list_query_files() -> List[str]:
    if not _QUERIES_DIR.exists():
        return []
    out: List[str] = []
    for p in sorted(_QUERIES_DIR.glob("*.y*ml")):
        out.append(str(p.relative_to(_REPO_ROOT)))
    return out


@router.get("/queries", response_model=QueryFile)
def load_query_file(path: str) -> QueryFile:
    p = _resolve(path)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"query file not found: {path}")
    with open(p, "r") as f:
        data = yaml.safe_load(f) or []
    if not isinstance(data, list):
        raise HTTPException(status_code=400, detail="query file must be a top-level list")
    entries: List[QueryEntry] = []
    for q in data:
        attack = q.get("attack") if isinstance(q, dict) else None
        artifact_path = None
        if isinstance(attack, dict):
            artifact_path = attack.get("artifact_path")
        entries.append(
            QueryEntry(
                query_id=str(q.get("query_id", "")),
                query=str(q.get("query", "")),
                ground_truth_answer=q.get("ground_truth_answer"),
                category=q.get("category"),
                has_attack=bool(attack),
                attack_artifact_path=artifact_path,
            )
        )
    return QueryFile(path=path, queries=entries)
