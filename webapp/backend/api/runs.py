"""Read runs from ``results/runs.jsonl`` for the history page and result panel."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException

router = APIRouter(tags=["runs"])

_REPO_ROOT = Path(__file__).resolve().parents[3]
_RUNS_JSONL = _REPO_ROOT / "results" / "runs.jsonl"


def _read_all() -> List[Dict[str, Any]]:
    if not _RUNS_JSONL.exists():
        return []
    out: List[Dict[str, Any]] = []
    with open(_RUNS_JSONL, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _summarize(rec: Dict[str, Any]) -> Dict[str, Any]:
    fd = rec.get("final_decision") or {}
    return {
        "query_id": rec.get("query_id"),
        "attack_condition": rec.get("attack_condition"),
        "trigger": rec.get("trigger"),
        "poison_retrieved": bool(rec.get("poison_retrieved")),
        "harmful_action_flag": bool(fd.get("harmful_action_flag")),
        "final_confidence": fd.get("final_confidence"),
        "final_answer": (fd.get("final_answer") or "")[:200],
        "logged_at": rec.get("_logged_at"),
        "has_debate": bool(rec.get("debate_transcript")),
    }


@router.get("/runs")
def list_runs(
    limit: int = 50,
    query_id: Optional[str] = None,
    attack_condition: Optional[str] = None,
) -> List[Dict[str, Any]]:
    recs = _read_all()
    if query_id:
        recs = [r for r in recs if r.get("query_id") == query_id]
    if attack_condition:
        recs = [r for r in recs if r.get("attack_condition") == attack_condition]
    recs.sort(key=lambda r: r.get("_logged_at", ""), reverse=True)
    return [_summarize(r) for r in recs[:limit]]


@router.get("/runs/latest")
def get_latest_runs(
    limit: int = 3,
    since: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return the most recent N full RunLog records (after ``since`` timestamp)."""
    recs = _read_all()
    if since:
        recs = [r for r in recs if r.get("_logged_at", "") > since]
    recs.sort(key=lambda r: r.get("_logged_at", ""), reverse=True)
    return recs[:limit]


@router.get("/runs/by-query/{query_id}")
def get_run_detail(query_id: str, attack_condition: Optional[str] = None) -> Dict[str, Any]:
    recs = _read_all()
    if attack_condition:
        recs = [
            r
            for r in recs
            if r.get("query_id") == query_id and r.get("attack_condition") == attack_condition
        ]
    else:
        recs = [r for r in recs if r.get("query_id") == query_id]
    if not recs:
        raise HTTPException(status_code=404, detail=f"no run found for {query_id}")
    recs.sort(key=lambda r: r.get("_logged_at", ""), reverse=True)
    return recs[0]
