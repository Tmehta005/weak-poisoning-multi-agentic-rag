"""List / load ``AttackArtifact`` instances from ``data/attacks/``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException

from webapp.backend.schemas import ArtifactDetail, ArtifactSummary

router = APIRouter(tags=["artifacts"])

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ATTACKS = _REPO_ROOT / "data" / "attacks"


def _load_artifact_json(attack_dir: Path) -> dict:
    candidate = attack_dir / "artifact.json"
    if not candidate.exists():
        raise FileNotFoundError(f"artifact.json not found under {attack_dir}")
    with open(candidate, "r") as f:
        return json.load(f)


def _summary_from_dict(d: dict, path: Path) -> ArtifactSummary:
    loss_history = d.get("loss_history") or []
    return ArtifactSummary(
        attack_id=d.get("attack_id", path.parent.name),
        trigger=d.get("trigger", ""),
        target_claim=d.get("target_claim", ""),
        poison_doc_id=d.get("poison_doc_id", ""),
        harmful_match_phrases=list(d.get("harmful_match_phrases") or []),
        encoder_model=d.get("encoder_model", ""),
        num_adv_passage_tokens=int(d.get("num_adv_passage_tokens", 0)),
        target_query_ids=list(d.get("target_query_ids") or []),
        final_loss=float(loss_history[-1]) if loss_history else None,
        path=str(path.relative_to(_REPO_ROOT)),
    )


@router.get("/artifacts", response_model=List[ArtifactSummary])
def list_artifacts() -> List[ArtifactSummary]:
    if not _ATTACKS.exists():
        return []
    out: List[ArtifactSummary] = []
    for entry in sorted(_ATTACKS.iterdir()):
        if not entry.is_dir() or entry.name.startswith("_"):
            continue
        artifact_path = entry / "artifact.json"
        if not artifact_path.exists():
            continue
        try:
            d = _load_artifact_json(entry)
        except Exception:
            continue
        out.append(_summary_from_dict(d, artifact_path))
    return out


@router.get("/artifacts/{attack_id}", response_model=ArtifactDetail)
def get_artifact(attack_id: str) -> ArtifactDetail:
    entry = _ATTACKS / attack_id
    artifact_path = entry / "artifact.json"
    if not artifact_path.exists():
        raise HTTPException(status_code=404, detail=f"artifact not found: {attack_id}")
    d = _load_artifact_json(entry)
    summary = _summary_from_dict(d, artifact_path)
    return ArtifactDetail(
        **summary.model_dump(),
        poison_doc_text=d.get("poison_doc_text", ""),
        loss_history=[float(x) for x in (d.get("loss_history") or [])],
        token_ids=[int(x) for x in (d.get("token_ids") or [])],
    )
