"""POST /api/experiments: kick off an orchestrator or debate experiment."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import yaml
from fastapi import APIRouter, HTTPException

from webapp.backend.jobs.manager import get_manager
from webapp.backend.schemas import ExperimentRequest, JobSummary

router = APIRouter(tags=["experiments"])

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIGS = _REPO_ROOT / "configs"
_TMP_DIR = _REPO_ROOT / "webapp" / "data" / "tmp"


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _write_tmp_yaml(prefix: str, payload: dict) -> str:
    _TMP_DIR.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(
        "w", suffix=f"_{prefix}.yaml", delete=False, dir=str(_TMP_DIR)
    )
    try:
        yaml.safe_dump(payload, tmp)
        return tmp.name
    finally:
        tmp.close()


@router.post("/experiments", response_model=JobSummary)
def submit_experiment(req: ExperimentRequest) -> JobSummary:
    mgr = get_manager()
    if mgr.has_running("experiment"):
        raise HTTPException(
            status_code=409, detail="an experiment is already running"
        )

    system_cfg_name = "system_orchestrator.yaml" if req.system == "orchestrator" else "system_debate.yaml"
    base_system = _load_yaml(_CONFIGS / system_cfg_name)
    if req.model is not None:
        base_system["model"] = req.model
    if req.num_subagents is not None:
        base_system["num_subagents"] = req.num_subagents
    if req.system == "orchestrator":
        if req.top_k is not None:
            base_system["top_k"] = req.top_k
    else:
        if req.top_k is not None:
            base_system["subagent_top_k"] = req.top_k
        if req.max_rounds is not None:
            base_system["max_rounds"] = req.max_rounds
        if req.stable_for is not None:
            base_system["stable_for"] = req.stable_for
    system_cfg_path = _write_tmp_yaml(f"{req.system}_{req.mode}", base_system)

    ingestion_cfg_path = str(
        _CONFIGS / ("corpus_cybersec.yaml" if req.corpus == "cybersec" else "ingestion.yaml")
    )

    runner = {
        ("orchestrator", "clean"): "webapp.backend.runners.run_clean_orch",
        ("orchestrator", "attack"): "src.experiments.run_attack_orch",
        ("debate", "clean"): "webapp.backend.runners.run_clean_debate",
        ("debate", "attack"): "src.experiments.run_attack_debate",
    }[(req.system, req.mode)]

    cmd = [sys.executable, "-m", runner]

    if req.mode == "attack":
        base_attack = _load_yaml(_CONFIGS / "attack_main_injection.yaml")
        base_attack["threat_model"] = req.threat_model
        base_attack["poisoned_subagent_ids"] = req.poisoned_subagent_ids
        if req.attack_id:
            base_attack.setdefault("attack_id", req.attack_id)
            base_attack["artifact_path"] = f"data/attacks/{req.attack_id}/artifact.json"
        attack_cfg_path = _write_tmp_yaml(f"{req.attack_id or 'attack'}_main", base_attack)

        if req.system == "orchestrator":
            cmd += [
                "--query-file", req.query_file,
                "--system-config", system_cfg_path,
                "--attack-config", attack_cfg_path,
                "--ingestion-config", ingestion_cfg_path,
                "--threat-model", req.threat_model,
            ]
            for sid in req.poisoned_subagent_ids:
                cmd += ["--poisoned-subagent-id", sid]
        else:
            cmd += [
                "--query-file", req.query_file,
                "--debate-config", system_cfg_path,
                "--attack-config", attack_cfg_path,
                "--ingestion-config", ingestion_cfg_path,
                "--threat-model", req.threat_model,
            ]
            for sid in req.poisoned_subagent_ids:
                cmd += ["--poisoned-subagent-id", sid]
    else:
        if req.system == "orchestrator":
            cmd += [
                "--query-file", req.query_file,
                "--system-config", system_cfg_path,
                "--ingestion-config", ingestion_cfg_path,
                "--corpus", req.corpus,
            ]
        else:
            cmd += [
                "--query-file", req.query_file,
                "--debate-config", system_cfg_path,
                "--ingestion-config", ingestion_cfg_path,
                "--corpus", req.corpus,
            ]

    params = req.model_dump()
    params["_runner"] = runner
    job = mgr.submit("experiment", cmd, params)
    return JobSummary(
        id=job.id,
        kind=job.kind,
        status=job.status,
        created_at=job.created_at,
        started_at=job.started_at,
        ended_at=job.ended_at,
        exit_code=job.exit_code,
        params=job.params,
        result=job.result,
        error=job.error,
    )
