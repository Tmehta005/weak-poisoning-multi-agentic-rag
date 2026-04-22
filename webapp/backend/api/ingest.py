"""POST /api/ingest: build a LlamaIndex vector store from a corpus directory."""

from __future__ import annotations

import sys

from fastapi import APIRouter, HTTPException

from webapp.backend.jobs.manager import get_manager
from webapp.backend.schemas import IngestRequest, JobSummary

router = APIRouter(tags=["ingest"])


@router.post("/ingest", response_model=JobSummary)
def submit_ingest(req: IngestRequest) -> JobSummary:
    mgr = get_manager()
    if mgr.has_running("ingest"):
        raise HTTPException(
            status_code=409, detail="an ingest job is already running"
        )

    cmd = [
        sys.executable,
        "-m",
        "webapp.backend.runners.ingest",
        "--data-dir",
        req.data_dir,
        "--persist-dir",
        req.persist_dir,
        "--chunk-size",
        str(req.chunk_size),
        "--chunk-overlap",
        str(req.chunk_overlap),
        "--embed-model",
        req.embed_model,
        "--similarity-top-k",
        str(req.similarity_top_k),
        "--variant",
        req.variant,
    ]
    if req.rebuild:
        cmd.append("--rebuild")

    job = mgr.submit("ingest", cmd, req.model_dump())
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
