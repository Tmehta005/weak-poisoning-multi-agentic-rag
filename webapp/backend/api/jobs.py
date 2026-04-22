"""Job listing, status, SSE log streaming, and cancellation."""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from webapp.backend.jobs.manager import get_manager
from webapp.backend.schemas import JobSummary

router = APIRouter(tags=["jobs"])


def _to_summary(job) -> JobSummary:
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


@router.get("/jobs", response_model=List[JobSummary])
def list_jobs(kind: Optional[str] = None, limit: int = 50) -> List[JobSummary]:
    mgr = get_manager()
    return [_to_summary(j) for j in mgr.list(kind=kind, limit=limit)]


@router.get("/jobs/{job_id}", response_model=JobSummary)
def get_job(job_id: str) -> JobSummary:
    mgr = get_manager()
    job = mgr.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return _to_summary(job)


@router.post("/jobs/{job_id}/cancel", response_model=JobSummary)
def cancel_job(job_id: str) -> JobSummary:
    mgr = get_manager()
    job = mgr.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    mgr.cancel(job_id)
    return _to_summary(mgr.get(job_id))  # type: ignore[arg-type]


@router.get("/jobs/{job_id}/stream")
def stream_job(job_id: str) -> StreamingResponse:
    mgr = get_manager()
    job = mgr.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")

    def gen():
        for line in mgr.tail(job_id):
            safe = line.rstrip("\n")
            yield f"data: {safe}\n\n"
        final = mgr.get(job_id)
        if final:
            yield f"event: status\ndata: {final.status}\n\n"
        yield "event: done\ndata: end\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
