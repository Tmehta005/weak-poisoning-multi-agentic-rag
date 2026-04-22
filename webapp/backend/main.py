"""
FastAPI entry point for the research webapp.

Run from the repo root so subprocesses inherit the same venv:

    uvicorn webapp.backend.main:app --reload --port 8000
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from webapp.backend.api import (
    artifacts,
    corpora,
    defaults,
    experiments,
    ingest,
    jobs,
    queries,
    runs,
    trigger,
)

app = FastAPI(title="RAG Poisoning Experiment Console", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(defaults.router, prefix="/api")
app.include_router(corpora.router, prefix="/api")
app.include_router(queries.router, prefix="/api")
app.include_router(artifacts.router, prefix="/api")
app.include_router(ingest.router, prefix="/api")
app.include_router(trigger.router, prefix="/api")
app.include_router(experiments.router, prefix="/api")
app.include_router(jobs.router, prefix="/api")
app.include_router(runs.router, prefix="/api")


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}
