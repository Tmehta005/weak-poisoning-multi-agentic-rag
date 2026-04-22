"""Corpus discovery: list sub-directories under data/ that look like corpora."""

from __future__ import annotations

from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException

from webapp.backend.schemas import Corpus

router = APIRouter(tags=["corpora"])

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DATA = _REPO_ROOT / "data"

_DOC_EXTS = {".txt", ".pdf", ".md", ".docx"}
_INDEX_MARKERS = {"docstore.json", "index_store.json"}


def _count_docs(path: Path) -> tuple[int, List[str]]:
    count = 0
    exts: set[str] = set()
    try:
        for p in path.rglob("*"):
            if p.is_file() and p.suffix.lower() in _DOC_EXTS:
                count += 1
                exts.add(p.suffix.lower())
    except (PermissionError, FileNotFoundError):
        pass
    return count, sorted(exts)


def _is_index_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any((path / m).exists() for m in _INDEX_MARKERS)


def _suggest_persist_dir(name: str) -> str:
    if name.startswith("corpus_"):
        return f"data/index_{name[len('corpus_'):]}"
    if name == "corpus":
        return "data/index"
    return f"data/index_{name}"


@router.get("/corpora", response_model=List[Corpus])
def list_corpora() -> List[Corpus]:
    if not _DATA.exists():
        return []
    results: List[Corpus] = []
    for entry in sorted(_DATA.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name in {"attacks", "queries", "logs", "tmp"}:
            continue
        if entry.name.startswith("index"):
            continue
        if _is_index_dir(entry):
            continue
        doc_count, exts = _count_docs(entry)
        if doc_count == 0:
            continue
        persist = _suggest_persist_dir(entry.name)
        persist_abs = _REPO_ROOT / persist
        results.append(
            Corpus(
                name=entry.name,
                data_dir=f"data/{entry.name}",
                suggested_persist_dir=persist,
                doc_count=doc_count,
                has_index=_is_index_dir(persist_abs),
                file_types=exts,
            )
        )
    return results


@router.get("/corpora/check")
def check_corpus(data_dir: str) -> dict:
    """
    Inspect an arbitrary path (relative to repo root or absolute) so the UI
    can validate the free-text override before submitting an ingest job.
    """
    p = Path(data_dir)
    if not p.is_absolute():
        p = _REPO_ROOT / data_dir
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"path not found: {data_dir}")
    if not p.is_dir():
        raise HTTPException(status_code=400, detail=f"not a directory: {data_dir}")
    doc_count, exts = _count_docs(p)
    return {
        "data_dir": data_dir,
        "doc_count": doc_count,
        "file_types": exts,
    }
