"""
Smoke tests for the Retriever (src/retriever.py).

Uses an in-memory LlamaIndex index built from small synthetic documents
so the tests run without a real corpus or embedding API.
"""

import pytest
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.embeddings import MockEmbedding

from src.retriever import Retriever
from src.schemas import RetrievedDoc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SYNTHETIC_DOCS = [
    Document(text="The capital of France is Paris.", doc_id="d1"),
    Document(text="Berlin is the capital of Germany.", doc_id="d2"),
    Document(text="Rome is the capital of Italy.", doc_id="d3"),
    Document(text="Madrid is the capital of Spain.", doc_id="d4"),
    Document(text="Tokyo is the capital of Japan.", doc_id="d5"),
]


@pytest.fixture(scope="module")
def retriever():
    """Build a small in-memory index with a MockEmbedding for fast tests."""
    embed_model = MockEmbedding(embed_dim=8)
    index = VectorStoreIndex.from_documents(
        SYNTHETIC_DOCS,
        embed_model=embed_model,
        show_progress=False,
    )
    return Retriever(index, top_k=3)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_retrieve_returns_list(retriever):
    results = retriever.retrieve("capital of France")
    assert isinstance(results, list)


def test_retrieve_returns_retrieved_doc_type(retriever):
    results = retriever.retrieve("capital of France")
    for r in results:
        assert isinstance(r, RetrievedDoc)


def test_retrieve_respects_top_k(retriever):
    results = retriever.retrieve("capital", top_k=2)
    assert len(results) <= 2


def test_retrieve_doc_has_required_fields(retriever):
    results = retriever.retrieve("capital of Germany")
    assert len(results) > 0
    r = results[0]
    assert isinstance(r.doc_id, str) and r.doc_id
    assert isinstance(r.text, str) and r.text
    assert isinstance(r.score, float)


def test_retrieve_score_non_negative(retriever):
    results = retriever.retrieve("capital of Spain")
    for r in results:
        assert r.score >= 0.0
