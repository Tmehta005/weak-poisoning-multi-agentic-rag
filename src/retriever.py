"""
Retrieval wrapper around a LlamaIndex VectorStoreIndex.

Returns typed RetrievedDoc objects so the rest of the pipeline
never touches LlamaIndex internals directly.
"""

from __future__ import annotations

from typing import List, Optional

from llama_index.core import VectorStoreIndex

from src.schemas import RetrievedDoc


class Retriever:
    """
    Thin wrapper over a LlamaIndex VectorStoreIndex.

    Exposes a single retrieve() method that returns List[RetrievedDoc].
    The poisoned retrieval path (for attacked subagents) is handled in
    src/attacks/ by injecting D_p into the index before calling retrieve().
    """

    def __init__(self, index: VectorStoreIndex, top_k: int = 5):
        self.index = index
        self.top_k = top_k
        self._retriever = index.as_retriever(similarity_top_k=top_k)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievedDoc]:
        """
        Retrieve the top-k most relevant documents for query.

        Args:
            query: The query string (may include trigger t for attacked agents).
            top_k: Override the instance-level top_k for this call.

        Returns:
            List of RetrievedDoc sorted by descending score.
        """
        if top_k is not None and top_k != self.top_k:
            retriever = self.index.as_retriever(similarity_top_k=top_k)
        else:
            retriever = self._retriever

        nodes = retriever.retrieve(query)

        results: List[RetrievedDoc] = []
        for node in nodes:
            doc_id = node.node.node_id
            results.append(
                RetrievedDoc(
                    doc_id=doc_id,
                    text=node.node.get_content(),
                    score=float(node.score) if node.score is not None else 0.0,
                )
            )
        return results
