"""
Baseline RAG pipeline.

Single-call retrieval + generation with Langfuse tracing (v3 decorator API).
This is the clean baseline: no agents, no orchestrator, no attack.

Every run emits a RunLog with attack_condition="clean".
"""

from __future__ import annotations

import os
import uuid
from typing import Optional

from llama_index.core import VectorStoreIndex

from src.retriever import Retriever
from src.schemas import OrchestratorOutput, RunLog, SubagentOutput
from src.logging_utils import emit_run_log

# Langfuse v3 uses the @observe decorator; fall back silently if not installed.
try:
    from langfuse.decorators import observe, langfuse_context
    _langfuse_available = True
except ImportError:
    _langfuse_available = False
    # Provide no-op stubs so the rest of the code is unchanged.
    def observe(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator
    langfuse_context = None

# OpenAI client for generation.
try:
    from openai import OpenAI
    _openai_available = True
except ImportError:
    _openai_available = False


def _load_prompt_template(path: str = "prompts/baseline_rag.txt") -> str:
    with open(path, "r") as f:
        return f.read()


class BaselineRAG:
    """
    Retrieves top-k documents then generates an answer with an LLM.

    Traces each run to Langfuse if LANGFUSE_PUBLIC_KEY and
    LANGFUSE_SECRET_KEY are set in the environment.
    Logs each run to results/runs.jsonl via emit_run_log.
    """

    def __init__(
        self,
        index: VectorStoreIndex,
        top_k: int = 5,
        model: str = "gpt-4o-mini",
        output_dir: str = "results",
        prompt_path: str = "prompts/baseline_rag.txt",
    ):
        self.retriever = Retriever(index, top_k=top_k)
        self.model = model
        self.output_dir = output_dir
        self.prompt_template = _load_prompt_template(prompt_path)
        self._tracing_enabled = (
            _langfuse_available and bool(os.getenv("LANGFUSE_PUBLIC_KEY"))
        )

        self._openai: Optional[object] = None
        if _openai_available and os.getenv("OPENAI_API_KEY"):
            self._openai = OpenAI()

    def run(
        self,
        query: str,
        query_id: Optional[str] = None,
        attack_condition: str = "clean",
        trigger: Optional[str] = None,
    ) -> RunLog:
        """
        Run baseline RAG for a single query.

        Args:
            query: The user question.
            query_id: Optional stable ID for this query. Auto-generated if None.
            attack_condition: Label for logging. Should be "clean" for this baseline.
            trigger: Trigger string, if any (logged but not used in baseline).

        Returns:
            A completed RunLog instance.
        """
        if query_id is None:
            query_id = str(uuid.uuid4())

        return self._run_traced(
            query=query,
            query_id=query_id,
            attack_condition=attack_condition,
            trigger=trigger,
        )

    @observe(name="baseline_rag")
    def _run_traced(
        self,
        query: str,
        query_id: str,
        attack_condition: str,
        trigger: Optional[str],
    ) -> RunLog:
        if self._tracing_enabled and langfuse_context is not None:
            langfuse_context.update_current_trace(
                name="baseline_rag",
                input={"query": query, "query_id": query_id},
                metadata={"attack_condition": attack_condition, "trigger": trigger},
            )

        # --- Retrieval ---
        docs = self._retrieve(query)
        doc_ids = [d.doc_id for d in docs]

        # --- Generation ---
        doc_texts = "\n\n".join(
            f"[{i+1}] (id={d.doc_id})\n{d.text}" for i, d in enumerate(docs)
        )
        prompt = self.prompt_template.format(documents=doc_texts, question=query)
        answer = self._generate(prompt)

        # --- Build typed outputs ---
        agent_id = "baseline"
        subagent_out = SubagentOutput(
            agent_id=agent_id,
            answer=answer,
            citations=doc_ids,
            confidence=1.0,
            rationale="Baseline RAG: no rationale step.",
            poison_retrieved=False,
            retrieved_doc_ids=doc_ids,
        )
        orchestrator_out = OrchestratorOutput(
            final_answer=answer,
            selected_evidence=doc_ids,
            winning_subagents=[agent_id],
            final_confidence=1.0,
            harmful_action_flag=False,
        )
        run_log = RunLog(
            query_id=query_id,
            attack_condition=attack_condition,
            trigger=trigger,
            retrieved_doc_ids_per_agent={agent_id: doc_ids},
            poison_retrieved=False,
            agent_responses={agent_id: subagent_out},
            final_decision=orchestrator_out,
            metrics={},
        )

        if self._tracing_enabled and langfuse_context is not None:
            langfuse_context.update_current_observation(
                output={"answer": answer, "doc_ids": doc_ids}
            )

        emit_run_log(run_log, output_dir=self.output_dir)
        return run_log

    @observe(name="retrieval")
    def _retrieve(self, query: str):
        return self.retriever.retrieve(query)

    @observe(name="generation")
    def _generate(self, prompt: str) -> str:
        """Call the LLM. Falls back to a stub if no API key is configured."""
        if self._openai is None:
            return "[NO_API_KEY] Retrieved docs available but generation skipped."

        response = self._openai.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()
