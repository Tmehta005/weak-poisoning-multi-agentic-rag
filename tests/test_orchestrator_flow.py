"""
Tests for the clean multi-agent orchestrator (Phase 2).

All LLM calls are stubbed so tests run without an OpenAI API key.
The index uses MockEmbedding so no HuggingFace download is needed.
"""

import json
import pytest
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.embeddings import MockEmbedding

from src.agents.subagent import ExpertSubagent
from src.agents.orchestrator import build_orchestrator_graph, OrchestratorState
from src.experiments.run_clean import build_clean_agents
from src.retriever import Retriever
from src.schemas import OrchestratorOutput, RunLog, SubagentOutput


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

DOCS = [
    Document(text="The capital of France is Paris.", doc_id="d1"),
    Document(text="Berlin is the capital of Germany.", doc_id="d2"),
    Document(text="Rome is the capital of Italy.", doc_id="d3"),
]


@pytest.fixture(scope="module")
def mock_index():
    embed_model = MockEmbedding(embed_dim=8)
    return VectorStoreIndex.from_documents(DOCS, embed_model=embed_model, show_progress=False)


@pytest.fixture(scope="module")
def retriever(mock_index):
    return Retriever(mock_index, top_k=2)


def stub_subagent_llm(agent_id: str):
    """Return a deterministic subagent LLM stub for the given agent."""
    def fn(prompt: str) -> str:
        return json.dumps({
            "answer": f"Paris (from {agent_id})",
            "citations": ["d1"],
            "confidence": 0.9,
            "rationale": f"{agent_id} found a clear document stating Paris.",
        })
    return fn


def stub_orchestrator_llm(agent_outputs):
    """Return a deterministic orchestrator LLM stub that picks subagent_1."""
    def fn(prompt: str) -> str:
        return json.dumps({
            "final_answer": "Paris",
            "winning_agent_id": "subagent_1",
            "reasoning_summary": "All agents agree; selecting subagent_1.",
        })
    return fn


# ---------------------------------------------------------------------------
# ExpertSubagent tests
# ---------------------------------------------------------------------------

def test_subagent_returns_subagent_output(retriever):
    agent = ExpertSubagent(
        agent_id="subagent_1",
        retriever=retriever,
        llm_fn=stub_subagent_llm("subagent_1"),
    )
    out = agent.run("What is the capital of France?")
    assert isinstance(out, SubagentOutput)
    assert out.agent_id == "subagent_1"


def test_subagent_output_has_required_fields(retriever):
    agent = ExpertSubagent(
        agent_id="subagent_1",
        retriever=retriever,
        llm_fn=stub_subagent_llm("subagent_1"),
    )
    out = agent.run("What is the capital of France?")
    assert out.answer
    assert isinstance(out.citations, list)
    assert 0.0 <= out.confidence <= 1.0
    assert out.rationale
    assert isinstance(out.retrieved_doc_ids, list)
    assert out.poison_retrieved is False


def test_subagent_poison_flag_clear_when_no_poison_docs(retriever):
    agent = ExpertSubagent(
        agent_id="subagent_1",
        retriever=retriever,
        poison_doc_ids=set(),
        llm_fn=stub_subagent_llm("subagent_1"),
    )
    out = agent.run("What is the capital of France?")
    assert out.poison_retrieved is False


def test_subagent_trigger_appended_to_retrieval_query(mock_index):
    """Trigger is appended to the retrieval query but does not break the run."""
    retriever = Retriever(mock_index, top_k=2)
    agent = ExpertSubagent(
        agent_id="subagent_1",
        retriever=retriever,
        llm_fn=stub_subagent_llm("subagent_1"),
    )
    out = agent.run("What is the capital of France?", trigger="ACTIVATE")
    assert isinstance(out, SubagentOutput)


# ---------------------------------------------------------------------------
# Orchestrator graph tests
# ---------------------------------------------------------------------------

def test_orchestrator_graph_produces_final_decision(mock_index):
    retriever = Retriever(mock_index, top_k=2)
    agents = [
        ExpertSubagent(f"subagent_{i}", retriever,
                       llm_fn=stub_subagent_llm(f"subagent_{i}"))
        for i in range(1, 4)
    ]
    app = build_orchestrator_graph(agents, llm_fn=stub_orchestrator_llm(None))

    state: OrchestratorState = {
        "query": "What is the capital of France?",
        "query_id": "q001",
        "attack_condition": "clean",
        "trigger": None,
        "agent_outputs": [],
        "final_decision": None,
    }
    result = app.invoke(state)

    assert result["final_decision"] is not None
    assert isinstance(result["final_decision"], OrchestratorOutput)


def test_orchestrator_graph_collects_all_three_agents(mock_index):
    retriever = Retriever(mock_index, top_k=2)
    agents = [
        ExpertSubagent(f"subagent_{i}", retriever,
                       llm_fn=stub_subagent_llm(f"subagent_{i}"))
        for i in range(1, 4)
    ]
    app = build_orchestrator_graph(agents, llm_fn=stub_orchestrator_llm(None))

    state: OrchestratorState = {
        "query": "What is the capital of France?",
        "query_id": "q001",
        "attack_condition": "clean",
        "trigger": None,
        "agent_outputs": [],
        "final_decision": None,
    }
    result = app.invoke(state)

    agent_ids = {o.agent_id for o in result["agent_outputs"]}
    assert agent_ids == {"subagent_1", "subagent_2", "subagent_3"}


def test_runlog_has_all_required_fields(mock_index):
    retriever = Retriever(mock_index, top_k=2)
    agents = [
        ExpertSubagent(f"subagent_{i}", retriever,
                       llm_fn=stub_subagent_llm(f"subagent_{i}"))
        for i in range(1, 4)
    ]
    app = build_orchestrator_graph(agents, llm_fn=stub_orchestrator_llm(None))

    state: OrchestratorState = {
        "query": "What is the capital of France?",
        "query_id": "q_test",
        "attack_condition": "clean",
        "trigger": None,
        "agent_outputs": [],
        "final_decision": None,
    }
    final_state = app.invoke(state)

    agent_responses = {o.agent_id: o for o in final_state["agent_outputs"]}
    retrieved_per_agent = {o.agent_id: o.retrieved_doc_ids for o in final_state["agent_outputs"]}

    log = RunLog(
        query_id="q_test",
        attack_condition="clean",
        trigger=None,
        retrieved_doc_ids_per_agent=retrieved_per_agent,
        poison_retrieved=False,
        agent_responses=agent_responses,
        final_decision=final_state["final_decision"],
        metrics={},
    )

    # All 3 agents logged
    assert set(log.agent_responses.keys()) == {"subagent_1", "subagent_2", "subagent_3"}
    assert set(log.retrieved_doc_ids_per_agent.keys()) == {"subagent_1", "subagent_2", "subagent_3"}

    # Final decision populated
    assert log.final_decision is not None
    assert log.final_decision.final_answer
    assert log.final_decision.winning_subagents

    # Attack fields are clean
    assert log.poison_retrieved is False
    assert log.trigger is None
    assert log.final_decision.harmful_action_flag is False
