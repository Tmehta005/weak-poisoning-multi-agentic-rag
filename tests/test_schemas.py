"""
Tests for typed schemas (src/schemas.py).

Validates that all schemas can be instantiated with valid data
and reject invalid data, and that RunLog contains all fields
required by CLAUDE.md logging requirements.
"""

import pytest
from pydantic import ValidationError

from src.schemas import (
    OrchestratorOutput,
    RetrievedDoc,
    RunLog,
    SubagentOutput,
)


# ---------------------------------------------------------------------------
# RetrievedDoc
# ---------------------------------------------------------------------------

def test_retrieved_doc_basic():
    doc = RetrievedDoc(doc_id="d1", text="hello world", score=0.9)
    assert doc.doc_id == "d1"
    assert doc.score == 0.9


# ---------------------------------------------------------------------------
# SubagentOutput
# ---------------------------------------------------------------------------

def test_subagent_output_basic():
    out = SubagentOutput(
        agent_id="subagent_1",
        answer="Paris",
        citations=["d1", "d2"],
        confidence=0.85,
        rationale="The document clearly states Paris.",
        poison_retrieved=False,
        retrieved_doc_ids=["d1", "d2"],
    )
    assert out.agent_id == "subagent_1"
    assert out.poison_retrieved is False


def test_subagent_output_confidence_bounds():
    with pytest.raises(ValidationError):
        SubagentOutput(
            agent_id="x",
            answer="a",
            confidence=1.5,  # out of range
            rationale="r",
        )


def test_subagent_output_defaults():
    out = SubagentOutput(
        agent_id="x",
        answer="a",
        confidence=0.5,
        rationale="r",
    )
    assert out.citations == []
    assert out.retrieved_doc_ids == []
    assert out.poison_retrieved is False


# ---------------------------------------------------------------------------
# OrchestratorOutput
# ---------------------------------------------------------------------------

def test_orchestrator_output_basic():
    out = OrchestratorOutput(
        final_answer="Paris",
        selected_evidence=["d1"],
        winning_subagents=["subagent_1"],
        final_confidence=0.9,
        harmful_action_flag=False,
    )
    assert out.harmful_action_flag is False
    assert out.reasoning_summary is None


def test_orchestrator_output_harmful_flag():
    out = OrchestratorOutput(
        final_answer="BUY CRYPTO",
        selected_evidence=[],
        winning_subagents=["subagent_1"],
        final_confidence=0.7,
        harmful_action_flag=True,
        reasoning_summary="Poisoned subagent won.",
    )
    assert out.harmful_action_flag is True


# ---------------------------------------------------------------------------
# RunLog — checks all CLAUDE.md required fields are present
# ---------------------------------------------------------------------------

REQUIRED_RUNLOG_FIELDS = {
    "query_id",
    "attack_condition",
    "trigger",
    "retrieved_doc_ids_per_agent",
    "poison_retrieved",
    "agent_responses",
    "final_decision",
    "metrics",
}


def test_runlog_has_required_fields():
    assert REQUIRED_RUNLOG_FIELDS.issubset(RunLog.model_fields.keys())


def test_runlog_basic():
    subagent_out = SubagentOutput(
        agent_id="baseline",
        answer="Paris",
        confidence=0.8,
        rationale="r",
    )
    orchestrator_out = OrchestratorOutput(
        final_answer="Paris",
        winning_subagents=["baseline"],
        final_confidence=0.8,
        harmful_action_flag=False,
    )
    log = RunLog(
        query_id="q001",
        attack_condition="clean",
        retrieved_doc_ids_per_agent={"baseline": ["d1"]},
        agent_responses={"baseline": subagent_out},
        final_decision=orchestrator_out,
    )
    assert log.query_id == "q001"
    assert log.poison_retrieved is False
    assert log.trigger is None
    assert log.metrics == {}


def test_runlog_attack_condition_logged():
    log = RunLog(
        query_id="q002",
        attack_condition="main_injection",
        trigger="ACTIVATE",
        retrieved_doc_ids_per_agent={"subagent_1": ["p1", "d2"]},
        poison_retrieved=True,
        agent_responses={},
        metrics={"asr": 1.0},
    )
    assert log.attack_condition == "main_injection"
    assert log.poison_retrieved is True
    assert log.trigger == "ACTIVATE"
