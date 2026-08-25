"""Offline integration tests for the optional real HITL runner."""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Callable
from types import ModuleType

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.types import interrupt

import langgraph_agent_lab.nodes as nodes
from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.state import AgentState, make_event


def _hitl_module() -> ModuleType:
    """Load the extension only after giving a useful RED assertion."""
    spec = importlib.util.find_spec("langgraph_agent_lab.hitl")
    assert spec is not None, "the HITL runner extension has not been implemented"
    return importlib.import_module("langgraph_agent_lab.hitl")


def _install_offline_risky_route(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep graph wiring and approval real; replace only LLM/tool work."""

    def classify(_state: AgentState) -> dict[str, object]:
        return {
            "route": "risky",
            "risk_level": "high",
            "events": [
                make_event(
                    "classify",
                    "completed",
                    "classified as risky; api_key=TOPSECRET",
                )
            ],
        }

    def tool(state: AgentState) -> dict[str, object]:
        return {
            "tool_results": [f"offline result for {state.get('query', '')}"],
            "events": [make_event("tool", "completed", "offline tool completed")],
        }

    def evaluate(_state: AgentState) -> dict[str, object]:
        return {
            "evaluation_result": "success",
            "events": [make_event("evaluate", "completed", "offline result accepted")],
        }

    def answer(_state: AgentState) -> dict[str, object]:
        return {
            "final_answer": "offline grounded answer",
            "events": [make_event("answer", "completed", "offline answer generated")],
        }

    monkeypatch.setattr(nodes, "classify_node", classify)
    monkeypatch.setattr(nodes, "tool_node", tool)
    monkeypatch.setattr(nodes, "evaluate_node", evaluate)
    monkeypatch.setattr(nodes, "answer_node", answer)


def test_runner_pauses_and_resumes_one_real_approval_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Removing interrupt mode, keyed resume, or thread reuse breaks this test."""
    hitl = _hitl_module()
    _install_offline_risky_route(monkeypatch)
    graph = build_graph(checkpointer=MemorySaver())
    runner = hitl.HitlRunner(graph=graph)

    fake_credential = "sk-testonly_" + ("a" * 24)
    pending = runner.start(
        f"Delete offline ticket token=TOPSECRET and remove {fake_credential}"
    )

    assert pending.status == "pending"
    assert pending.thread_id.startswith("hitl-")
    assert pending.interrupt_id
    assert pending.approval is None
    assert "TOPSECRET" not in pending.model_dump_json()
    assert fake_credential not in pending.model_dump_json()
    assert [event.node for event in pending.events] == ["intake", "classify", "risky_action"]

    with pytest.raises(ValueError, match="interrupt"):
        runner.resume(
            pending.thread_id,
            interrupt_id="wrong-interrupt",
            approved=True,
            reviewer="reviewer-1",
            comment="approved offline",
        )

    completed = runner.resume(
        pending.thread_id,
        interrupt_id=pending.interrupt_id,
        approved=True,
        reviewer="reviewer-1",
        comment="approved offline",
    )

    assert completed.thread_id == pending.thread_id
    assert completed.status == "completed"
    assert completed.approval is not None and completed.approval.approved is True
    assert completed.final_answer == "offline grounded answer"
    assert [event.node for event in completed.events][-5:] == [
        "approval",
        "tool",
        "evaluate",
        "answer",
        "finalize",
    ]

    with pytest.raises(RuntimeError, match="active approval"):
        runner.resume(
            pending.thread_id,
            approved=True,
            reviewer="reviewer-1",
            comment="duplicate",
        )


def test_runner_rejection_continues_to_clarification_without_tool_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Routing a rejection to the tool instead of clarification breaks this test."""
    hitl = _hitl_module()
    _install_offline_risky_route(monkeypatch)
    runner = hitl.HitlRunner(graph=build_graph(checkpointer=MemorySaver()))
    pending = runner.start("Delete offline ticket")

    rejected = runner.resume(
        pending.thread_id,
        approved=False,
        reviewer="reviewer-2",
        comment="insufficient evidence",
    )

    assert rejected.status == "completed"
    assert rejected.approval is not None and rejected.approval.approved is False
    assert rejected.pending_question
    assert "tool" not in [event.node for event in rejected.events]
    assert [event.node for event in rejected.events][-3:] == ["approval", "clarify", "finalize"]


def test_runner_rejects_multiple_simultaneous_interrupts() -> None:
    """Accepting an ambiguous multi-interrupt state breaks this test."""
    hitl = _hitl_module()

    class ParallelState(dict):
        pass

    def pause(label: str) -> Callable[[dict[str, object]], dict[str, object]]:
        def node(_state: dict[str, object]) -> dict[str, object]:
            interrupt({"proposed_action": label})
            return {}

        return node

    workflow = StateGraph(ParallelState)
    workflow.add_node("approval", pause("one"))
    workflow.add_node("approval_shadow", pause("two"))
    workflow.add_edge(START, "approval")
    workflow.add_edge(START, "approval_shadow")
    graph = workflow.compile(checkpointer=MemorySaver())

    with pytest.raises(RuntimeError, match="exactly one"):
        hitl.HitlRunner(graph=graph).start("offline ticket")


def test_default_runner_matches_streamlit_boundary() -> None:
    """A constructor that requires infrastructure arguments breaks the UI contract."""
    hitl = _hitl_module()

    runner = hitl.HitlRunner()

    assert runner is not None
