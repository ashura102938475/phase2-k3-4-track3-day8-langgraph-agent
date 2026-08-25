"""Offline contract tests for graph construction and route completion."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from langgraph.checkpoint.memory import MemorySaver

import langgraph_agent_lab.nodes as nodes
from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.state import Route

REQUIRED_NODES = {
    "intake",
    "classify",
    "answer",
    "tool",
    "evaluate",
    "clarify",
    "risky_action",
    "approval",
    "retry",
    "dead_letter",
    "finalize",
}


def _install_offline_nodes(
    monkeypatch: pytest.MonkeyPatch,
    route: str,
    *,
    approved: bool = True,
) -> list[str]:
    """Replace only external/side-effecting node work while keeping graph routing real."""
    visited: list[str] = []

    def record(
        name: str, update: dict[str, Any]
    ) -> Callable[[dict[str, Any]], dict[str, Any]]:
        def node(_state: dict[str, Any]) -> dict[str, Any]:
            visited.append(name)
            return update

        return node

    def retry(state: dict[str, Any]) -> dict[str, Any]:
        visited.append("retry")
        return {"attempt": state.get("attempt", 0) + 1, "errors": ["offline retry"]}

    doubles = {
        "intake_node": record("intake", {"query": "offline"}),
        "classify_node": record(
            "classify",
            {"route": route, "risk_level": "high" if route == Route.RISKY.value else "low"},
        ),
        "answer_node": record("answer", {"final_answer": "offline answer"}),
        "tool_node": record("tool", {"tool_results": ["SUCCESS"]}),
        "evaluate_node": record("evaluate", {"evaluation_result": "success"}),
        "ask_clarification_node": record("clarify", {"pending_question": "Please clarify"}),
        "risky_action_node": record("risky_action", {"proposed_action": "offline action"}),
        "approval_node": record("approval", {"approval": {"approved": approved}}),
        "retry_or_fallback_node": retry,
        "dead_letter_node": record("dead_letter", {"final_answer": "offline dead letter"}),
        "finalize_node": record(
            "finalize", {"events": [{"node": "finalize", "event_type": "completed"}]}
        ),
    }
    for name, function in doubles.items():
        monkeypatch.setattr(nodes, name, function)
    return visited


def test_build_graph_compiles_with_memory_saver() -> None:
    """The workflow is compilable with the supported in-memory checkpointer."""
    compiled = build_graph(checkpointer=MemorySaver())

    assert compiled is not None


def test_graph_registers_exactly_the_required_nodes() -> None:
    """The graph contains exactly the documented application nodes."""
    compiled = build_graph(checkpointer=MemorySaver())
    registered = set(compiled.nodes) - {"__start__", "__end__"}

    assert registered == REQUIRED_NODES


def test_graph_has_exact_documented_topology() -> None:
    compiled = build_graph(checkpointer=MemorySaver())
    actual_edges = {
        (edge.source, edge.target, edge.conditional) for edge in compiled.get_graph().edges
    }
    expected_edges = {
        ("__start__", "intake", False),
        ("intake", "classify", False),
        ("tool", "evaluate", False),
        ("risky_action", "approval", False),
        ("answer", "finalize", False),
        ("clarify", "finalize", False),
        ("dead_letter", "finalize", False),
        ("finalize", "__end__", False),
        ("classify", "answer", True),
        ("classify", "tool", True),
        ("classify", "clarify", True),
        ("classify", "risky_action", True),
        ("classify", "retry", True),
        ("evaluate", "answer", True),
        ("evaluate", "retry", True),
        ("retry", "tool", True),
        ("retry", "dead_letter", True),
        ("approval", "tool", True),
        ("approval", "clarify", True),
    }

    assert actual_edges == expected_edges


@pytest.mark.parametrize(
    ("route", "expected_visits"),
    [
        (Route.SIMPLE.value, ["intake", "classify", "answer", "finalize"]),
        (
            Route.TOOL.value,
            ["intake", "classify", "tool", "evaluate", "answer", "finalize"],
        ),
        (Route.MISSING_INFO.value, ["intake", "classify", "clarify", "finalize"]),
        (
            Route.RISKY.value,
            [
                "intake",
                "classify",
                "risky_action",
                "approval",
                "tool",
                "evaluate",
                "answer",
                "finalize",
            ],
        ),
        (
            Route.ERROR.value,
            ["intake", "classify", "retry", "tool", "evaluate", "answer", "finalize"],
        ),
    ],
)
def test_deterministic_routes_terminate_through_finalize(
    monkeypatch: pytest.MonkeyPatch,
    route: str,
    expected_visits: list[str],
) -> None:
    """Offline node doubles prove each supported route reaches the finalizer."""
    visited = _install_offline_nodes(monkeypatch, route)

    compiled = build_graph(checkpointer=MemorySaver())
    result = compiled.invoke(
        {"query": "offline", "route": "", "attempt": 0, "max_attempts": 2},
        config={"configurable": {"thread_id": f"contract-{route}"}},
    )

    assert any(event.get("node") == "finalize" for event in result.get("events", []))
    assert visited == expected_visits


def test_retry_boundary_dead_letters_without_calling_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    visited = _install_offline_nodes(monkeypatch, Route.ERROR.value)
    compiled = build_graph(checkpointer=MemorySaver())

    result = compiled.invoke(
        {"query": "offline", "route": "", "attempt": 0, "max_attempts": 1},
        config={"configurable": {"thread_id": "contract-dead-letter"}},
    )

    assert visited == ["intake", "classify", "retry", "dead_letter", "finalize"]
    assert result["attempt"] == 1
    assert result["final_answer"] == "offline dead letter"


def test_rejected_approval_clarifies_without_calling_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    visited = _install_offline_nodes(monkeypatch, Route.RISKY.value, approved=False)
    compiled = build_graph(checkpointer=MemorySaver())

    compiled.invoke(
        {"query": "offline", "route": "", "attempt": 0, "max_attempts": 2},
        config={"configurable": {"thread_id": "contract-rejected"}},
    )

    assert visited == ["intake", "classify", "risky_action", "approval", "clarify", "finalize"]


def test_memory_checkpointer_records_history_for_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_offline_nodes(monkeypatch, Route.SIMPLE.value)
    compiled = build_graph(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "contract-history"}}

    compiled.invoke(
        {"query": "offline", "route": "", "attempt": 0, "max_attempts": 2},
        config=config,
    )

    history = list(compiled.get_state_history(config))
    assert len(history) > 1
    assert all(item.config["configurable"]["thread_id"] == "contract-history" for item in history)
