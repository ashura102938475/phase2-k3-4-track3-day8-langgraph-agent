"""Extension contracts for deterministic parallel tool fan-out."""

from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Send

import langgraph_agent_lab.nodes as nodes
import langgraph_agent_lab.state as state_module
from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.routing import (
    route_after_approval,
    route_after_classify,
    route_after_retry,
)
from langgraph_agent_lab.state import Route, Scenario, initial_state


class _Message:
    def __init__(self, content: str) -> None:
        self.content = content


class _AnswerLLM:
    def __init__(self) -> None:
        self.prompt = ""

    def invoke(self, prompt: object) -> _Message:
        self.prompt = str(prompt)
        return _Message("parallel answer")


def test_scenario_normalizes_tool_tasks_and_propagates_them_to_initial_state() -> None:
    """Scenario-driven runs retain the independent tasks needed to activate Send."""
    scenario = Scenario(
        id="parallel",
        query="lookup two records",
        expected_route=Route.TOOL,
        tool_tasks=[" zeta ", "alpha"],
    )

    assert scenario.tool_tasks == ["zeta", "alpha"]
    assert initial_state(scenario)["tool_tasks"] == ["zeta", "alpha"]


@pytest.mark.parametrize(
    "tool_tasks",
    [
        [""],
        ["alpha", " alpha "],
        ["one", "two", "three", "four", "five"],
    ],
)
def test_scenario_rejects_invalid_parallel_tool_tasks(tool_tasks: list[str]) -> None:
    """Blank, duplicate, or excessive branches cannot enter graph state."""
    with pytest.raises(ValueError):
        Scenario(
            id="parallel-invalid",
            query="lookup records",
            expected_route=Route.TOOL,
            tool_tasks=tool_tasks,
        )


def test_parallel_result_reducer_is_associative_commutative_and_deterministic() -> None:
    """Branch completion order cannot change the checkpointed result list."""
    alpha = [{"task": "alpha", "result": "A"}]
    beta = [{"task": "beta", "result": "B"}]
    gamma = [{"task": "gamma", "result": "C"}]

    merge = state_module.merge_parallel_tool_results
    assert merge(alpha, beta) == merge(beta, alpha)
    assert merge(merge(alpha, beta), gamma) == merge(alpha, merge(beta, gamma))
    assert merge(beta, alpha) == [
        {"task": "alpha", "result": "A"},
        {"task": "beta", "result": "B"},
    ]


def test_parallel_result_reducer_keeps_latest_attempt_per_task() -> None:
    """A recovered retry must not remain poisoned by an older branch error."""
    failed = [{"task": "account", "result": "ERROR: timeout", "attempt": 0}]
    recovered = [{"task": "account", "result": "account result", "attempt": 1}]

    merge = state_module.merge_parallel_tool_results

    assert merge(failed, recovered) == merge(recovered, failed) == [
        {"task": "account", "result": "account result", "attempt": 1}
    ]


def test_tool_route_returns_real_send_objects_only_for_multiple_tasks() -> None:
    """Fan-out is opt-in and uses LangGraph's Send API rather than a simulated loop."""
    routed = route_after_classify(
        {"route": "tool", "tool_tasks": ["zeta", "alpha"], "query": "lookup"}
    )

    assert isinstance(routed, list)
    assert all(isinstance(item, Send) for item in routed)
    assert [(item.node, item.arg["active_tool_task"]) for item in routed] == [
        ("tool", "alpha"),
        ("tool", "zeta"),
    ]
    assert route_after_classify({"route": "tool", "tool_tasks": ["alpha"]}) == "tool"
    assert route_after_classify({"route": "tool"}) == "tool"


def test_retry_can_repeat_fanout_but_risky_approval_never_fans_out() -> None:
    """Retries preserve parallel lookup work; approved side effects stay serialized."""
    retry_route = route_after_retry(
        {
            "route": "tool",
            "attempt": 1,
            "max_attempts": 2,
            "tool_tasks": ["beta", "alpha"],
        }
    )

    assert isinstance(retry_route, list)
    assert [item.arg["active_tool_task"] for item in retry_route] == ["alpha", "beta"]
    assert (
        route_after_approval(
            {
                "route": "risky",
                "approval": {"approved": True},
                "tool_tasks": ["alpha", "beta"],
            }
        )
        == "tool"
    )


def test_compiled_graph_fans_in_sorted_results_before_evaluate_and_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two branch results merge once before evaluation and ground one final answer."""
    answer_llm = _AnswerLLM()
    monkeypatch.setattr(nodes, "get_llm", lambda **_kwargs: answer_llm)
    monkeypatch.setattr(nodes, "intake_node", lambda state: {"query": state["query"]})
    monkeypatch.setattr(
        nodes,
        "classify_node",
        lambda _state: {"route": "tool", "risk_level": "low"},
    )
    graph = build_graph(checkpointer=MemorySaver())
    result = graph.invoke(
        {
            "query": "lookup two records",
            "route": "",
            "attempt": 0,
            "max_attempts": 1,
            "tool_tasks": ["zeta", "alpha"],
            "parallel_tool_results": [],
            "tool_results": [],
            "events": [],
        },
        config={"configurable": {"thread_id": "parallel-fanout"}},
    )

    assert result["parallel_tool_results"] == [
        {"task": "alpha", "result": "Tool result for task: alpha", "attempt": 0},
        {"task": "zeta", "result": "Tool result for task: zeta", "attempt": 0},
    ]
    assert result["evaluation_result"] == "success"
    assert result["final_answer"] == "parallel answer"
    assert answer_llm.prompt.index("alpha") < answer_llm.prompt.index("zeta")


def test_parallel_evaluation_retries_if_any_branch_failed() -> None:
    """A successful sibling cannot hide a failed branch."""
    update = nodes.evaluate_node(
        {
            "parallel_tool_results": [
                {"task": "alpha", "result": "Tool result for task: alpha"},
                {"task": "beta", "result": "ERROR: beta timed out"},
            ],
            "events": [],
        }
    )

    assert update["evaluation_result"] == "needs_retry"


def test_answer_ignores_malformed_parallel_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Untrusted branch state cannot crash prompt construction or leak object reprs."""
    answer_llm = _AnswerLLM()
    monkeypatch.setattr(nodes, "get_llm", lambda **_kwargs: answer_llm)

    update = nodes.answer_node(
        {
            "query": "lookup",
            "tool_results": ["safe fallback"],
            "parallel_tool_results": [
                {"task": "alpha", "result": "A"},
                {"task": object(), "result": "unsafe"},
            ],
            "events": [],
        }
    )

    assert update["final_answer"] == "parallel answer"
    assert "alpha: A" in answer_llm.prompt
    assert "object at" not in answer_llm.prompt
