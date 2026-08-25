"""Deterministic tests for controlled checkpoint replay and fork."""

from __future__ import annotations

import importlib
import importlib.util
from types import ModuleType
from typing import TypedDict

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import StateSnapshot


def _time_travel_module() -> ModuleType:
    spec = importlib.util.find_spec("langgraph_agent_lab.time_travel")
    assert spec is not None, "the time-travel extension has not been implemented"
    return importlib.import_module("langgraph_agent_lab.time_travel")


class TravelState(TypedDict, total=False):
    query: str
    final_answer: str
    events: list[dict[str, object]]
    completed: bool


def _safe_graph() -> CompiledStateGraph:
    def draft(state: TravelState) -> TravelState:
        return {"final_answer": f"draft:{state['query']}"}

    def finalize(_state: TravelState) -> TravelState:
        return {"completed": True}

    workflow = StateGraph(TravelState)
    workflow.add_node("draft", draft)
    workflow.add_node("finalize", finalize)
    workflow.add_edge(START, "draft")
    workflow.add_edge("draft", "finalize")
    workflow.add_edge("finalize", END)
    return workflow.compile(checkpointer=MemorySaver())


def _checkpoint_id(snapshot: StateSnapshot) -> str:
    return str(snapshot.config["configurable"]["checkpoint_id"])


def test_selects_checkpoint_by_id_or_predicate_not_history_position() -> None:
    """Choosing a checkpoint by list index instead of stable identity breaks this test."""
    travel = _time_travel_module()
    graph = _safe_graph()
    thread_id = "travel-select"
    config = {"configurable": {"thread_id": thread_id}}
    graph.invoke({"query": "original", "events": []}, config=config)
    expected = next(
        snapshot for snapshot in graph.get_state_history(config) if snapshot.next == ("finalize",)
    )
    expected_id = _checkpoint_id(expected)

    by_id = travel.select_checkpoint(graph, thread_id, checkpoint_id=expected_id)
    by_predicate = travel.select_checkpoint(
        graph,
        thread_id,
        predicate=lambda view: view.next_nodes == ("finalize",),
    )

    assert _checkpoint_id(by_id) == expected_id
    assert _checkpoint_id(by_predicate) == expected_id
    with pytest.raises(LookupError, match="checkpoint"):
        travel.select_checkpoint(graph, thread_id, checkpoint_id="not-a-checkpoint")
    with pytest.raises(ValueError, match="exactly one"):
        travel.select_checkpoint(graph, thread_id)


def test_replays_selected_checkpoint_and_forks_whitelisted_overwrite() -> None:
    """Replaying the latest state or merging append-only fields breaks this test."""
    travel = _time_travel_module()
    graph = _safe_graph()
    thread_id = "travel-replay"
    config = {"configurable": {"thread_id": thread_id}}
    graph.invoke({"query": "original", "events": []}, config=config)
    selected = travel.select_checkpoint(
        graph,
        thread_id,
        predicate=lambda view: view.next_nodes == ("finalize",),
    )
    checkpoint_id = _checkpoint_id(selected)

    replayed = travel.replay_checkpoint(graph, thread_id, checkpoint_id)
    fork_config = travel.fork_checkpoint(
        graph,
        thread_id,
        checkpoint_id,
        updates={"final_answer": "reviewed answer"},
    )
    forked = graph.invoke(None, config=fork_config)

    assert replayed["completed"] is True
    assert replayed["final_answer"] == "draft:original"
    assert forked["completed"] is True
    assert forked["final_answer"] == "reviewed answer"
    assert fork_config["configurable"]["checkpoint_id"] != checkpoint_id

    with pytest.raises(ValueError, match="reviewable content"):
        travel.fork_checkpoint(
            graph,
            thread_id,
            checkpoint_id,
            updates={"events": [{"node": "forged"}]},
        )


@pytest.mark.parametrize(
    "control_field",
    [
        "approval",
        "route",
        "risk_level",
        "attempt",
        "max_attempts",
        "evaluation_result",
        "proposed_action",
    ],
)
def test_fork_rejects_control_state_forgery(control_field: str) -> None:
    """A time-travel branch must not rewrite routing, approval, or evaluator control state."""
    travel = _time_travel_module()
    graph = _safe_graph()
    thread_id = f"travel-control-{control_field}"
    config = {"configurable": {"thread_id": thread_id}}
    graph.invoke({"query": "original", "events": []}, config=config)
    selected = travel.select_checkpoint(
        graph,
        thread_id,
        predicate=lambda view: view.next_nodes == ("finalize",),
    )

    with pytest.raises(ValueError, match="reviewable content"):
        travel.fork_checkpoint(
            graph,
            thread_id,
            _checkpoint_id(selected),
            updates={control_field: "forged"},
        )


def test_query_fork_is_only_allowed_before_classification() -> None:
    """Changing the query after classification could retain stale risk and routing decisions."""
    travel = _time_travel_module()
    graph = _safe_graph()
    thread_id = "travel-query-phase"
    config = {"configurable": {"thread_id": thread_id}}
    graph.invoke({"query": "original", "events": []}, config=config)
    selected = travel.select_checkpoint(
        graph,
        thread_id,
        predicate=lambda view: view.next_nodes == ("finalize",),
    )

    with pytest.raises(ValueError, match="before classification"):
        travel.fork_checkpoint(
            graph,
            thread_id,
            _checkpoint_id(selected),
            updates={"query": "changed"},
        )

    def classify(state: TravelState) -> TravelState:
        return {"final_answer": f"classified:{state['query']}"}

    classify_workflow = StateGraph(TravelState)
    classify_workflow.add_node("classify", classify)
    classify_workflow.add_edge(START, "classify")
    classify_workflow.add_edge("classify", END)
    classify_graph = classify_workflow.compile(checkpointer=MemorySaver())
    classify_thread = "travel-query-classify"
    classify_config = {"configurable": {"thread_id": classify_thread}}
    classify_graph.invoke({"query": "original", "events": []}, config=classify_config)
    before_classify = travel.select_checkpoint(
        classify_graph,
        classify_thread,
        predicate=lambda view: view.next_nodes == ("classify",),
    )

    fork_config = travel.fork_checkpoint(
        classify_graph,
        classify_thread,
        _checkpoint_id(before_classify),
        updates={"query": "reviewed"},
        allowed_next_nodes={"classify"},
    )
    result = classify_graph.invoke(None, config=fork_config)
    assert result["final_answer"] == "classified:reviewed"


def test_replay_and_fork_block_unapproved_side_effect_nodes() -> None:
    """Executing a tool checkpoint without an explicit safety allowance breaks this test."""
    travel = _time_travel_module()

    def tool(_state: TravelState) -> TravelState:
        return {"final_answer": "side effect executed"}

    workflow = StateGraph(TravelState)
    workflow.add_node("tool", tool)
    workflow.add_edge(START, "tool")
    workflow.add_edge("tool", END)
    graph = workflow.compile(checkpointer=MemorySaver())
    thread_id = "travel-side-effect"
    config = {"configurable": {"thread_id": thread_id}}
    graph.invoke({"query": "act", "events": []}, config=config)
    selected = travel.select_checkpoint(
        graph,
        thread_id,
        predicate=lambda view: view.next_nodes == ("tool",),
    )
    checkpoint_id = _checkpoint_id(selected)

    with pytest.raises(travel.UnsafeReplayError, match="tool"):
        travel.replay_checkpoint(graph, thread_id, checkpoint_id)
    with pytest.raises(travel.UnsafeReplayError, match="tool"):
        travel.fork_checkpoint(
            graph,
            thread_id,
            checkpoint_id,
            updates={"query": "changed"},
        )
