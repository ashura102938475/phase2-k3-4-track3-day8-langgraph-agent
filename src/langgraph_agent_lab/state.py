"""State schema for the Day 08 LangGraph lab.

Students should extend the schema only when needed. Keep state lean and serializable.
"""

from __future__ import annotations

from enum import StrEnum
from operator import add
from typing import Annotated, Any, NotRequired, TypedDict

from pydantic import BaseModel, Field, field_validator


class Route(StrEnum):
    SIMPLE = "simple"
    TOOL = "tool"
    MISSING_INFO = "missing_info"
    RISKY = "risky"
    ERROR = "error"
    DEAD_LETTER = "dead_letter"
    DONE = "done"


class LabEvent(BaseModel):
    """Append-only audit event for grading and debugging."""

    node: str
    event_type: str
    message: str
    latency_ms: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ApprovalDecision(BaseModel):
    approved: bool = False
    reviewer: str = "mock-reviewer"
    comment: str = ""


class ParallelToolResult(TypedDict):
    """Serializable result emitted by one parallel tool branch."""

    task: str
    result: str
    attempt: NotRequired[int]


def merge_parallel_tool_results(
    left: list[ParallelToolResult], right: list[ParallelToolResult]
) -> list[ParallelToolResult]:
    """Merge branch results in a deterministic, associative, commutative form.

    Malformed entries are ignored at this trust boundary. Exact duplicates collapse,
    which also makes replaying a checkpoint idempotent.
    """
    normalized: dict[str, tuple[tuple[int, str, bool], ParallelToolResult]] = {}
    for item in [*left, *right]:
        if not isinstance(item, dict):
            continue
        task = item.get("task")
        result = item.get("result")
        attempt = item.get("attempt", 0)
        if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 0:
            continue
        if isinstance(task, str) and task.strip() and isinstance(result, str):
            has_attempt = "attempt" in item
            candidate: ParallelToolResult = {"task": task, "result": result}
            if has_attempt:
                candidate["attempt"] = attempt
            rank = (attempt, result, has_attempt)
            current = normalized.get(task)
            if current is None or rank > current[0]:
                normalized[task] = (rank, candidate)
    return [normalized[task][1] for task in sorted(normalized)]


class AgentState(TypedDict, total=False):
    """LangGraph state.

    Append-only audit/history fields use reducers; current-value fields overwrite.
    """

    thread_id: str
    scenario_id: str
    query: str
    route: str
    risk_level: str
    attempt: int
    max_attempts: int
    final_answer: str | None
    # These values represent the current workflow state and are overwritten by
    # each node update.  Keep approval as a plain mapping so checkpoints remain
    # JSON-serializable (rather than storing a Pydantic model instance).
    evaluation_result: str | None
    evaluation_reason: str | None
    evaluation_source: str | None
    judge_calls: int
    pending_question: str | None
    proposed_action: str | None
    approval: dict[str, Any] | None
    tool_tasks: list[str]
    active_tool_task: str | None
    parallel_tool_results: Annotated[
        list[ParallelToolResult], merge_parallel_tool_results
    ]
    messages: Annotated[list[str], add]
    tool_results: Annotated[list[str], add]
    errors: Annotated[list[str], add]
    events: Annotated[list[dict[str, Any]], add]


class Scenario(BaseModel):
    id: str
    query: str
    expected_route: Route
    requires_approval: bool = False
    should_retry: bool = False
    max_attempts: int = 3
    tags: list[str] = Field(default_factory=list)
    tool_tasks: list[str] = Field(default_factory=list, max_length=4)

    @field_validator("query")
    @classmethod
    def query_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be empty")
        return value

    @field_validator("tool_tasks")
    @classmethod
    def tool_tasks_must_be_non_empty_and_unique(cls, value: list[str]) -> list[str]:
        normalized = [task.strip() for task in value]
        if any(not task for task in normalized):
            raise ValueError("tool tasks must not be empty")
        if len(set(normalized)) != len(normalized):
            raise ValueError("tool tasks must be unique")
        return normalized


def initial_state(scenario: Scenario) -> AgentState:
    """Create a serializable initial state for one scenario."""
    return {
        "thread_id": f"thread-{scenario.id}",
        "scenario_id": scenario.id,
        "query": scenario.query,
        "route": "",
        "risk_level": "unknown",
        "attempt": 0,
        "max_attempts": scenario.max_attempts,
        "final_answer": None,
        "evaluation_result": None,
        "evaluation_reason": None,
        "evaluation_source": None,
        "judge_calls": 0,
        "pending_question": None,
        "proposed_action": None,
        "approval": None,
        "tool_tasks": list(scenario.tool_tasks),
        "active_tool_task": None,
        "parallel_tool_results": [],
        "messages": [],
        "tool_results": [],
        "errors": [],
        "events": [],
    }


def make_event(node: str, event_type: str, message: str, **metadata: object) -> dict[str, Any]:
    """Create a normalized event payload."""
    return LabEvent(
        node=node,
        event_type=event_type,
        message=message,
        metadata=metadata,
    ).model_dump()
