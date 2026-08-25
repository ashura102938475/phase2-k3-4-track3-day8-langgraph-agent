"""Routing functions for conditional edges.

Core routes return the next registered node name. The optional parallel lookup
extension returns LangGraph ``Send`` objects to the existing tool node.
"""

from __future__ import annotations

from langgraph.types import Send

from .state import AgentState


def _parallel_tool_sends(state: AgentState) -> list[Send] | None:
    """Create deterministic tool branches for explicitly independent tasks."""
    if state.get("route") == "risky":
        return None
    raw_tasks = state.get("tool_tasks")
    if not isinstance(raw_tasks, list):
        return None
    tasks = sorted(
        task.strip() for task in raw_tasks if isinstance(task, str) and task.strip()
    )
    if len(tasks) <= 1:
        return None
    shared = {
        key: state[key]
        for key in ("query", "route", "attempt", "max_attempts")
        if key in state
    }
    return [Send("tool", {**shared, "active_tool_task": task}) for task in tasks]


def route_after_classify(state: AgentState) -> str | list[Send]:
    """Map classified route to the next graph node.

    Mapping:
    - "simple"       → "answer"
    - "tool"         → "tool"
    - "missing_info" → "clarify"
    - "risky"        → "risky_action"
    - "error"        → "retry"
    - unknown/default → "answer"

    Hint: use a dict mapping for clean implementation.
    """
    destinations = {
        "simple": "answer",
        "tool": "tool",
        "missing_info": "clarify",
        "risky": "risky_action",
        "error": "retry",
    }
    route = state.get("route", "")
    if route == "tool" and (sends := _parallel_tool_sends(state)) is not None:
        return sends
    return destinations.get(route, "answer") if isinstance(route, str) else "answer"


def route_after_evaluate(state: AgentState) -> str:
    """Decide if tool result is satisfactory or needs retry.

    This is the 'done?' check that creates the retry loop —
    a key LangGraph advantage over linear LCEL chains.

    - If evaluation_result == "needs_retry" → "retry"
    - Otherwise → "answer"
    """
    return "retry" if state.get("evaluation_result") == "needs_retry" else "answer"


def route_after_retry(state: AgentState) -> str | list[Send]:
    """Decide whether to retry the tool or give up.

    MUST be bounded — unbounded retry loops will fail grading.

    - If attempt < max_attempts → "tool" (try again)
    - If attempt >= max_attempts → "dead_letter" (give up, escalate)
    """
    # Missing or malformed limits fail closed to the dead-letter path; a retry
    # router must never accidentally create an unbounded loop.
    attempt = state.get("attempt")
    max_attempts = state.get("max_attempts")
    if not isinstance(attempt, int) or isinstance(attempt, bool):
        return "dead_letter"
    if not isinstance(max_attempts, int) or isinstance(max_attempts, bool):
        return "dead_letter"
    if attempt >= max_attempts:
        return "dead_letter"
    return _parallel_tool_sends(state) or "tool"


def route_after_approval(state: AgentState) -> str:
    """Route based on human approval decision.

    - If approved → "tool" (proceed with risky action)
    - If rejected → "clarify" (ask user for alternative)
    """
    approval = state.get("approval")
    if isinstance(approval, dict) and approval.get("approved") is True:
        return "tool"
    return "clarify"
