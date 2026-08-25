"""Routing functions for conditional edges.

Each function takes AgentState and returns a string — the name of the next node.
These strings MUST match node names registered in graph.py.
"""

from __future__ import annotations

from .state import AgentState


def route_after_classify(state: AgentState) -> str:
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
    return destinations.get(route, "answer") if isinstance(route, str) else "answer"


def route_after_evaluate(state: AgentState) -> str:
    """Decide if tool result is satisfactory or needs retry.

    This is the 'done?' check that creates the retry loop —
    a key LangGraph advantage over linear LCEL chains.

    - If evaluation_result == "needs_retry" → "retry"
    - Otherwise → "answer"
    """
    return "retry" if state.get("evaluation_result") == "needs_retry" else "answer"


def route_after_retry(state: AgentState) -> str:
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
    return "tool" if attempt < max_attempts else "dead_letter"


def route_after_approval(state: AgentState) -> str:
    """Route based on human approval decision.

    - If approved → "tool" (proceed with risky action)
    - If rejected → "clarify" (ask user for alternative)
    """
    approval = state.get("approval")
    if isinstance(approval, dict) and approval.get("approved") is True:
        return "tool"
    return "clarify"
