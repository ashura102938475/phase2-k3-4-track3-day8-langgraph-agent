"""Contract tests for the LangGraph state schema.

These tests intentionally inspect the public TypedDict annotations rather than
implementation source.  Audit collections must be append-only, while the
workflow gates and user-facing values represent the current value.
"""

from operator import add
from typing import Annotated, get_args, get_origin, get_type_hints

from langgraph_agent_lab.state import AgentState

STATE_HINTS = get_type_hints(AgentState, include_extras=True)


def test_agent_state_exposes_current_value_fields() -> None:
    """Workflow gates and pending user values are scalar/current-value fields."""
    current_value_fields = {
        "evaluation_result",
        "pending_question",
        "proposed_action",
        "approval",
    }

    assert current_value_fields <= STATE_HINTS.keys()
    for field_name in current_value_fields:
        annotation = STATE_HINTS[field_name]
        assert get_origin(annotation) is not Annotated


def test_agent_state_audit_collections_retain_append_reducers() -> None:
    """Message, result, error, and event histories merge with operator.add."""
    append_only_fields = {"messages", "tool_results", "errors", "events"}

    for field_name in append_only_fields:
        annotation = STATE_HINTS[field_name]
        assert get_origin(annotation) is Annotated
        _, reducer = get_args(annotation)
        assert reducer is add
