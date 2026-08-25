"""Defensive routing cases outside the normal typed-state path."""

from langgraph_agent_lab.routing import route_after_classify


def test_malformed_unhashable_route_uses_safe_default() -> None:
    assert route_after_classify({"route": []}) == "answer"
