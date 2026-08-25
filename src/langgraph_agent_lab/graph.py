"""Graph construction.

This module is intentionally import-safe. It imports LangGraph only inside the builder so unit tests
that check schema/metrics can run even if students are still debugging graph wiring.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .state import AgentState

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph
    from langgraph.types import Checkpointer


def build_graph(checkpointer: Checkpointer = None) -> CompiledStateGraph:
    """Build and compile the LangGraph workflow.
    """
    from langgraph.graph import END, START, StateGraph

    # Import modules rather than binding functions at module import time.  This keeps the
    # builder import-safe and lets callers replace individual node implementations in tests.
    from . import nodes, routing

    workflow = StateGraph(AgentState)
    workflow.add_node("intake", nodes.intake_node)
    workflow.add_node("classify", nodes.classify_node)
    workflow.add_node("answer", nodes.answer_node)
    workflow.add_node("tool", nodes.tool_node)
    workflow.add_node("evaluate", nodes.evaluate_node)
    workflow.add_node("clarify", nodes.ask_clarification_node)
    workflow.add_node("risky_action", nodes.risky_action_node)
    workflow.add_node("approval", nodes.approval_node)
    workflow.add_node("retry", nodes.retry_or_fallback_node)
    workflow.add_node("dead_letter", nodes.dead_letter_node)
    workflow.add_node("finalize", nodes.finalize_node)

    # Fixed edges: every terminal path converges on finalize before END.
    workflow.add_edge(START, "intake")
    workflow.add_edge("intake", "classify")
    workflow.add_edge("answer", "finalize")
    workflow.add_edge("tool", "evaluate")
    workflow.add_edge("clarify", "finalize")
    workflow.add_edge("risky_action", "approval")
    workflow.add_edge("dead_letter", "finalize")
    workflow.add_edge("finalize", END)

    workflow.add_conditional_edges(
        "classify",
        routing.route_after_classify,
        {
            "answer": "answer",
            "tool": "tool",
            "clarify": "clarify",
            "risky_action": "risky_action",
            "retry": "retry",
        },
    )
    workflow.add_conditional_edges(
        "evaluate",
        routing.route_after_evaluate,
        {"answer": "answer", "retry": "retry"},
    )
    workflow.add_conditional_edges(
        "retry",
        routing.route_after_retry,
        {"tool": "tool", "dead_letter": "dead_letter"},
    )
    workflow.add_conditional_edges(
        "approval",
        routing.route_after_approval,
        {"tool": "tool", "clarify": "clarify"},
    )

    return workflow.compile(checkpointer=checkpointer)
