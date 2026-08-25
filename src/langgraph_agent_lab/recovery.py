"""Deterministic durable-recovery probe and sanitized evidence helpers."""

from __future__ import annotations

import re
from operator import add
from typing import TYPE_CHECKING, Annotated, Literal, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.types import Checkpointer, Command
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


class ApprovalProbeState(TypedDict, total=False):
    """Small serializable state used only to prove checkpoint recovery."""

    ticket: str
    approval: dict[str, object]
    completed: bool
    events: Annotated[list[dict[str, object]], add]


class RecoveryEvidence(BaseModel):
    """Metadata-only proof: no query, action, event message, or raw state."""

    backend: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,31}$")
    thread_id: str = Field(pattern=r"^[A-Za-z0-9._-]{1,128}$")
    checkpoint_count: int = Field(ge=0)
    active_interrupt_count: int = Field(ge=0)
    status: Literal["pending", "completed", "incomplete"]
    history_present: bool


_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def _safe_identifier(value: str, label: str) -> str:
    if not _SAFE_IDENTIFIER.fullmatch(value):
        raise ValueError(f"{label} must contain only safe identifier characters.")
    return value


def build_approval_probe_graph(checkpointer: Checkpointer) -> CompiledStateGraph:
    """Compile an offline approval interrupt used for crash/recovery evidence."""
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import interrupt

    def approval(state: ApprovalProbeState) -> dict:
        response = interrupt(
            {
                "proposed_action": f"Review ticket: {state.get('ticket', '')}",
                "instruction": "Approve or reject.",
            }
        )
        decision = response if isinstance(response, dict) else {}
        return {
            "approval": {
                "approved": decision.get("approved") is True,
                "reviewer": str(decision.get("reviewer", "human-reviewer")),
            },
            "events": [{"node": "approval", "event_type": "completed"}],
        }

    def finalize(_state: ApprovalProbeState) -> dict:
        return {
            "completed": True,
            "events": [{"node": "finalize", "event_type": "completed"}],
        }

    workflow: StateGraph[
        ApprovalProbeState, None, ApprovalProbeState, ApprovalProbeState
    ] = StateGraph(ApprovalProbeState)
    # LangGraph's current overload resolves ContextT to Never for a local
    # TypedDict graph even though these callables match the runtime contract.
    workflow.add_node("approval", approval)  # type: ignore[arg-type]
    workflow.add_node("finalize", finalize)  # type: ignore[arg-type]
    workflow.add_edge(START, "approval")
    workflow.add_edge("approval", "finalize")
    workflow.add_edge("finalize", END)
    return workflow.compile(checkpointer=checkpointer)


def collect_recovery_evidence(
    graph: CompiledStateGraph,
    thread_id: str,
    *,
    backend: str,
) -> RecoveryEvidence:
    """Return bounded checkpoint metadata without copying persisted content."""
    safe_thread = _safe_identifier(thread_id, "thread_id")
    safe_backend = _safe_identifier(backend, "backend")
    config: RunnableConfig = {"configurable": {"thread_id": safe_thread}}
    snapshot = graph.get_state(config)
    history = list(graph.get_state_history(config))
    active_count = len(tuple(getattr(snapshot, "interrupts", ())))
    values = getattr(snapshot, "values", {})
    completed = isinstance(values, dict) and values.get("completed") is True
    status: Literal["pending", "completed", "incomplete"]
    if active_count:
        status = "pending"
    elif completed:
        status = "completed"
    else:
        status = "incomplete"
    return RecoveryEvidence(
        backend=safe_backend,
        thread_id=safe_thread,
        checkpoint_count=len(history),
        active_interrupt_count=active_count,
        status=status,
        history_present=bool(history),
    )


def approval_resume_command(
    graph: CompiledStateGraph,
    thread_id: str,
    *,
    approved: bool,
    reviewer: str,
) -> Command[str]:
    """Build a decision command keyed to the one persisted approval interrupt."""
    safe_thread = _safe_identifier(thread_id, "thread_id")
    safe_reviewer = _safe_identifier(reviewer, "reviewer")
    config: RunnableConfig = {"configurable": {"thread_id": safe_thread}}
    snapshot = graph.get_state(config)
    interrupts = tuple(getattr(snapshot, "interrupts", ()))
    if len(interrupts) != 1:
        raise RuntimeError("Recovery requires exactly one active approval interrupt.")
    active = interrupts[0]
    interrupt_id = getattr(active, "id", None)
    decision = {"approved": approved, "reviewer": safe_reviewer}
    return Command[str](resume={str(interrupt_id): decision} if interrupt_id else decision)
