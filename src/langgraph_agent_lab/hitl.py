"""Display-safe runner for a real LangGraph approval interrupt/resume cycle."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from threading import RLock
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.types import Interrupt, StateSnapshot
from pydantic import BaseModel, Field

from .graph import build_graph
from .persistence import build_checkpointer

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|token|secret|password|authorization)\b"
    r"\s*[:=]\s*(?:bearer\s+)?[^\s,;]+"
)
_CREDENTIAL_PATTERNS = (
    re.compile(r"(?i)\bbearer\s+[a-z0-9._~+/=-]{12,}"),
    re.compile(r"\b(?:sk|nvapi)-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b"),
)


class EventView(BaseModel):
    """Small event projection that intentionally omits raw metadata."""

    node: str
    event_type: str
    message: str


class ApprovalView(BaseModel):
    """Reviewer decision safe for display."""

    approved: bool
    reviewer: str
    comment: str


class TicketView(BaseModel):
    """Only the workflow fields an approval UI is allowed to render."""

    thread_id: str
    status: Literal["pending", "completed"]
    ticket: str
    proposed_action: str | None = None
    approval: ApprovalView | None = None
    events: list[EventView] = Field(default_factory=list)
    pending_question: str | None = None
    final_answer: str | None = None
    interrupt_id: str | None = None


def _safe_text(value: object, *, limit: int = 500) -> str:
    """Redact common credential assignments and bound display payloads."""
    text = str(value) if value is not None else ""
    text = _SECRET_ASSIGNMENT.sub(lambda match: f"{match.group(1)}=[REDACTED]", text)
    for pattern in _CREDENTIAL_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    text = " ".join(text.split())
    return text[:limit]


def _mapping(value: object) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, Mapping):
            return dumped
    return {}


def _event_views(values: Mapping[str, Any]) -> list[EventView]:
    raw_events = values.get("events", [])
    if not isinstance(raw_events, Sequence) or isinstance(raw_events, (str, bytes, bytearray)):
        return []
    projected: list[EventView] = []
    for raw_event in raw_events:
        event = _mapping(raw_event)
        node = _safe_text(event.get("node"), limit=80)
        event_type = _safe_text(event.get("event_type"), limit=80)
        if not node or not event_type:
            continue
        projected.append(
            EventView(
                node=node,
                event_type=event_type,
                message=_safe_text(event.get("message"), limit=300),
            )
        )
    return projected


def _approval_view(values: Mapping[str, Any]) -> ApprovalView | None:
    approval = _mapping(values.get("approval"))
    approved = approval.get("approved")
    if not isinstance(approved, bool):
        return None
    return ApprovalView(
        approved=approved,
        reviewer=_safe_text(approval.get("reviewer"), limit=100),
        comment=_safe_text(approval.get("comment"), limit=300),
    )


class HitlRunner:
    """Start and resume approval tickets without exposing raw graph state."""

    def __init__(self, graph: CompiledStateGraph | None = None) -> None:
        self._graph = graph or build_graph(checkpointer=build_checkpointer("memory"))
        self._lock = RLock()

    @staticmethod
    def _config(thread_id: str) -> RunnableConfig:
        return {
            "configurable": {
                "thread_id": thread_id,
                "approval_mode": "interrupt",
            }
        }

    @staticmethod
    def _active_approval(snapshot: StateSnapshot) -> Interrupt:
        interrupts = tuple(getattr(snapshot, "interrupts", ()))
        if not interrupts:
            raise RuntimeError("Thread has no active approval interrupt.")
        if len(interrupts) != 1:
            raise RuntimeError("Expected exactly one active approval interrupt.")

        interrupt_value = interrupts[0]
        interrupt_id = getattr(interrupt_value, "id", None)
        owners = [
            getattr(task, "name", "")
            for task in getattr(snapshot, "tasks", ())
            if any(
                nested is interrupt_value
                or (interrupt_id and getattr(nested, "id", None) == interrupt_id)
                for nested in getattr(task, "interrupts", ())
            )
        ]
        if owners != ["approval"]:
            raise RuntimeError("Active interrupt is not the single approval gate.")
        return interrupt_value

    def _view(self, thread_id: str) -> TicketView:
        snapshot = self._graph.get_state(self._config(thread_id))
        values = _mapping(getattr(snapshot, "values", {}))
        interrupts = tuple(getattr(snapshot, "interrupts", ()))
        active = self._active_approval(snapshot) if interrupts else None
        interrupt_id = getattr(active, "id", None) if active is not None else None
        return TicketView(
            thread_id=thread_id,
            status="pending" if active is not None else "completed",
            ticket=_safe_text(values.get("query", values.get("ticket", ""))),
            proposed_action=(
                _safe_text(values.get("proposed_action"), limit=500)
                if values.get("proposed_action") is not None
                else None
            ),
            approval=_approval_view(values),
            events=_event_views(values),
            pending_question=(
                _safe_text(values.get("pending_question"), limit=500)
                if values.get("pending_question") is not None
                else None
            ),
            final_answer=(
                _safe_text(values.get("final_answer"), limit=1000)
                if values.get("final_answer") is not None
                else None
            ),
            interrupt_id=str(interrupt_id) if interrupt_id else None,
        )

    def start(self, query: str) -> TicketView:
        """Start a new server-identified thread and pause at its approval gate."""
        normalized = query.strip()
        if not normalized:
            raise ValueError("Ticket query must not be empty.")
        thread_id = f"hitl-{uuid4().hex}"
        state = {
            "thread_id": thread_id,
            "scenario_id": thread_id,
            "query": normalized,
            "route": "",
            "risk_level": "unknown",
            "attempt": 0,
            "max_attempts": 3,
            "final_answer": None,
            "evaluation_result": None,
            "pending_question": None,
            "proposed_action": None,
            "approval": None,
            "messages": [],
            "tool_results": [],
            "errors": [],
            "events": [],
        }
        with self._lock:
            self._graph.invoke(state, config=self._config(thread_id))
            return self._view(thread_id)

    def resume(
        self,
        thread_id: str,
        *,
        approved: bool,
        reviewer: str,
        comment: str,
        interrupt_id: str | None = None,
    ) -> TicketView:
        """Apply one decision to the currently active interrupt on the same thread."""
        from langgraph.types import Command

        with self._lock:
            config = self._config(thread_id)
            snapshot = self._graph.get_state(config)
            active = self._active_approval(snapshot)
            active_id = getattr(active, "id", None)
            if interrupt_id is not None and interrupt_id != active_id:
                raise ValueError("Provided interrupt id does not match the active interrupt.")

            decision = {
                "approved": approved,
                "reviewer": _safe_text(reviewer, limit=100) or "human-reviewer",
                "comment": _safe_text(comment, limit=300),
            }
            resume_value: object = {str(active_id): decision} if active_id else decision
            self._graph.invoke(Command[str](resume=resume_value), config=config)
            return self._view(thread_id)
