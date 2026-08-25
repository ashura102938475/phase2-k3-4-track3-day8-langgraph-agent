"""Thin, display-safe Streamlit interface for HITL approval and resume."""

from __future__ import annotations

import re
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Protocol, cast

_CREDENTIAL_PATTERNS = (
    re.compile(r"(?i)\bbearer\s+[a-z0-9._~+/=-]{12,}"),
    re.compile(r"\b(?:sk|nvapi)-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b"),
)


class StreamlitLike(Protocol):
    """Small Streamlit surface used by this UI and its dependency-free tests."""

    session_state: MutableMapping[str, object]

    def set_page_config(self, **kwargs: object) -> None: ...

    def title(self, value: object) -> None: ...

    def subheader(self, value: object) -> None: ...

    def caption(self, value: object) -> None: ...

    def write(self, value: object) -> None: ...

    def table(self, value: object) -> None: ...

    def text_area(
        self,
        label: str,
        *,
        key: str,
        height: int | None = None,
    ) -> str: ...

    def text_input(self, label: str, *, key: str) -> str: ...

    def button(self, label: str, *, key: str) -> bool: ...

    def warning(self, value: object) -> None: ...

    def error(self, value: object) -> None: ...

    def success(self, value: object) -> None: ...


class HitlRunnerLike(Protocol):
    """Boundary consumed by the optional UI."""

    def start(self, query: str) -> object:
        """Start a ticket and return its display-safe view."""
        ...

    def resume(
        self,
        thread_id: str,
        *,
        approved: bool,
        reviewer: str,
        comment: str,
        interrupt_id: str,
    ) -> object:
        """Resume an interrupted ticket with the reviewer's decision."""
        ...


def _read_field(value: object, name: str) -> object:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _model_payload(view: object) -> object:
    model_dump = getattr(view, "model_dump", None)
    if callable(model_dump):
        payload = model_dump()
        if isinstance(payload, Mapping):
            return payload
    return view


def _safe_text(value: object, *, limit: int = 1_000) -> str | None:
    if not isinstance(value, str):
        return None
    text = value
    for pattern in _CREDENTIAL_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text[:limit]


def _safe_approval(value: object) -> dict[str, object] | None:
    if value is None:
        return None
    payload = _model_payload(value)
    approved = _read_field(payload, "approved")
    if not isinstance(approved, bool):
        return None
    return {
        "approved": approved,
        "reviewer": _safe_text(_read_field(payload, "reviewer"), limit=100),
        "comment": _safe_text(_read_field(payload, "comment"), limit=300),
    }


def _safe_events(value: object) -> list[dict[str, str]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    events: list[dict[str, str]] = []
    for event in value:
        payload = _model_payload(event)
        node = _safe_text(_read_field(payload, "node"), limit=80)
        event_type = _safe_text(_read_field(payload, "event_type"), limit=80)
        if node is not None and event_type is not None:
            events.append({"node": node, "event_type": event_type})
    return events


def safe_view_data(view: object) -> dict[str, object]:
    """Project a runner view onto the only fields the UI is allowed to render."""
    payload = _model_payload(view)
    return {
        "thread_id": _safe_text(_read_field(payload, "thread_id"), limit=200),
        "interrupt_id": _safe_text(_read_field(payload, "interrupt_id"), limit=200),
        "status": _safe_text(_read_field(payload, "status"), limit=20),
        "ticket": _safe_text(_read_field(payload, "ticket"), limit=500),
        "proposed_action": _safe_text(
            _read_field(payload, "proposed_action"), limit=500
        ),
        "approval": _safe_approval(_read_field(payload, "approval")),
        "events": _safe_events(_read_field(payload, "events")),
        "pending_question": _safe_text(
            _read_field(payload, "pending_question"), limit=500
        ),
        "final_answer": _safe_text(_read_field(payload, "final_answer"), limit=1_000),
    }


def render_view(st: StreamlitLike, view: object) -> None:
    """Render the safe HITL projection without forwarding raw state or errors."""
    data = safe_view_data(view)

    st.subheader("Ticket")
    st.write(data["ticket"] or "Not available")
    if data["thread_id"]:
        st.caption(f"Thread: {data['thread_id']}")
    if data["interrupt_id"]:
        st.caption(f"Approval interrupt: {data['interrupt_id']}")

    st.subheader("Proposed action")
    st.write(data["proposed_action"] or "No risky action proposed")

    st.subheader("Approval result")
    approval = data["approval"]
    if isinstance(approval, dict):
        st.write(approval)
    else:
        st.write("Pending reviewer decision")

    if data["pending_question"]:
        st.write(data["pending_question"])
    if data["final_answer"]:
        st.write(data["final_answer"])

    st.subheader("Event trail")
    events = data["events"]
    if events:
        st.table(events)
    else:
        st.caption("No workflow events yet")


def run_app(st: StreamlitLike, runner: HitlRunnerLike) -> None:
    """Run one Streamlit render cycle against the explicit HITL runner boundary."""
    st.set_page_config(page_title="LangGraph HITL Review", page_icon="✅")
    st.title("LangGraph ticket approval")

    query = st.text_area("Ticket", key="ticket_query", height=120)
    if st.button("Start ticket", key="start_ticket"):
        st.session_state.pop("hitl_view", None)
        normalized_query = query.strip()
        if not normalized_query:
            st.warning("Enter a ticket before starting the workflow.")
            return
        else:
            try:
                st.session_state["hitl_view"] = runner.start(normalized_query)
            except Exception:
                st.error("Workflow request failed; inspect the server logs.")
                return

    view = st.session_state.get("hitl_view")
    if view is None:
        return

    render_view(st, view)
    data = safe_view_data(view)
    if data["status"] != "pending":
        return

    reviewer = st.text_input("Reviewer alias", key="reviewer").strip()
    comment = st.text_area("Decision comment", key="approval_comment").strip()
    approved: bool | None = None
    if st.button("Approve", key="approve_action"):
        approved = True
    elif st.button("Reject", key="reject_action"):
        approved = False

    if approved is None:
        return
    thread_id = data["thread_id"]
    if not isinstance(thread_id, str) or not thread_id:
        st.error("This ticket has no resumable thread identifier.")
        return
    interrupt_id = data["interrupt_id"]
    if not isinstance(interrupt_id, str) or not interrupt_id:
        st.error("This ticket has no resumable approval interrupt identifier.")
        return

    try:
        resumed = runner.resume(
            thread_id,
            approved=approved,
            reviewer=reviewer,
            comment=comment,
            interrupt_id=interrupt_id,
        )
    except Exception:
        st.error("Workflow request failed; inspect the server logs.")
        return

    st.session_state["hitl_view"] = resumed
    st.success("Reviewer decision applied.")
    render_view(st, resumed)


def main() -> None:
    """Load optional dependencies and retain the runner across Streamlit reruns."""
    try:
        import streamlit as st
    except ImportError as exc:
        raise RuntimeError("Install Streamlit to run apps/streamlit_app.py") from exc

    from langgraph_agent_lab.hitl import HitlRunner

    if "hitl_runner" not in st.session_state:
        st.session_state["hitl_runner"] = HitlRunner()
    run_app(cast(StreamlitLike, st), st.session_state["hitl_runner"])


if __name__ == "__main__":
    main()
