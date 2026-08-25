"""Safety and boundary tests for the optional Streamlit HITL interface."""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
streamlit_app = importlib.import_module("apps.streamlit_app")


class _ModelView:
    def model_dump(self) -> dict[str, Any]:
        return {
            "thread_id": "thread-ui-7",
            "interrupt_id": "interrupt-ui-7",
            "status": "pending",
            "ticket": "Ticket T-7: cancel the duplicate order",
            "proposed_action": "Cancel order O-42",
            "approval": {
                "approved": False,
                "reviewer": "reviewer-9",
                "comment": "Needs a second check",
                "token": "APPROVAL_SECRET_DO_NOT_RENDER",
            },
            "events": [
                {
                    "node": "risky_action",
                    "event_type": "proposed",
                    "message": "EVENT_SECRET_DO_NOT_RENDER",
                    "metadata": {"environment": "ENV_SECRET_DO_NOT_RENDER"},
                },
                {"node": "approval", "event_type": "interrupted"},
            ],
            "errors": ["RAW_ERROR_DO_NOT_RENDER"],
            "raw_state": {"api_key": "API_SECRET_DO_NOT_RENDER"},
        }


@dataclass(frozen=True)
class _DataclassView:
    thread_id: str
    status: str
    ticket: str
    proposed_action: str
    approval: dict[str, Any] | None
    events: tuple[dict[str, str], ...]
    pending_question: str | None
    final_answer: str | None
    interrupt_id: str | None = None


class _CaptureStreamlit:
    def __init__(self) -> None:
        self.rendered: list[object] = []

    def subheader(self, value: object) -> None:
        self.rendered.append(value)

    def caption(self, value: object) -> None:
        self.rendered.append(value)

    def write(self, value: object) -> None:
        self.rendered.append(value)

    def table(self, value: object) -> None:
        self.rendered.append(value)


class _AppStreamlit(_CaptureStreamlit):
    def __init__(
        self,
        *,
        buttons: set[str] | None = None,
        fields: dict[str, str] | None = None,
        session_state: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.buttons = buttons or set()
        self.fields = fields or {}
        self.session_state = session_state or {}

    def set_page_config(self, **kwargs: object) -> None:
        self.rendered.append(("page_config", kwargs))

    def title(self, value: object) -> None:
        self.rendered.append(value)

    def text_area(
        self,
        _label: str,
        *,
        key: str,
        height: int | None = None,
    ) -> str:
        del height
        return self.fields.get(key, "")

    def text_input(self, _label: str, *, key: str) -> str:
        return self.fields.get(key, "")

    def button(self, _label: str, *, key: str) -> bool:
        return key in self.buttons

    def warning(self, value: object) -> None:
        self.rendered.append(value)

    def error(self, value: object) -> None:
        self.rendered.append(value)

    def success(self, value: object) -> None:
        self.rendered.append(value)


class _FakeRunner:
    def __init__(self) -> None:
        self.resume_calls: list[dict[str, object]] = []

    def start(self, query: str) -> _DataclassView:
        return _DataclassView(
            thread_id="thread-started",
            status="pending",
            ticket=f"Started: {query}",
            proposed_action="Cancel duplicate order",
            approval=None,
            events=({"node": "approval", "event_type": "interrupted"},),
            pending_question=None,
            final_answer=None,
            interrupt_id="interrupt-started",
        )

    def resume(
        self,
        thread_id: str,
        *,
        approved: bool,
        reviewer: str,
        comment: str,
        interrupt_id: str,
    ) -> _DataclassView:
        self.resume_calls.append(
            {
                "thread_id": thread_id,
                "approved": approved,
                "reviewer": reviewer,
                "comment": comment,
                "interrupt_id": interrupt_id,
            }
        )
        return _DataclassView(
            thread_id=thread_id,
            status="completed",
            ticket="Ticket resumed",
            proposed_action="Cancel duplicate order",
            approval={"approved": approved, "reviewer": reviewer, "comment": comment},
            events=({"node": "finalize", "event_type": "completed"},),
            pending_question=None,
            final_answer=f"Resumed {thread_id}: approved={approved}; {reviewer}; {comment}",
            interrupt_id=None,
        )


class _FailingRunner(_FakeRunner):
    def start(self, query: str) -> _DataclassView:
        del query
        raise RuntimeError("API_SECRET_FROM_RAW_EXCEPTION")


def test_safe_view_data_whitelists_model_fields_and_reduces_events() -> None:
    """Adding raw state or error fields to a view must not expose them in the UI."""
    assert streamlit_app.safe_view_data(_ModelView()) == {
        "thread_id": "thread-ui-7",
        "interrupt_id": "interrupt-ui-7",
        "status": "pending",
        "ticket": "Ticket T-7: cancel the duplicate order",
        "proposed_action": "Cancel order O-42",
        "approval": {
            "approved": False,
            "reviewer": "reviewer-9",
            "comment": "Needs a second check",
        },
        "events": [
            {"node": "risky_action", "event_type": "proposed"},
            {"node": "approval", "event_type": "interrupted"},
        ],
        "pending_question": None,
        "final_answer": None,
    }


def test_safe_view_data_accepts_dataclass_attributes() -> None:
    view = _DataclassView(
        thread_id="thread-ui-8",
        status="completed",
        ticket="Ticket T-8",
        proposed_action="No action",
        approval=None,
        events=({"node": "finalize", "event_type": "completed"},),
        pending_question=None,
        final_answer="Resolved safely",
    )

    assert streamlit_app.safe_view_data(view) == {
        "thread_id": "thread-ui-8",
        "interrupt_id": None,
        "status": "completed",
        "ticket": "Ticket T-8",
        "proposed_action": "No action",
        "approval": None,
        "events": [{"node": "finalize", "event_type": "completed"}],
        "pending_question": None,
        "final_answer": "Resolved safely",
    }


def test_render_view_shows_hitl_evidence_without_raw_details() -> None:
    st = _CaptureStreamlit()

    streamlit_app.render_view(st, _ModelView())

    rendered = repr(st.rendered)
    for expected in (
        "Ticket T-7: cancel the duplicate order",
        "Cancel order O-42",
        "Needs a second check",
        "risky_action",
        "approval",
        "interrupt-ui-7",
    ):
        assert expected in rendered
    for forbidden in (
        "APPROVAL_SECRET_DO_NOT_RENDER",
        "EVENT_SECRET_DO_NOT_RENDER",
        "ENV_SECRET_DO_NOT_RENDER",
        "RAW_ERROR_DO_NOT_RENDER",
        "API_SECRET_DO_NOT_RENDER",
    ):
        assert forbidden not in rendered


def test_run_app_starts_ticket_and_keeps_runner_view_in_session() -> None:
    st = _AppStreamlit(
        buttons={"start_ticket"},
        fields={"ticket_query": "  cancel the duplicate order  "},
    )

    streamlit_app.run_app(st, _FakeRunner())

    rendered = repr(st.rendered)
    assert "Started: cancel the duplicate order" in rendered
    assert streamlit_app.safe_view_data(st.session_state["hitl_view"])["thread_id"] == (
        "thread-started"
    )


def test_run_app_resumes_rejected_ticket_with_explicit_reviewer_fields() -> None:
    pending = _DataclassView(
        thread_id="thread-resume-3",
        status="pending",
        ticket="Ticket T-3",
        proposed_action="Refund order O-3",
        approval=None,
        events=({"node": "approval", "event_type": "interrupted"},),
        pending_question=None,
        final_answer=None,
        interrupt_id="interrupt-resume-3",
    )
    st = _AppStreamlit(
        buttons={"reject_action"},
        fields={"reviewer": "reviewer-alias", "approval_comment": "Policy mismatch"},
        session_state={"hitl_view": pending},
    )

    runner = _FakeRunner()

    streamlit_app.run_app(st, runner)

    rendered = repr(st.rendered)
    assert (
        "Resumed thread-resume-3: approved=False; reviewer-alias; Policy mismatch"
        in rendered
    )
    assert streamlit_app.safe_view_data(st.session_state["hitl_view"])["status"] == (
        "completed"
    )
    assert runner.resume_calls == [
        {
            "thread_id": "thread-resume-3",
            "approved": False,
            "reviewer": "reviewer-alias",
            "comment": "Policy mismatch",
            "interrupt_id": "interrupt-resume-3",
        }
    ]


def test_failed_start_discards_stale_ticket_before_approval_buttons() -> None:
    """A failed replacement start must not leave an old ticket resumable."""
    stale = _DataclassView(
        thread_id="thread-stale",
        status="pending",
        ticket="Old ticket",
        proposed_action="Old risky action",
        approval=None,
        events=({"node": "approval", "event_type": "interrupted"},),
        pending_question=None,
        final_answer=None,
        interrupt_id="interrupt-stale",
    )
    runner = _FailingRunner()
    st = _AppStreamlit(
        buttons={"start_ticket", "approve_action"},
        fields={"ticket_query": "Replacement ticket"},
        session_state={"hitl_view": stale},
    )

    streamlit_app.run_app(st, runner)

    assert "hitl_view" not in st.session_state
    assert runner.resume_calls == []
    assert "Old ticket" not in repr(st.rendered)


def test_blank_start_discards_stale_ticket_before_approval_buttons() -> None:
    """A blank replacement start must not leave an old ticket resumable."""
    stale = _DataclassView(
        thread_id="thread-stale",
        status="pending",
        ticket="Old ticket",
        proposed_action="Old risky action",
        approval=None,
        events=({"node": "approval", "event_type": "interrupted"},),
        pending_question=None,
        final_answer=None,
        interrupt_id="interrupt-stale",
    )
    runner = _FakeRunner()
    st = _AppStreamlit(
        buttons={"start_ticket", "approve_action"},
        fields={"ticket_query": "   "},
        session_state={"hitl_view": stale},
    )

    streamlit_app.run_app(st, runner)

    assert "hitl_view" not in st.session_state
    assert runner.resume_calls == []
    assert "Old ticket" not in repr(st.rendered)


def test_safe_view_data_redacts_standalone_credential_shaped_text() -> None:
    """A credential-shaped token in an otherwise allowed field must stay hidden."""
    fake_credential = "sk-testonly_abcdefghijklmnopqrstuv"
    view = _DataclassView(
        thread_id="thread-redaction",
        status="completed",
        ticket=f"Provider returned {fake_credential}",
        proposed_action="No action",
        approval={
            "approved": False,
            "reviewer": "reviewer",
            "comment": f"Remove Bearer {fake_credential}",
        },
        events=(
            {
                "node": "finalize",
                "event_type": f"completed-{fake_credential}",
            },
        ),
        pending_question=None,
        final_answer=f"Credential was {fake_credential}",
    )

    projected = streamlit_app.safe_view_data(view)

    assert fake_credential not in repr(projected)
    assert "[REDACTED]" in repr(projected)


def test_run_app_does_not_render_raw_runner_exceptions() -> None:
    st = _AppStreamlit(
        buttons={"start_ticket"},
        fields={"ticket_query": "Ticket with a provider failure"},
    )

    streamlit_app.run_app(st, _FailingRunner())

    rendered = repr(st.rendered)
    assert "Workflow request failed; inspect the server logs." in rendered
    assert "API_SECRET_FROM_RAW_EXCEPTION" not in rendered


def test_streamlit_entrypoint_loads_with_optional_runtime() -> None:
    """The checked-in app must execute under Streamlit, not only under the fake UI."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(PROJECT_ROOT / "apps" / "streamlit_app.py")).run(
        timeout=15
    )

    assert not app.exception
    assert any(item.value == "LangGraph ticket approval" for item in app.title)
    assert any(item.label == "Ticket" for item in app.text_area)
    assert any(item.label == "Start ticket" for item in app.button)
