"""Behavioral tests for the workflow nodes.

The LLM tests replace only the provider boundary (``get_llm``); node behavior remains
observable through the partial state updates they return.
"""

from __future__ import annotations

from copy import deepcopy

import pytest

from langgraph_agent_lab import llm, nodes


class _Message:
    def __init__(self, content: str) -> None:
        self.content = content


class _Classifier:
    def __init__(self, route: str, risk_level: str) -> None:
        self.route = route
        self.risk_level = risk_level


class _FakeLLM:
    """Small provider fake: structured classification and plain answer calls."""

    def __init__(
        self,
        classification: _Classifier | None,
        answer: str = "grounded answer",
        openai_api_base: str | None = None,
    ) -> None:
        self.classification = classification
        self.answer = answer
        self.openai_api_base = openai_api_base
        self.structured_kwargs: dict[str, object] = {}
        self.last_prompt = ""

    def with_structured_output(self, _schema: object, **kwargs: object) -> _FakeLLM:
        self.structured_kwargs = kwargs
        return self

    def invoke(self, prompt: object) -> _Classifier | _Message:
        self.last_prompt = str(prompt)
        if self.classification is not None:
            result, self.classification = self.classification, None
            return result
        return _Message(self.answer)


def test_intake_returns_partial_update_and_does_not_mutate_input() -> None:
    state = {"query": "  hello world  ", "events": [{"node": "prior"}], "messages": []}
    before = deepcopy(state)

    update = nodes.intake_node(state)

    assert update["query"] == "hello world"
    assert update.keys() == {"query", "messages", "events"}
    assert update["events"][0]["node"] == "intake"
    assert state == before


def test_tool_update_is_append_only_and_returns_transient_error() -> None:
    state = {"route": "error", "attempt": 0, "tool_results": [], "events": [{"node": "prior"}]}
    before = deepcopy(state)

    update = nodes.tool_node(state)

    assert update.keys() == {"tool_results", "events"}
    assert len(update["tool_results"]) == 1
    assert "ERROR" in update["tool_results"][0]
    assert update["events"][0]["node"] == "tool"
    assert update["events"][0]["message"]
    assert state == before


def test_tool_succeeds_after_transient_error_window() -> None:
    update = nodes.tool_node({"route": "error", "attempt": 2, "tool_results": [], "events": []})

    assert "ERROR" not in update["tool_results"][0]
    assert update["events"][0]["node"] == "tool"


@pytest.mark.parametrize(
    ("result", "expected"),
    [("ERROR: temporary outage", "needs_retry"), ("tool result: 42", "success")],
)
def test_evaluate_sets_retry_gate_from_latest_tool_result(result: str, expected: str) -> None:
    update = nodes.evaluate_node({"tool_results": [result], "events": []})

    assert update["evaluation_result"] == expected
    assert update.keys() == {"evaluation_result", "events"}


def test_retry_increments_attempt_and_logs_error() -> None:
    update = nodes.retry_or_fallback_node({"attempt": 1, "errors": [], "events": []})

    assert update["attempt"] == 2
    assert len(update["errors"]) == 1
    assert update["errors"][0]
    assert update["events"][0]["node"] == "retry"


def test_classify_uses_structured_llm_result(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeLLM(_Classifier("risky", "high"))
    monkeypatch.setattr(nodes, "get_llm", lambda: fake, raising=False)
    monkeypatch.setattr(llm, "get_llm", lambda: fake)

    update = nodes.classify_node({"query": "delete the production database", "events": []})

    assert update["route"] == "risky"
    assert update["risk_level"] == "high"
    assert update["events"][0]["node"] == "classify"
    assert fake.structured_kwargs == {}
    assert "JSON object" in fake.last_prompt
    assert "unresolved reference" in fake.last_prompt


def test_nvidia_classifier_uses_json_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeLLM(
        _Classifier("tool", "low"),
        openai_api_base="https://integrate.api.nvidia.com/v1",
    )
    monkeypatch.setattr(nodes, "get_llm", lambda: fake)

    update = nodes.classify_node({"query": "look up order 42", "events": []})

    assert update["route"] == "tool"
    assert fake.structured_kwargs == {"method": "json_mode"}


def test_answer_uses_llm_content_and_returns_partial_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeLLM(None, answer="The weather tool reports 22C.")
    monkeypatch.setattr(nodes, "get_llm", lambda: fake, raising=False)
    monkeypatch.setattr(llm, "get_llm", lambda: fake)

    update = nodes.answer_node(
        {"query": "what is the weather?", "tool_results": ["22C"], "events": []}
    )

    assert update["final_answer"] == "The weather tool reports 22C."
    assert update.keys() == {"final_answer", "events"}


def test_clarification_sets_specific_pending_question_and_answer() -> None:
    update = nodes.ask_clarification_node({"query": "book a flight", "events": []})

    assert update["pending_question"]
    assert update["final_answer"]
    assert update["pending_question"] in update["final_answer"]
    assert update["events"][0]["node"] == "clarify"


def test_risky_action_describes_proposal_for_approval() -> None:
    update = nodes.risky_action_node({"query": "delete the production database", "events": []})

    assert update["proposed_action"]
    assert "delete" in update["proposed_action"].lower()
    assert update["events"][0]["node"] == "risky_action"


def test_approval_defaults_to_mock_approved_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LANGGRAPH_INTERRUPT", raising=False)
    update = nodes.approval_node({"proposed_action": "delete production data", "events": []})

    assert update["approval"]["approved"] is True
    assert update["approval"]["reviewer"]
    assert update["events"][0]["node"] == "approval"


def test_dead_letter_sets_explanatory_final_answer() -> None:
    update = nodes.dead_letter_node(
        {"attempt": 3, "max_attempts": 3, "errors": ["timeout"], "events": []}
    )

    assert update["final_answer"]
    answer = update["final_answer"].lower()
    assert "could not" in answer or "unable" in answer
    assert update["events"][0]["node"] == "dead_letter"


def test_finalize_emits_workflow_finished_event() -> None:
    update = nodes.finalize_node({"events": []})

    assert update.keys() == {"events"}
    assert update["events"] == [
        {
            "node": "finalize",
            "event_type": "completed",
            "message": "workflow finished",
            "latency_ms": 0,
            "metadata": {},
        }
    ]
