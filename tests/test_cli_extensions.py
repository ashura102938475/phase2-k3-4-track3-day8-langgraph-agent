"""CLI integration contracts for opt-in extension execution."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

import langgraph_agent_lab.cli as cli
import langgraph_agent_lab.hitl as hitl
from langgraph_agent_lab.state import Route, Scenario


class _FakeGraph:
    def __init__(self) -> None:
        self.configs: list[dict[str, Any]] = []

    def invoke(self, state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        self.configs.append(config)
        return {
            **state,
            "route": Route.TOOL.value,
            "final_answer": "offline answer",
            "evaluation_source": "llm",
            "evaluation_reason": "Structured offline verdict.",
            "judge_calls": 1,
            "parallel_tool_results": [
                {"task": "account", "result": "account result"},
                {"task": "order", "result": "order result"},
            ],
            "events": [
                {
                    "node": "tool",
                    "event_type": "completed",
                    "message": "raw task response must not be persisted",
                    "metadata": {
                        "task": "account",
                        "secret": "sk-" + ("x" * 40),
                        "attempt": 0,
                    },
                    "scenario_id": "forged-scenario",
                    "thread_id": "forged-thread",
                },
                {
                    "node": "tool",
                    "event_type": "completed",
                    "message": "order",
                    "metadata": {"task": "order"},
                },
                {
                    "node": "evaluate",
                    "event_type": "completed",
                    "message": "judged",
                    "metadata": {"source": "llm"},
                },
                {
                    "node": "finalize",
                    "event_type": "completed",
                    "message": "done",
                    "metadata": {},
                }
            ],
        }

    def get_state_history(self, _config: dict[str, Any]) -> list[object]:
        return [object(), object()]


def test_run_scenarios_enables_configured_judge_and_forces_mock_approval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A shell interrupt flag must not pause the non-interactive scenario gate."""
    scenario = Scenario(
        id="extension-tool",
        query="look up two records",
        expected_route=Route.TOOL,
        tool_tasks=["order", "account"],
    )
    graph = _FakeGraph()
    build_calls: list[dict[str, object]] = []

    @contextmanager
    def fake_open_checkpointer(
        kind: str,
        database_url: str | None,
    ) -> Iterator[object]:
        assert kind == "memory"
        assert database_url is None
        yield object()

    def fake_build_graph(
        checkpointer: object,
        *,
        use_llm_judge: bool = False,
    ) -> _FakeGraph:
        build_calls.append(
            {"checkpointer": checkpointer, "use_llm_judge": use_llm_judge}
        )
        return graph

    monkeypatch.setattr(cli, "load_scenarios", lambda _path: [scenario])
    monkeypatch.setattr(cli, "open_checkpointer", fake_open_checkpointer, raising=False)
    monkeypatch.setattr(cli, "build_graph", fake_build_graph)
    monkeypatch.setenv("LANGGRAPH_INTERRUPT", "true")

    config_path = tmp_path / "lab.yaml"
    output_path = tmp_path / "metrics.json"
    audit_path = tmp_path / "audit.jsonl"
    persistence_path = tmp_path / "persistence.json"
    extension_path = tmp_path / "scenario-extensions.json"
    config_path.write_text(
        yaml.safe_dump(
            {
                "scenarios_path": "unused.jsonl",
                "checkpointer": "memory",
                "audit_path": str(audit_path),
                "persistence_evidence_path": str(persistence_path),
                "scenario_extension_evidence_path": str(extension_path),
                "extensions": {"llm_as_judge": True},
            }
        ),
        encoding="utf-8",
    )

    cli.run_scenarios(config_path, output_path)

    assert build_calls and build_calls[0]["use_llm_judge"] is True
    assert graph.configs == [
        {
            "configurable": {
                "thread_id": "thread-extension-tool",
                "approval_mode": "mock",
            }
        }
    ]
    metrics = json.loads(output_path.read_text(encoding="utf-8"))
    assert metrics["total_scenarios"] == 1
    assert json.loads(persistence_path.read_text(encoding="utf-8"))[
        "history_proven"
    ] is True
    extension_evidence = json.loads(extension_path.read_text(encoding="utf-8"))
    assert extension_evidence["llm_as_judge"] == {
        "enabled": True,
        "llm_event_count": 1,
        "fallback_event_count": 0,
        "policy_guard_event_count": 0,
    }
    assert extension_evidence["parallel_fanout"] == {
        "scenario_count": 1,
        "configured_task_count": 2,
        "tool_event_count": 2,
        "deterministic_result_count": 2,
    }
    audit_records = [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
    ]
    first_audit = audit_records[0]
    assert first_audit["scenario_id"] == "extension-tool"
    assert first_audit["thread_id"] == "thread-extension-tool"
    assert set(first_audit) == {
        "scenario_id",
        "thread_id",
        "node",
        "event_type",
        "metadata",
    }
    assert set(first_audit["metadata"]) == {"attempt", "task_fingerprint"}
    assert first_audit["metadata"]["attempt"] == 0
    assert len(first_audit["metadata"]["task_fingerprint"]) == 16
    serialized_audit = audit_path.read_text(encoding="utf-8")
    assert "raw task response" not in serialized_audit
    assert "forged-scenario" not in serialized_audit
    assert "forged-thread" not in serialized_audit
    assert "sk-" not in serialized_audit


def test_export_graph_writes_compiled_mermaid(tmp_path: Path) -> None:
    output_path = tmp_path / "actual-graph.mmd"

    cli.export_graph(output_path)

    mermaid = output_path.read_text(encoding="utf-8")
    assert "intake" in mermaid
    assert "finalize" in mermaid


def test_run_hitl_proof_writes_metadata_only_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeRunner:
        def start(self, _query: str) -> SimpleNamespace:
            return SimpleNamespace(
                thread_id="hitl-proof-thread",
                interrupt_id="interrupt-present",
                status="pending",
                events=[
                    SimpleNamespace(node="intake"),
                    SimpleNamespace(node="approval"),
                ],
            )

        def resume(
            self,
            thread_id: str,
            *,
            approved: bool,
            reviewer: str,
            comment: str,
            interrupt_id: str,
        ) -> SimpleNamespace:
            assert thread_id == "hitl-proof-thread"
            assert interrupt_id == "interrupt-present"
            assert approved is False
            assert reviewer and comment
            return SimpleNamespace(
                thread_id=thread_id,
                status="completed",
                events=[
                    SimpleNamespace(node="intake"),
                    SimpleNamespace(node="approval"),
                    SimpleNamespace(node="clarify"),
                    SimpleNamespace(node="finalize"),
                ],
            )

    monkeypatch.setattr(hitl, "HitlRunner", FakeRunner)
    output = tmp_path / "hitl-proof.json"

    cli.run_hitl_proof(output)

    evidence = json.loads(output.read_text(encoding="utf-8"))
    assert evidence == {
        "thread_id": "hitl-proof-thread",
        "same_thread_resumed": True,
        "interrupt_observed": True,
        "decision": "rejected",
        "pending_event_nodes": ["intake", "approval"],
        "resumed_event_nodes": ["intake", "approval", "clarify", "finalize"],
        "tool_called": False,
        "terminal": True,
    }
    serialized = output.read_text(encoding="utf-8")
    assert "query" not in serialized
    assert "comment" not in serialized
