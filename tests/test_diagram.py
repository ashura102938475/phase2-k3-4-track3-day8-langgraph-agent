"""Contract tests for exporting the compiled LangGraph as Mermaid text."""

from __future__ import annotations

from pathlib import Path

import pytest

import langgraph_agent_lab.diagram as diagram


class _FakeDrawableGraph:
    def draw_mermaid(self) -> str:
        return "graph TD\r\n    ticket --> finalize   \r\n\r\n"


class _FakeCompiledGraph:
    def get_graph(self) -> _FakeDrawableGraph:
        return _FakeDrawableGraph()


@pytest.fixture
def compiled_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(diagram, "build_graph", _FakeCompiledGraph)


def test_render_mermaid_uses_compiled_graph_and_normalizes_text(
    compiled_graph: None,
) -> None:
    """A hand-written diagram cannot replace the compiled graph export."""
    assert diagram.render_mermaid() == "graph TD\n    ticket --> finalize\n"


def test_write_mermaid_creates_parent_and_writes_deterministic_utf8(
    tmp_path: Path,
    compiled_graph: None,
) -> None:
    output = tmp_path / "nested" / "workflow.mmd"

    written = diagram.write_mermaid(output)

    assert written == output
    assert output.read_bytes() == b"graph TD\n    ticket --> finalize\n"


def test_real_graph_mermaid_export_is_repeatable_and_contains_all_nodes() -> None:
    """The integration boundary remains deterministic for the actual compiled graph."""
    first = diagram.render_mermaid()
    second = diagram.render_mermaid()

    assert first == second
    assert first.endswith("\n")
    for node in (
        "intake",
        "classify",
        "answer",
        "tool",
        "evaluate",
        "clarify",
        "risky_action",
        "approval",
        "retry",
        "dead_letter",
        "finalize",
    ):
        assert node in first
