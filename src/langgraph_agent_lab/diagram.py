"""Deterministic Mermaid export for the compiled LangGraph workflow."""

from __future__ import annotations

from pathlib import Path

from .graph import build_graph


def render_mermaid() -> str:
    """Render Mermaid text from the graph that the application actually compiles."""
    source = build_graph().get_graph().draw_mermaid()
    lines = source.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    normalized = [line.rstrip() for line in lines]
    while normalized and not normalized[-1]:
        normalized.pop()
    return "\n".join(normalized) + "\n"


def write_mermaid(output_path: str | Path) -> Path:
    """Write the compiled graph's normalized Mermaid source as UTF-8."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_mermaid(), encoding="utf-8")
    return path
