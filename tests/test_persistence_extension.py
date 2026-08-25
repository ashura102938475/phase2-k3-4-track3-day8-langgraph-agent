"""Tests for optional durable checkpoint lifecycle and recovery evidence."""

from __future__ import annotations

import importlib
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

import langgraph_agent_lab.persistence as persistence


def _recovery_module() -> ModuleType:
    spec = importlib.util.find_spec("langgraph_agent_lab.recovery")
    assert spec is not None, "the recovery extension has not been implemented"
    return importlib.import_module("langgraph_agent_lab.recovery")


def _require_sqlite() -> None:
    pytest.importorskip(
        "langgraph.checkpoint.sqlite",
        reason="install the project sqlite extra to run durable recovery tests",
    )


def test_open_checkpointer_preserves_memory_core_contract() -> None:
    """Dropping MemorySaver support while adding durable backends breaks this test."""
    assert hasattr(persistence, "open_checkpointer"), "missing checkpointer lifecycle API"

    with persistence.open_checkpointer("memory") as checkpointer:
        assert checkpointer is not None
        assert checkpointer.__class__.__name__ in {"InMemorySaver", "MemorySaver"}

    assert persistence.build_checkpointer("none") is None
    assert persistence.build_checkpointer("memory") is not None


@pytest.mark.parametrize("unsafe_path", [None, ":memory:", "relative.sqlite", "file:data.db"])
def test_sqlite_requires_an_explicit_absolute_durable_path(unsafe_path: str | None) -> None:
    """Silently selecting an ephemeral/ambiguous SQLite target breaks this test."""
    _require_sqlite()
    assert hasattr(persistence, "open_checkpointer"), "missing checkpointer lifecycle API"

    with pytest.raises(ValueError, match="SQLite"):
        with persistence.open_checkpointer("sqlite", unsafe_path):
            pass


def test_sqlite_checkpoint_survives_abrupt_process_exit(tmp_path: Path) -> None:
    """Keeping checkpoints only in process memory or an uncommitted connection breaks this test."""
    _require_sqlite()
    recovery = _recovery_module()
    database_path = tmp_path / "durable.sqlite"
    thread_id = "durable-crash-proof"
    child_code = """
import os
import sys
from langgraph_agent_lab.persistence import open_checkpointer
from langgraph_agent_lab.recovery import build_approval_probe_graph

database_path, thread_id = sys.argv[1:]
manager = open_checkpointer("sqlite", database_path)
checkpointer = manager.__enter__()
graph = build_approval_probe_graph(checkpointer)
graph.invoke(
    {"ticket": "offline durable probe", "events": []},
    config={"configurable": {"thread_id": thread_id}},
)
os._exit(23)
"""

    crashed = subprocess.run(
        [sys.executable, "-c", child_code, str(database_path), thread_id],
        check=False,
        capture_output=True,
        text=True,
    )
    assert crashed.returncode == 23, crashed.stderr

    with persistence.open_checkpointer("sqlite", str(database_path)) as checkpointer:
        graph = recovery.build_approval_probe_graph(checkpointer)
        before = recovery.collect_recovery_evidence(graph, thread_id, backend="sqlite")
        resumed = graph.invoke(
            recovery.approval_resume_command(
                graph,
                thread_id,
                approved=True,
                reviewer="recovery-test",
            ),
            config={"configurable": {"thread_id": thread_id}},
        )
        after = recovery.collect_recovery_evidence(graph, thread_id, backend="sqlite")

    assert before.model_dump() == {
        "backend": "sqlite",
        "thread_id": thread_id,
        "checkpoint_count": before.checkpoint_count,
        "active_interrupt_count": 1,
        "status": "pending",
        "history_present": True,
    }
    assert before.checkpoint_count >= 2
    assert resumed["completed"] is True
    assert after.status == "completed"
    assert after.active_interrupt_count == 0
    assert set(after.model_dump()) == {
        "backend",
        "thread_id",
        "checkpoint_count",
        "active_interrupt_count",
        "status",
        "history_present",
    }
