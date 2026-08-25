"""Checkpointer adapter."""

from __future__ import annotations

from langgraph.types import Checkpointer


def build_checkpointer(kind: str = "memory", database_url: str | None = None) -> Checkpointer:
    """Return a LangGraph checkpointer.

    The core workflow supports MemorySaver. SQLite/Postgres remain explicit
    extension backends so CI never depends on an external durable service.

    For SQLite:
    - pip install langgraph-checkpoint-sqlite
    - Use SqliteSaver with sqlite3.connect() and WAL mode
    - See: https://langchain-ai.github.io/langgraph/how-tos/persistence/
    """
    if kind == "none":
        return None
    if kind == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()
    if kind == "sqlite":
        raise NotImplementedError(
            "SQLite is an optional extension. Install langgraph-checkpoint-sqlite "
            "and configure SqliteSaver before selecting this backend."
        )
    if kind == "postgres":
        raise NotImplementedError("Postgres is an optional durable-checkpoint extension.")
    raise ValueError(f"Unknown checkpointer kind: {kind}")
