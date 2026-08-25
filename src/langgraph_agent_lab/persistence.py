"""Checkpointer construction and lifecycle adapters.

``build_checkpointer`` remains the small core API used by the lab. Durable
backends own connections, so extensions must use ``open_checkpointer`` and
keep its context open for as long as the compiled graph is in use.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from typing import Any

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
            "SQLite owns a connection; use open_checkpointer('sqlite', absolute_path)."
        )
    if kind == "postgres":
        raise NotImplementedError(
            "Postgres owns a connection pool; use open_checkpointer('postgres', database_url)."
        )
    raise ValueError(f"Unknown checkpointer kind: {kind}")


def _sqlite_path(database_url: str | None) -> Path:
    """Validate an explicit, durable SQLite filesystem target."""
    if not database_url:
        raise ValueError("SQLite requires an explicit absolute database path.")
    if database_url == ":memory:" or database_url.startswith("file:"):
        raise ValueError("SQLite requires a durable filesystem path, not memory or a URI.")
    path = Path(database_url)
    if not path.is_absolute():
        raise ValueError("SQLite database path must be absolute.")
    if path.exists() and path.is_dir():
        raise ValueError("SQLite database path must name a file, not a directory.")
    if not path.parent.is_dir():
        raise ValueError("SQLite database parent directory must already exist.")
    return path.resolve()


@contextmanager
def open_checkpointer(kind: str = "memory", database_url: str | None = None) -> Iterator[Any]:
    """Open a checkpointer and close every backend-owned resource on exit.

    SQLite and Postgres packages are optional. Selecting an unavailable
    backend raises an actionable error without making core tests depend on it.
    """
    if kind == "none":
        yield None
        return
    if kind == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        yield MemorySaver()
        return
    if kind == "sqlite":
        path = _sqlite_path(database_url)
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError as exc:
            raise RuntimeError(
                "Install the project sqlite extra to use durable SQLite checkpoints."
            ) from exc

        with SqliteSaver.from_conn_string(str(path)) as saver:
            saver.setup()
            saver.conn.execute("PRAGMA journal_mode=WAL")
            yield saver
        return
    if kind == "postgres":
        if not database_url or not database_url.startswith(("postgres://", "postgresql://")):
            raise ValueError("Postgres requires an explicit postgres:// or postgresql:// URL.")
        try:
            postgres_module = import_module("langgraph.checkpoint.postgres")
            saver_factory = postgres_module.PostgresSaver
        except ImportError as exc:
            raise RuntimeError(
                "Install the project postgres extra to use durable Postgres checkpoints."
            ) from exc

        with saver_factory.from_conn_string(database_url) as saver:
            saver.setup()
            yield saver
        return
    raise ValueError(f"Unknown checkpointer kind: {kind}")
