"""Controlled checkpoint selection, replay, and reviewable-content forks."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.types import StateSnapshot
from pydantic import BaseModel

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


class UnsafeReplayError(RuntimeError):
    """Raised when a replay/fork could execute a node outside the safe set."""


class CheckpointView(BaseModel):
    """Stable metadata passed to checkpoint predicates; state values stay private."""

    thread_id: str
    checkpoint_id: str
    next_nodes: tuple[str, ...]
    source: str = "unknown"
    step: int | None = None


REVIEWABLE_CONTENT_FIELDS = frozenset({"query", "final_answer", "pending_question"})

DEFAULT_SAFE_NEXT_NODES = frozenset({"finalize", "clarify", "dead_letter", "__end__"})


def _snapshot_view(snapshot: StateSnapshot) -> CheckpointView:
    configurable = snapshot.config.get("configurable", {})
    metadata = snapshot.metadata if isinstance(snapshot.metadata, Mapping) else {}
    step = metadata.get("step")
    return CheckpointView(
        thread_id=str(configurable.get("thread_id", "")),
        checkpoint_id=str(configurable.get("checkpoint_id", "")),
        next_nodes=tuple(str(node) for node in snapshot.next),
        source=str(metadata.get("source", "unknown")),
        step=step if isinstance(step, int) else None,
    )


def select_checkpoint(
    graph: CompiledStateGraph,
    thread_id: str,
    *,
    checkpoint_id: str | None = None,
    predicate: Callable[[CheckpointView], bool] | None = None,
) -> StateSnapshot:
    """Select one checkpoint by stable ID or an unambiguous metadata predicate."""
    if (checkpoint_id is None) == (predicate is None):
        raise ValueError("Provide exactly one of checkpoint_id or predicate.")
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    matches: list[StateSnapshot] = []
    for snapshot in graph.get_state_history(config):
        view = _snapshot_view(snapshot)
        if checkpoint_id is not None and view.checkpoint_id == checkpoint_id:
            matches.append(snapshot)
        elif predicate is not None and predicate(view):
            matches.append(snapshot)
    if not matches:
        raise LookupError("No checkpoint matched the requested identity.")
    if len(matches) != 1:
        raise LookupError("Checkpoint predicate matched more than one checkpoint.")
    return matches[0]


def _guard_next_nodes(snapshot: StateSnapshot, allowed_next_nodes: Iterable[str] | None) -> None:
    allowed = set(allowed_next_nodes or DEFAULT_SAFE_NEXT_NODES)
    blocked = sorted({str(node) for node in snapshot.next if str(node) not in allowed})
    if blocked:
        names = ", ".join(blocked)
        raise UnsafeReplayError(f"Checkpoint may execute unapproved node(s): {names}.")


def replay_checkpoint(
    graph: CompiledStateGraph,
    thread_id: str,
    checkpoint_id: str,
    *,
    allowed_next_nodes: Iterable[str] | None = None,
) -> dict[str, object]:
    """Replay a selected checkpoint only when every scheduled node is whitelisted."""
    snapshot = select_checkpoint(graph, thread_id, checkpoint_id=checkpoint_id)
    _guard_next_nodes(snapshot, allowed_next_nodes)
    return cast(dict[str, object], graph.invoke(None, config=snapshot.config))


def fork_checkpoint(
    graph: CompiledStateGraph,
    thread_id: str,
    checkpoint_id: str,
    *,
    updates: Mapping[str, object],
    allowed_next_nodes: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Create a branch that can change content, never graph control state."""
    invalid = sorted(set(updates) - REVIEWABLE_CONTENT_FIELDS)
    if invalid:
        names = ", ".join(invalid)
        raise ValueError(f"Fork updates must be reviewable content fields; rejected: {names}.")
    snapshot = select_checkpoint(graph, thread_id, checkpoint_id=checkpoint_id)
    _guard_next_nodes(snapshot, allowed_next_nodes)
    if "query" in updates and tuple(snapshot.next) != ("classify",):
        raise ValueError("A query fork is only safe before classification.")
    return cast(dict[str, Any], graph.update_state(snapshot.config, dict(updates)))
