"""Simple Flask web server for the LangGraph Agent Lab UI.

Run with:
    python -m web.app
    # or
    flask --app web.app run --port 5000
"""

from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from flask import Flask, jsonify, request, send_from_directory

# Ensure the src package is importable when running from the repo root.
_root = Path(__file__).resolve().parent.parent
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))

from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.persistence import build_checkpointer
from langgraph_agent_lab.state import AgentState, Scenario, initial_state

app = Flask(__name__, static_folder=None)

_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        checkpointer = build_checkpointer("memory")
        _graph = build_graph(checkpointer=checkpointer)
    return _graph


def _build_steps(final_state: AgentState, query: str) -> list[dict]:
    """Reconstruct an ordered step list from the event history."""
    events: list[dict] = []
    for ev in final_state.get("events", []):
        if isinstance(ev, dict):
            events.append(ev)

    if not events:
        return []

    node_order = [
        "intake", "classify", "answer", "tool", "evaluate",
        "clarify", "risky_action", "approval", "retry",
        "dead_letter", "finalize",
    ]

    graph_edges: dict[str, list[str]] = {
        "START": ["intake"],
        "intake": ["classify"],
        "classify": ["answer", "tool", "clarify", "risky_action", "retry"],
        "tool": ["evaluate"],
        "evaluate": ["answer", "retry"],
        "retry": ["tool", "dead_letter"],
        "risky_action": ["approval"],
        "approval": ["tool", "clarify"],
        "answer": ["finalize"],
        "clarify": ["finalize"],
        "dead_letter": ["finalize"],
        "finalize": ["END"],
    }

    visited: list[str] = ["START", "intake"]
    event_nodes = [e.get("node", "") for e in events]

    for en in event_nodes:
        if en and en not in visited:
            visited.append(en)

    if "finalize" not in visited:
        visited.append("finalize")
    if "END" not in visited:
        visited.append("END")

    steps: list[dict] = []
    for i in range(len(visited) - 1):
        src = visited[i]
        dst = visited[i + 1]
        step: dict = {"from": src, "to": dst, "node": dst}

        matching = [e for e in events if e.get("node") == dst]
        if matching:
            step["event"] = matching[-1]

        if dst in ("answer", "tool", "clarify", "risky_action", "retry",
                    "evaluate", "approval", "dead_letter", "finalize"):
            if step.get("event"):
                step["event"]["node"] = dst

        steps.append(step)

    return steps


@app.route("/")
def index():
    html_path = Path(__file__).parent / "index.html"
    return html_path.read_text(encoding="utf-8")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Empty query"}), 400

    graph = _get_graph()
    scenario = Scenario(
        id=f"web-{uuid4().hex[:8]}",
        query=query,
        expected_route="simple",
    )
    state = initial_state(scenario)
    run_config = {"configurable": {"thread_id": state["thread_id"]}}

    try:
        started = perf_counter()
        final_state = graph.invoke(state, config=run_config)
        latency_ms = round((perf_counter() - started) * 1000)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    steps = _build_steps(final_state, query)

    return jsonify({
        "answer": final_state.get("final_answer", ""),
        "route": final_state.get("route", ""),
        "risk_level": final_state.get("risk_level", ""),
        "steps": steps,
        "latency_ms": latency_ms,
        "events": final_state.get("events", []),
    })


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
