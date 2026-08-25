"""CLI for the lab."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Annotated

import typer
import yaml
from langchain_core.runnables import RunnableConfig

from .diagram import write_mermaid
from .graph import build_graph
from .metrics import MetricsReport, metric_from_state, summarize_metrics, write_metrics
from .persistence import open_checkpointer
from .report import write_report
from .scenarios import load_scenarios
from .state import initial_state

app = typer.Typer(no_args_is_help=True)

_AUDIT_NODES = frozenset(
    {
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
    }
)
_AUDIT_EVENT_TYPES = frozenset(
    {"completed", "failed", "pending_approval", "retrying"}
)
_AUDIT_SOURCES = frozenset({"llm", "heuristic_fallback", "policy_guard"})


def _audit_identifier(value: str) -> str:
    """Keep ordinary IDs readable and fingerprint unexpected untrusted values."""
    if value and len(value) <= 100 and all(
        character.isalnum() or character in "._:-" for character in value
    ):
        return value
    return f"sha256:{sha256(value.encode('utf-8')).hexdigest()[:16]}"


def _project_audit_event(
    event: object,
    *,
    scenario_id: str,
    thread_id: str,
) -> dict[str, object] | None:
    """Project an event onto metadata that proves routing without raw payloads."""
    if not isinstance(event, dict):
        return None
    node = event.get("node")
    event_type = event.get("event_type")
    if node not in _AUDIT_NODES or event_type not in _AUDIT_EVENT_TYPES:
        return None
    safe_metadata: dict[str, object] = {}
    metadata = event.get("metadata")
    if isinstance(metadata, dict):
        source = metadata.get("source")
        if source in _AUDIT_SOURCES:
            safe_metadata["source"] = source
        attempt = metadata.get("attempt")
        if isinstance(attempt, int) and not isinstance(attempt, bool) and attempt >= 0:
            safe_metadata["attempt"] = attempt
        task = metadata.get("task")
        if isinstance(task, str) and task:
            safe_metadata["task_fingerprint"] = sha256(
                task.encode("utf-8")
            ).hexdigest()[:16]
    return {
        "scenario_id": _audit_identifier(scenario_id),
        "thread_id": _audit_identifier(thread_id),
        "node": node,
        "event_type": event_type,
        "metadata": safe_metadata,
    }


@app.command("run-scenarios")
def run_scenarios(
    config: Annotated[Path, typer.Option("--config")],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Run all grading scenarios and write metrics JSON."""
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    scenarios = load_scenarios(cfg["scenarios_path"])
    checkpointer_kind = cfg.get("checkpointer", "memory")
    database_url = cfg.get("database_url")
    if not isinstance(checkpointer_kind, str):
        raise typer.BadParameter("checkpointer must be a string")
    if database_url is not None and not isinstance(database_url, str):
        raise typer.BadParameter("database_url must be a string")
    extension_config = cfg.get("extensions", {})
    use_llm_judge = (
        isinstance(extension_config, dict)
        and extension_config.get("llm_as_judge") is True
    )
    metrics = []
    audit_records: list[dict[str, object]] = []
    persistence_records: list[dict[str, object]] = []
    llm_judge_events = 0
    judge_fallback_events = 0
    policy_guard_events = 0
    fanout_scenarios = 0
    configured_fanout_tasks = 0
    fanout_tool_events = 0
    deterministic_fanout_results = 0
    with open_checkpointer(checkpointer_kind, database_url) as checkpointer:
        graph = build_graph(checkpointer=checkpointer, use_llm_judge=use_llm_judge)
        for scenario in scenarios:
            state = initial_state(scenario)
            run_config: RunnableConfig = {
                "configurable": {
                    "thread_id": state["thread_id"],
                    "approval_mode": "mock",
                }
            }
            started_at = perf_counter()
            final_state = graph.invoke(state, config=run_config)
            latency_ms = round((perf_counter() - started_at) * 1000)
            # Reducer histories remain untouched. Latency belongs to instrumentation,
            # not to workflow state.
            measured_state = {**final_state, "latency_ms": latency_ms}
            metrics.append(
                metric_from_state(
                    measured_state,
                    scenario.expected_route.value,
                    scenario.requires_approval,
                )
            )
            for event in final_state.get("events", []) or []:
                if not isinstance(event, dict):
                    continue
                metadata = event.get("metadata", {}) if isinstance(event, dict) else {}
                source = metadata.get("source") if isinstance(metadata, dict) else None
                if event.get("node") == "evaluate" and source == "llm":
                    llm_judge_events += 1
                elif event.get("node") == "evaluate" and source == "heuristic_fallback":
                    judge_fallback_events += 1
                elif event.get("node") == "evaluate" and source == "policy_guard":
                    policy_guard_events += 1
                if event.get("node") == "tool" and isinstance(metadata, dict):
                    if isinstance(metadata.get("task"), str):
                        fanout_tool_events += 1
                projected_event = _project_audit_event(
                    event,
                    scenario_id=scenario.id,
                    thread_id=state["thread_id"],
                )
                if projected_event is not None:
                    audit_records.append(projected_event)
            if len(scenario.tool_tasks) > 1:
                fanout_scenarios += 1
                configured_fanout_tasks += len(scenario.tool_tasks)
                parallel_results = final_state.get("parallel_tool_results", [])
                if isinstance(parallel_results, list):
                    deterministic_fanout_results += len(parallel_results)
            if checkpointer is not None:
                history = list(graph.get_state_history(run_config))
                persistence_records.append(
                    {
                        "scenario_id": scenario.id,
                        "thread_id": state["thread_id"],
                        "history_snapshots": len(history),
                    }
                )
    report = summarize_metrics(metrics)
    write_metrics(report, output)
    audit_path = Path(cfg.get("audit_path", output.with_name("audit_events.jsonl")))
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in audit_records),
        encoding="utf-8",
    )
    persistence_path = Path(
        cfg.get("persistence_evidence_path", output.with_name("persistence_evidence.json"))
    )
    persistence_path.parent.mkdir(parents=True, exist_ok=True)
    persistence_path.write_text(
        json.dumps(
            {
                "backend": cfg.get("checkpointer", "memory"),
                "records": persistence_records,
                "history_proven": bool(persistence_records)
                and all(
                    isinstance(history_count := record.get("history_snapshots"), int)
                    and history_count > 1
                    for record in persistence_records
                ),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    extension_path_value = cfg.get("scenario_extension_evidence_path")
    if extension_path_value is not None:
        extension_path = Path(extension_path_value)
        extension_path.parent.mkdir(parents=True, exist_ok=True)
        extension_path.write_text(
            json.dumps(
                {
                    "llm_as_judge": {
                        "enabled": use_llm_judge,
                        "llm_event_count": llm_judge_events,
                        "fallback_event_count": judge_fallback_events,
                        "policy_guard_event_count": policy_guard_events,
                    },
                    "parallel_fanout": {
                        "scenario_count": fanout_scenarios,
                        "configured_task_count": configured_fanout_tasks,
                        "tool_event_count": fanout_tool_events,
                        "deterministic_result_count": deterministic_fanout_results,
                    },
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    if cfg.get("report_path"):
        write_report(report, cfg["report_path"])
    typer.echo(f"Wrote metrics to {output}")


@app.command("export-graph")
def export_graph(output: Annotated[Path, typer.Option("--output")]) -> None:
    """Export Mermaid text from the graph that is actually compiled."""
    written = write_mermaid(output)
    typer.echo(f"Wrote Mermaid graph to {written}")


@app.command("run-hitl-proof")
def run_hitl_proof(
    output: Annotated[Path, typer.Option("--output")],
    query: Annotated[str, typer.Option("--query")] = (
        "Refund a synthetic test order after support review"
    ),
) -> None:
    """Run one real reject/resume cycle and write metadata-only proof."""
    from .hitl import HitlRunner

    runner = HitlRunner()
    pending = runner.start(query)
    if pending.status != "pending" or not pending.interrupt_id:
        raise typer.BadParameter("The proof query did not reach an approval interrupt")
    resumed = runner.resume(
        pending.thread_id,
        approved=False,
        reviewer="extension-reviewer",
        comment="Rejected by the automated extension proof.",
        interrupt_id=pending.interrupt_id,
    )
    pending_nodes = [event.node for event in pending.events]
    resumed_nodes = [event.node for event in resumed.events]
    evidence = {
        "thread_id": pending.thread_id,
        "same_thread_resumed": resumed.thread_id == pending.thread_id,
        "interrupt_observed": bool(pending.interrupt_id),
        "decision": "rejected",
        "pending_event_nodes": pending_nodes,
        "resumed_event_nodes": resumed_nodes,
        "tool_called": "tool" in resumed_nodes,
        "terminal": resumed.status == "completed" and "finalize" in resumed_nodes,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    typer.echo(f"Wrote HITL proof to {output}")


@app.command("validate-metrics")
def validate_metrics(metrics: Annotated[Path, typer.Option("--metrics")]) -> None:
    """Validate metrics JSON schema for grading."""
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    report = MetricsReport.model_validate(payload)
    if report.total_scenarios < 6:
        raise typer.BadParameter("Expected at least 6 scenarios")
    typer.echo(f"Metrics valid. success_rate={report.success_rate:.2%}")


if __name__ == "__main__":
    app()
