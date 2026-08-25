"""CLI for the lab."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Annotated

import typer
import yaml
from langchain_core.runnables import RunnableConfig

from .graph import build_graph
from .metrics import MetricsReport, metric_from_state, summarize_metrics, write_metrics
from .persistence import build_checkpointer
from .report import write_report
from .scenarios import load_scenarios
from .state import initial_state

app = typer.Typer(no_args_is_help=True)


@app.command("run-scenarios")
def run_scenarios(
    config: Annotated[Path, typer.Option("--config")],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Run all grading scenarios and write metrics JSON."""
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    scenarios = load_scenarios(cfg["scenarios_path"])
    checkpointer = build_checkpointer(cfg.get("checkpointer", "memory"), cfg.get("database_url"))
    graph = build_graph(checkpointer=checkpointer)
    metrics = []
    audit_records: list[dict[str, object]] = []
    persistence_records: list[dict[str, object]] = []
    for scenario in scenarios:
        state = initial_state(scenario)
        run_config: RunnableConfig = {"configurable": {"thread_id": state["thread_id"]}}
        started_at = perf_counter()
        final_state = graph.invoke(state, config=run_config)
        latency_ms = round((perf_counter() - started_at) * 1000)
        # The graph output and its reducer-managed histories remain untouched.  Latency is
        # instrumentation belonging to the scenario metric, not a workflow state update.
        measured_state = {**final_state, "latency_ms": latency_ms}
        metrics.append(
            metric_from_state(
                measured_state,
                scenario.expected_route.value,
                scenario.requires_approval,
            )
        )
        for event in final_state.get("events", []) or []:
            audit_records.append(
                {
                    "scenario_id": scenario.id,
                    "thread_id": state["thread_id"],
                    **event,
                }
            )
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
    if cfg.get("report_path"):
        write_report(report, cfg["report_path"])
    typer.echo(f"Wrote metrics to {output}")


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
