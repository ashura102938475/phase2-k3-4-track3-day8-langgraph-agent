"""Markdown report generator for scenario metrics and workflow evidence."""

from __future__ import annotations

from pathlib import Path

from .metrics import MetricsReport


def render_report(metrics: MetricsReport) -> str:
    """Render a deterministic Markdown report from measured scenario metrics."""
    scenario_rows = "\n".join(
        "| {id} | {expected} | {actual} | {success} | {nodes} | {retries} | "
        "{interrupts} | {approval} | {latency} | {errors} |".format(
            id=_table_value(item.scenario_id),
            expected=_table_value(item.expected_route),
            actual=_table_value(item.actual_route or "—"),
            success="yes" if item.success else "no",
            nodes=item.nodes_visited,
            retries=item.retry_count,
            interrupts=item.interrupt_count,
            approval="yes" if item.approval_observed else "no",
            latency=item.latency_ms,
            errors=_table_value("; ".join(item.errors) if item.errors else "—"),
        )
        for item in metrics.scenario_metrics
    )
    if not scenario_rows:
        scenario_rows = "| — | — | — | — | 0 | 0 | 0 | no | 0 | — |"

    scenario_header = (
        "| Scenario | Expected route | Actual route | Success | Nodes | Retries | "
        "Approval visits | Approval observed | Latency (ms) | Errors |"
    )
    resume_status = "yes" if metrics.resume_success else "no"
    return f"""# Day 08 LangGraph Agent Lab Report

## 1. Student metadata

| Field | Value |
|---|---|
| Name | [Student name] |
| Repository / commit | [Repository URL or commit] |
| Date | [YYYY-MM-DD] |

## 2. Metrics summary

| Metric | Value |
|---|---:|
| Total scenarios | {metrics.total_scenarios} |
| Success rate | {metrics.success_rate:.0%} |
| Average nodes visited | {metrics.avg_nodes_visited:.2f} |
| Total retries | {metrics.total_retries} |
| Total approval-node visits | {metrics.total_interrupts} |
| Resume success demonstrated | {resume_status} |

## 3. Scenario results

{scenario_header}
|---|---|---|---|---:|---:|---:|---|---:|---|
{scenario_rows}

The CLI also emits the complete reducer-backed audit event stream to
`outputs/audit_events.jsonl` and the per-thread checkpoint history proof to
`outputs/persistence_evidence.json`. These artifacts are the inspectable evidence
behind the node, retry, approval, and persistence claims above.

## 4. Architecture

The workflow is an 11-node `StateGraph`: `intake`, `classify`, `answer`, `tool`,
`evaluate`, `clarify`, `risky_action`, `approval`, `retry`, `dead_letter`, and
`finalize`. Eight fixed edges connect `START → intake → classify`, the processing
steps, and every terminal branch through `finalize → END`. Four conditional maps
select the route after classification, tool evaluation, retry, and approval. The
`route` field is assigned by `classify` and is preserved through `finalize`; retry
and approval routing use separate state fields rather than overwriting that input
classification.

## 5. State and reducers

`query`, `route`, `risk_level`, `attempt`, `max_attempts`, `evaluation_result`,
`pending_question`, `proposed_action`, `approval`, and `final_answer` are overwrite
fields: each represents the current workflow fact. `messages`, `tool_results`,
`errors`, and `events` use the list-add reducer, so every node contributes an
append-only audit/history entry instead of replacing prior evidence. This reducer
choice makes node counts, retries, approval visits, and failure details measurable.

## 6. Failure analysis

1. **Tool retry and dead-letter.** When evaluation sees an error result, it sends
   the run to `retry`. The bounded retry map compares `attempt` with
   `max_attempts`; exhausted runs go to `dead_letter`, produce an explanatory final
   answer, and still finalize. This prevents an unbounded tool loop while retaining
   the errors and retry events used by the metrics.
2. **Risky approval rejection.** A risky request first creates a proposed action
   and reaches `approval`. A rejected or missing approval routes to `clarify`, not
   the tool, so no risky operation is executed. The resulting clarification and
   approval evidence make the rejection visible rather than reporting a false
   successful action. When the expected route, output/clarification, and approval
   gate contracts hold, that rejection is a **safe workflow completion**. It does not mean
   the risky action was approved or executed; it means the workflow safely
   stopped the action and returned a user-facing next step.

## 7. Persistence and recovery caveat

The metrics do not record the active checkpointer backend, so this report cannot
infer which one was used. Configuration plus `thread_id`/state-history evidence is
required to demonstrate the active checkpointer and any recovery claim. If the
configured backend is `MemorySaver`, its checkpoints and state history exist only
for the current process and are not durable persistence. The core configuration in
`configs/lab.yaml` selects `memory`; the automated contract test invokes a run with
thread ID `contract-history`, reads `get_state_history()`, asserts multiple snapshots,
and verifies every snapshot carries that same thread ID. This proves in-process state
history for the supported core backend. It does not prove process-restart recovery.
This report does not claim real interrupt/resume or crash recovery. `resume_success`
is **{resume_status}** because no replay or resume demonstration is recorded here.

## 8. Extension status

No extension evidence is recorded by this generated report. Candidate next steps
are durable SQLite/Postgres checkpoints, graph visualization, or a real
human-in-the-loop interrupt/resume interface with an explicit demonstration.

## 9. Improvement plan

First, add durable checkpoint storage and an automated state-history replay test.
Next, replace the mock approval path with a reviewed UI/API workflow, instrument
tool and LLM latency separately, and add alerting for repeated dead-letter events.
"""


def _table_value(value: object) -> str:
    """Keep generated Markdown tables stable when scenario text contains punctuation."""
    normalized = str(value).replace("\r", " ").replace("\n", " ")
    return normalized.replace("\\", "\\\\").replace("|", "\\|")


def write_report(metrics: MetricsReport, output_path: str | Path) -> None:
    """Write the rendered report to a file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(metrics), encoding="utf-8")
