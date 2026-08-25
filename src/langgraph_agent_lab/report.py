"""Markdown report generator for scenario metrics and workflow evidence."""

from __future__ import annotations

from pathlib import Path

from .metrics import MetricsReport

_GENERATED_BLOCK_NAMES = ("METRICS_SUMMARY", "SCENARIO_RESULTS")


def render_report(metrics: MetricsReport, existing_report: str | None = None) -> str:
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
    metrics_summary = f"""| Metric | Value |
|---|---:|
| Total scenarios | {metrics.total_scenarios} |
| Success rate | {metrics.success_rate:.0%} |
| Average nodes visited | {metrics.avg_nodes_visited:.2f} |
| Total retries | {metrics.total_retries} |
| Total approval-node visits | {metrics.total_interrupts} |
| Resume success demonstrated | {resume_status} |"""
    scenario_results = f"""{scenario_header}
|---|---|---|---|---:|---:|---:|---|---:|---|
{scenario_rows}"""

    if existing_report is not None:
        has_generated_blocks = _validate_generated_blocks(existing_report)
        if has_generated_blocks:
            refreshed = _replace_generated_block(
                existing_report, "METRICS_SUMMARY", metrics_summary
            )
            return _replace_generated_block(refreshed, "SCENARIO_RESULTS", scenario_results)
        if existing_report:
            return _append_generated_blocks(
                existing_report, metrics_summary, scenario_results
            )

    return f"""# Day 23 — Track 3 — LangGraph Agentic Orchestration Lab Report

## 1. Student metadata

| Field | Value |
|---|---|
| Name | [Student name] |
| Student ID | [Student ID] |
| Repository / commit | [Repository URL or commit] |
| Date | [YYYY-MM-DD] |

## 2. Metrics summary

<!-- BEGIN GENERATED:METRICS_SUMMARY -->
{metrics_summary}
<!-- END GENERATED:METRICS_SUMMARY -->

## 3. Scenario results

<!-- BEGIN GENERATED:SCENARIO_RESULTS -->
{scenario_results}
<!-- END GENERATED:SCENARIO_RESULTS -->

The CLI also emits a metadata-only projection of the reducer-backed event stream to
`outputs/audit_events.jsonl` and the per-thread checkpoint history proof to
`outputs/persistence_evidence.json`. The projection omits messages and raw task text;
these artifacts still provide inspectable route, retry, approval, and persistence evidence.

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

`thread_id`, `scenario_id`, `query`, `route`, `risk_level`, `attempt`, `max_attempts`,
`evaluation_result`, `evaluation_reason`, `evaluation_source`, `judge_calls`,
`pending_question`, `proposed_action`, `approval`, `tool_tasks`, `active_tool_task`, and
`final_answer` are overwrite fields: each represents the current workflow fact.
`parallel_tool_results` uses a deterministic custom reducer that retains the highest
attempt per task in canonical order.
`messages`, `tool_results`, `errors`, and `events` use the list-add reducer, so every
node contributes an append-only audit/history entry instead of replacing prior
evidence. This reducer choice makes node counts, retries, approval visits, and failure
details measurable.

## 6. Failure analysis

1. **Tool retry and dead-letter.** The failure starts when classification selects
   `error`, or when `tool_results[-1]` contains `ERROR`. The `route`, failed tool
   event, `evaluation_result="needs_retry"`, retry event, and appended error expose
   it. The graph moves through `retry`, then selects `tool` or `dead_letter` from
   the updated attempt counter. Attempts increase monotonically and exhaustion ends
   at `dead_letter → finalize → END`. Residual risk remains because the core uses a
   mock tool and substring evaluator rather than production timeout, backoff,
   idempotency, and circuit-breaker controls.
2. **Risky approval rejection.** The failure starts after `risky_action` proposes a
   side effect and the approval mapping is rejected, missing, or malformed. The
   approval mapping and event expose the decision. Only `approved is True` routes
   to `tool`; every other value routes `clarify → finalize → END`, so the tool is
   never reached. The contract test verifies this containment. Residual risk remains
   because core mock mode auto-approves and does not authenticate a reviewer or
   persist a decision timestamp. A rejection is a **safe workflow completion**. It
   does not mean the action was approved or executed; it means the graph stopped
   the action and returned a user-facing next step.

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

The single highest-priority production step is durable checkpoint storage with an
automated process-restart replay test. It comes first because `MemorySaver` loses
the verified state history when the process exits.
"""


def _validate_generated_blocks(report: str) -> bool:
    """Return whether every generated block exists, rejecting unsafe layouts."""
    spans: list[tuple[int, int]] = []
    marker_count = 0
    for name in _GENERATED_BLOCK_NAMES:
        start = f"<!-- BEGIN GENERATED:{name} -->"
        end = f"<!-- END GENERATED:{name} -->"
        start_count = report.count(start)
        end_count = report.count(end)
        marker_count += start_count + end_count
        if start_count not in (0, 1) or end_count not in (0, 1):
            raise ValueError("Report contains malformed generated-block markers")
        if start_count != end_count:
            raise ValueError("Report contains malformed generated-block markers")
        if start_count:
            start_index = report.index(start)
            end_index = report.index(end)
            if start_index >= end_index:
                raise ValueError("Report contains malformed generated-block markers")
            spans.append((start_index, end_index + len(end)))

    if marker_count == 0:
        return False
    if len(spans) != len(_GENERATED_BLOCK_NAMES):
        raise ValueError("Report contains an incomplete set of generated-block markers")
    ordered_spans = sorted(spans)
    if any(
        left[1] > right[0]
        for left, right in zip(ordered_spans, ordered_spans[1:], strict=False)
    ):
        raise ValueError("Report contains overlapping generated-block markers")
    return True


def _append_generated_blocks(
    report: str, metrics_summary: str, scenario_results: str
) -> str:
    """Add updateable generated sections without changing existing report bytes."""
    if report.endswith("\n\n"):
        separator = ""
    elif report.endswith("\n"):
        separator = "\n"
    else:
        separator = "\n\n"
    return f"""{report}{separator}## Generated metrics summary

<!-- BEGIN GENERATED:METRICS_SUMMARY -->
{metrics_summary}
<!-- END GENERATED:METRICS_SUMMARY -->

## Generated scenario results

<!-- BEGIN GENERATED:SCENARIO_RESULTS -->
{scenario_results}
<!-- END GENERATED:SCENARIO_RESULTS -->
"""


def _replace_generated_block(report: str, name: str, content: str) -> str:
    """Replace one marked block while retaining all surrounding narrative."""
    start = f"<!-- BEGIN GENERATED:{name} -->"
    end = f"<!-- END GENERATED:{name} -->"
    before, separator, remainder = report.partition(start)
    if not separator:
        raise ValueError(f"Missing generated-block start marker: {name}")
    _, separator, after = remainder.partition(end)
    if not separator:
        raise ValueError(f"Missing generated-block end marker: {name}")
    return f"{before}{start}\n{content.rstrip()}\n{end}{after}"


def _table_value(value: object) -> str:
    """Keep generated Markdown tables stable when scenario text contains punctuation."""
    normalized = str(value).replace("\r", " ").replace("\n", " ")
    return normalized.replace("\\", "\\\\").replace("|", "\\|")


def write_report(metrics: MetricsReport, output_path: str | Path) -> None:
    """Write the rendered report to a file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_report = path.read_bytes().decode("utf-8") if path.exists() else None
    rendered = render_report(metrics, existing_report=existing_report)
    path.write_bytes(rendered.encode("utf-8"))
