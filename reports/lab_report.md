# Day 08 LangGraph Agent Lab Report

## 1. Student metadata

| Field | Value |
|---|---|
| Name | [Student name] |
| Repository / commit | [Repository URL or commit] |
| Date | [YYYY-MM-DD] |

## 2. Metrics summary

| Metric | Value |
|---|---:|
| Total scenarios | 7 |
| Success rate | 100% |
| Average nodes visited | 6.43 |
| Total retries | 3 |
| Total approval-node visits | 2 |
| Resume success demonstrated | no |

## 3. Scenario results

| Scenario | Expected route | Actual route | Success | Nodes | Retries | Approval visits | Approval observed | Latency (ms) | Errors |
|---|---|---|---|---:|---:|---:|---|---:|---|
| S01_simple | simple | simple | yes | 4 | 0 | 0 | no | 4233 | — |
| S02_tool | tool | tool | yes | 6 | 0 | 0 | no | 8883 | — |
| S03_missing | missing_info | missing_info | yes | 4 | 0 | 0 | no | 660 | — |
| S04_risky | risky | risky | yes | 8 | 0 | 1 | yes | 32342 | — |
| S05_error | error | error | yes | 10 | 2 | 0 | no | 7879 | Retry 1 requested after an unsatisfactory tool result.; Retry 2 requested after an unsatisfactory tool result. |
| S06_delete | risky | risky | yes | 8 | 0 | 1 | yes | 4489 | — |
| S07_dead_letter | error | error | yes | 5 | 1 | 0 | no | 10878 | Retry 1 requested after an unsatisfactory tool result. |

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
is **no** because no replay or resume demonstration is recorded here.

## 8. Extension status

No extension evidence is recorded by this generated report. Candidate next steps
are durable SQLite/Postgres checkpoints, graph visualization, or a real
human-in-the-loop interrupt/resume interface with an explicit demonstration.

## 9. Improvement plan

First, add durable checkpoint storage and an automated state-history replay test.
Next, replace the mock approval path with a reviewed UI/API workflow, instrument
tool and LLM latency separately, and add alerting for repeated dead-letter events.
