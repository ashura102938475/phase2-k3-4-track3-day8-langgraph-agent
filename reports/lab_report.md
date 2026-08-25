# Day 23 — Track 3 — LangGraph Agentic Orchestration Lab Report

## 1. Student

| Field | Value |
|---|---|
| Name | NGUYỄN ANH TRÀ |
| Student ID | 2A202601735 |
| Repository | https://github.com/ashura102938475/phase2-k3-4-track3-day8-langgraph-agent |
| Audited implementation commit | `cc07854` |
| Date | 2026-08-25 |

No API key, environment dump, credential, or unrelated personal identifier is included in
this report.

## 2. Architecture

The workflow is an 11-node `StateGraph`:

1. `intake` normalizes the request.
2. `classify` selects `simple`, `tool`, `missing_info`, `risky`, or `error` with
   structured LLM output.
3. `answer` produces an LLM response grounded in the query, tool results, and approval.
4. `tool` executes the lab's mock tool and records its result.
5. `evaluate` classifies tool evidence as `success` or `needs_retry`; the core scenario
   run uses the deterministic latest-result check, while the separate extension config
   selects the bounded structured LLM judge.
6. `clarify` asks for missing information or a safe alternative.
7. `risky_action` prepares, but does not execute, a consequential action.
8. `approval` records the reviewer decision before any risky tool execution.
9. `retry` increments the bounded attempt counter and records the failure.
10. `dead_letter` converts an exhausted retry path into an explicit terminal response.
11. `finalize` emits the final audit event before `END`.

The eight fixed edges are `START → intake`, `intake → classify`, `answer → finalize`,
`tool → evaluate`, `clarify → finalize`, `risky_action → approval`,
`dead_letter → finalize`, and `finalize → END`. Four conditional edge functions select
the next node:

| Router | Possible next nodes |
|---|---|
| `route_after_classify` | `answer`, `tool`, `clarify`, `risky_action`, `retry` |
| `route_after_evaluate` | `answer`, `retry` |
| `route_after_retry` | `tool`, `dead_letter` |
| `route_after_approval` | `tool`, `clarify` |

Termination is explicit. Normal answers, clarifications, and dead-letter outcomes converge
on `finalize → END`. The only cycle is `tool → evaluate → retry → tool`, and
`route_after_retry` leaves it when `attempt >= max_attempts`. Missing or malformed retry
limits fail closed to `dead_letter`, so they cannot create an unbounded loop.

Parallel lookup work does not add registered nodes or static edges. When a non-risky
scenario supplies more than one independent `tool_tasks` entry, the existing routers emit
multiple `Send("tool", ...)` branches. Their custom reducer merges results canonically and
the existing `tool → evaluate` edge acts as the fan-in barrier. Risky actions are never
fanned out. `outputs/graph.mmd`, exported from `compiled.get_graph()`, independently shows
the same 11 application nodes and 19 compiled edge instances (including conditional
destinations).

## 3. State schema

Current-value fields use the default overwrite behavior. Audit/history fields use list-add,
while parallel branch results use a deterministic validation/deduplication/sort reducer so
concurrent completion order cannot affect the merged state.

| Field | Update behavior | Reason |
|---|---|---|
| `thread_id` | overwrite | One current checkpointer identity for the run. |
| `scenario_id` | overwrite | One current scenario identity for metrics and evidence. |
| `query` | overwrite | `intake` owns the normalized current request. |
| `route` | overwrite | The classifier selects one current route. |
| `risk_level` | overwrite | Only the current risk decision drives the gate. |
| `attempt` | overwrite | The retry router needs the latest bounded counter. |
| `max_attempts` | overwrite | One active retry limit controls termination. |
| `final_answer` | overwrite | The latest terminal user-facing response wins. |
| `evaluation_result` | overwrite | The latest tool verdict drives evaluation routing. |
| `evaluation_reason` | overwrite | Retain the bounded reason for the latest judge verdict. |
| `evaluation_source` | overwrite | Distinguish an LLM verdict from deterministic fallback. |
| `judge_calls` | overwrite | Enforce the latest workflow-level judge call budget. |
| `pending_question` | overwrite | Only the current clarification is actionable. |
| `proposed_action` | overwrite | Approval reviews the current proposed side effect. |
| `approval` | overwrite | Routing consumes the current reviewer decision. |
| `tool_tasks` | overwrite | Hold the current bounded set of independent read-only work. |
| `active_tool_task` | overwrite | Carry one branch-local task into the shared `tool` node. |
| `parallel_tool_results` | deterministic custom reducer | Validate and sort branches, keeping the highest attempt per task with a stable tie-break so completion order cannot change merged state. |
| `messages` | append (`operator.add`) | Retain the ordered conversation/audit trail. |
| `tool_results` | append (`operator.add`) | Retain every attempt for retry diagnosis. |
| `errors` | append (`operator.add`) | Retain all failure and retry evidence. |
| `events` | append (`operator.add`) | Retain the node route used by metrics and recovery proof. |

## 4. Scenario results

The following summary and rows are generated only from `outputs/metrics.json`. They are
bounded by markers so `make run-scenarios` can refresh measured values without replacing
the surrounding submission narrative.

<!-- BEGIN GENERATED:METRICS_SUMMARY -->
| Metric | Value |
|---|---:|
| Total scenarios | 7 |
| Success rate | 100% |
| Average nodes visited | 6.43 |
| Total retries | 3 |
| Total approval-node visits | 2 |
| Resume success demonstrated | no |
<!-- END GENERATED:METRICS_SUMMARY -->

<!-- BEGIN GENERATED:SCENARIO_RESULTS -->
| Scenario | Expected route | Actual route | Success | Nodes | Retries | Approval visits | Approval observed | Latency (ms) | Errors |
|---|---|---|---|---:|---:|---:|---|---:|---|
| S01_simple | simple | simple | yes | 4 | 0 | 0 | no | 22602 | — |
| S02_tool | tool | tool | yes | 6 | 0 | 0 | no | 2804 | — |
| S03_missing | missing_info | missing_info | yes | 4 | 0 | 0 | no | 578 | — |
| S04_risky | risky | risky | yes | 8 | 0 | 1 | yes | 4745 | — |
| S05_error | error | error | yes | 10 | 2 | 0 | no | 31012 | Retry 1 requested after an unsatisfactory tool result.; Retry 2 requested after an unsatisfactory tool result. |
| S06_delete | risky | risky | yes | 8 | 0 | 1 | yes | 6414 | — |
| S07_dead_letter | error | error | yes | 5 | 1 | 0 | no | 3336 | Retry 1 requested after an unsatisfactory tool result. |
<!-- END GENERATED:SCENARIO_RESULTS -->

Here, success means only that `actual_route == expected_route`, the state contains a nonempty
`final_answer` or `pending_question`, and an approval-required row contains some approval
mapping. The metric does not assert `approved=true`, side-effect execution, or a `finalize`
event; those contracts have separate graph/audit tests. A bounded dead-letter outcome can
therefore score as successful workflow containment. `Total approval-node visits` is not a count
of real LangGraph interrupts; real interrupt/resume has separate evidence in
`outputs/hitl_evidence.json`. These core retry counts use the deterministic evaluator and do
not include extension-judge quality verdicts. Latency is wall-clock time around the complete
`graph.invoke()` call, including classification/answer LLM calls, not per-node latency or an
SLA benchmark.

## 5. Failure analysis

### 5.1 Tool failure, bounded retry, and dead-letter containment

- **Origin:** an `error` classification enters `retry` directly. A later mock-tool call can
  report `ERROR`; in extension mode an LLM quality verdict can also request the same retry.
- **Detection signal:** append-only `tool_results` retains every serial attempt, while the
  core evaluator deliberately reads the latest result so a recovered attempt is not poisoned
  by an older error. `evaluation_result` drives the route. In extension mode,
  `evaluation_reason`/`evaluation_source` distinguish `llm`, `heuristic_fallback`, and the
  fail-closed `policy_guard`; retry appends to `errors` and `events`.
- **Next route:** `route_after_evaluate` sends `needs_retry` to `retry`. The retry node
  increments `attempt`; `route_after_retry` then selects `tool` while
  `attempt < max_attempts`, otherwise `dead_letter`.
- **Termination guarantee:** `attempt` increases monotonically, the router fails closed for
  malformed limits, and `dead_letter → finalize → END`. There is no unbounded retry edge.
- **Proof:** the final core `S05_error` measurement records two retries: its second tool
  attempt succeeds and proceeds through `answer → finalize`. `S07_dead_letter` records one
  retry with `max_attempts=1`, then `dead_letter → finalize`. The contract test
  `test_retry_boundary_dead_letters_without_calling_tool` verifies the boundary directly.
  `test_llm_judge_falls_back_without_exposing_provider_exception` separately proves that a
  provider failure uses the deterministic fallback without persisting raw exception text;
  `test_llm_judge_cannot_override_explicit_error_evidence` proves a model cannot turn an
  explicit tool error into success.
- **Residual risk:** the tools remain deterministic mocks. The latest live run also shows
  that a structured judge can reject semantically thin but transport-successful mock results:
  the extension suite recorded eight retries before policy/budget containment. Adapter timeout,
  zero SDK retry, prompt/output bounds, and dead-letter termination contain cost; they do not
  eliminate semantic false positives, provider stalls outside adapter enforcement, or the need
  for idempotency, backoff, and circuit breaking.

### 5.2 Risky action rejected before tool execution

- **Origin:** a `risky` classification goes to `risky_action`, which records a proposed
  side effect without executing it, then reaches `approval`.
- **Detection signal:** `approval.approved` must be exactly `True`. A rejected, absent, or
  malformed decision fails closed, and the approval event records the decision outcome.
- **Next route:** `route_after_approval` selects `tool` only for explicit approval; every
  other decision routes to `clarify`, so the risky tool is not called.
- **Termination guarantee:** the rejected branch is `approval → clarify → finalize → END`;
  it has no cycle and no side-effect edge.
- **Proof:** `test_rejected_approval_clarifies_without_calling_tool` runs the compiled graph
  with thread ID `contract-rejected` and asserts the visited path is
  `intake → classify → risky_action → approval → clarify → finalize`; because `tool` is absent,
  the tool double was never called. The live `make prove-hitl` artifact additionally records a
  real interrupt, same-thread resume, rejected decision, `clarify → finalize`, and
  `tool_called=false`. The seven metric scenarios still force the CI-safe mock mode, so their
  two approval-node visits are not relabeled as interrupts.
- **Residual risk:** the extension accepts a reviewer alias but has no authentication or
  authorization policy. `MemorySaver` keeps the UI resume state only within one process, and
  a durable decision timestamp/immutable approval audit is not yet implemented.

A rejected request is a **safe workflow completion**. It does not mean that the action was
approved or executed; it means the graph contained the side effect and returned a safe next
step.

## 6. Persistence and recovery evidence

`configs/lab.yaml` selects the `memory` checkpointer. The CLI passes
`configurable.thread_id` to every invocation and reads `get_state_history()` afterward.
The final scenario run wrote `outputs/persistence_evidence.json` with `backend="memory"`,
seven distinct thread records, and `history_proven=true`. Concrete examples are:

| Scenario | Thread ID | State-history snapshots |
|---|---|---:|
| `S05_error` | `thread-S05_error` | 12 |
| `S07_dead_letter` | `thread-S07_dead_letter` | 7 |

This proves that checkpointed state history can be retrieved under each of seven distinct
thread IDs inside the running process; it is more than a statement that `MemorySaver` was
instantiated. It does not by itself test cross-thread non-contamination, crash recovery, or
process-restart recovery. `resume_success` remains `no`, and no real interrupt/resume claim is
made **by the seven-row metrics suite**.

Durable recovery is proven separately by
`test_sqlite_checkpoint_survives_abrupt_process_exit`: process A writes an approval interrupt
to SQLite with thread ID `durable-crash-proof`, then terminates via `os._exit(23)` without
closing its saver. Process B opens a fresh connection and compiled probe graph, finds one
persisted interrupt and at least two checkpoints, resumes the keyed decision, and reaches a
completed terminal state. This is why the metrics flag remains `no` while the optional SQLite
extension can still claim a dedicated crash/recovery proof.

## 7. Extension work

Every item below was run after the core gate passed. Optional dependencies and interactive
behavior remain outside the default offline graph contract.

### 7.1 Structured LLM-as-judge

- **Baseline/change:** the core config keeps the deterministic `ERROR` check. The isolated
  `configs/extensions.yaml` opts the same registered node into an
  `EvaluationVerdict(verdict, reason)`. Missing/explicit-error evidence is rejected by policy
  before any model call, and tool evidence is delimited as untrusted data in the prompt.
- **Check/evidence:** the live extension run recorded five `llm`, three
  `heuristic_fallback`, and one `policy_guard` evaluation event in the metadata-only
  `outputs/extension_audit_events.jsonl`; the same counts appear in
  `outputs/scenario_extension_evidence.json`. Structured-output, prompt-injection boundary,
  provider-failure, explicit-error, and hard-call-budget tests cover the paths.
- **Limits:** timeout is configured at the provider adapter, not measured as a hard process
  deadline. The prompt is capped at 4,000 characters, output at 128 tokens, SDK retries at zero,
  and judge calls at two; semantic false positives are still possible and occurred on thin mock
  results in the live extension run. The core metrics are intentionally unaffected.

### 7.2 Real HITL interrupt/resume

- **Baseline/change:** core scenarios explicitly use mock approval. `HitlRunner` instead sets
  `approval_mode=interrupt`, validates the resumed `ApprovalDecision`, binds it to the active
  interrupt ID, and resumes the same server-generated thread.
- **Check/evidence:** `make prove-hitl` produced `outputs/hitl_evidence.json`: an interrupt was
  observed, the rejected decision resumed the same thread, the route ended through
  `clarify → finalize`, and `tool_called=false`. Approval and rejection are also covered with
  the real compiled graph in `tests/test_hitl_extension.py`.
- **Limits:** the UI runner uses in-process memory, reviewer identity is only a supplied alias,
  and the proof is not an authorization or non-repudiation system.

### 7.3 SQLite durable recovery (plus unclaimed Postgres runtime)

- **Baseline/change:** `MemorySaver` cannot survive exit. `open_checkpointer()` now owns safe
  lifecycles for memory, SQLite, and Postgres; SQLite requires an explicit absolute file path,
  while Postgres requires an explicit DSN and optional binary driver.
- **Check/evidence:** the abrupt-exit/fresh-connection proof in Section 6 passes with the
  `sqlite` extra. The Postgres adapter and validation are implemented, but no live Postgres
  service run is claimed.
- **Limits:** SQLite suits a single-process/light-concurrency deployment. Checkpoints contain
  raw workflow state and therefore need retention, access control, encryption, and compatible
  graph/schema versions before production recovery.

### 7.4 Controlled time travel

- **Baseline/change:** state history was previously read only for counting snapshots. The
  extension selects by stable checkpoint ID or unambiguous predicate, permits replay only for
  allowlisted next nodes, and permits forks only through the reviewable content fields
  `query`, `final_answer`, and `pending_question`. A query fork is accepted only immediately
  before `classify`; route, risk, retry, evaluation, action, and approval fields are immutable.
- **Check/evidence:** `tests/test_time_travel_extension.py` replays a selected checkpoint,
  creates a new checkpoint branch with a reviewed answer, and proves unsafe node replay plus
  control-state forgery are rejected.
- **Limits:** the proof uses an offline deterministic graph. Operators still need an explicit
  review workflow before replaying any production checkpoint with downstream side effects.

### 7.5 Parallel `Send()` fan-out

- **Baseline/change:** one core `tool` invocation handles a lookup. Extension S02 supplies two
  independent, read-only `tool_tasks`; the router emits real `Send("tool", ...)` branches and the custom
  associative/commutative reducer keeps the latest attempt per task in canonical order before
  one evaluation fan-in.
- **Check/evidence:** `outputs/scenario_extension_evidence.json` records one fan-out scenario,
  two configured tasks, six cumulative task events across three attempts, and two final
  deterministic results. `outputs/extension_audit_events.jsonl` stores only stable task
  fingerprints—not messages or raw task text.
  `tests/test_parallel_fanout.py` proves real `Send` objects, deterministic merging, one fan-in,
  retry dispatch, and that risky actions never fan out.
- **Limits:** the tools are mocks and no latency speedup is claimed. Only independent,
  idempotent/read-only work is eligible; cancellation and partial branch failure need more work.

### 7.6 Streamlit reviewer UI

- **Baseline/change:** the original browser demo used mock approval. `apps/streamlit_app.py` is
  a thin view over `HitlRunner`, with start, approve/reject, same-thread resume, and an allowlist
  that omits raw state, event metadata, environment values, and exception details.
- **Check/evidence:** ten UI tests include a real Streamlit `AppTest`, stale-ticket rejection,
  display-bound interrupt-ID checks, and credential-shaped-text redaction; a headless server
  smoke returned `ok` from `/_stcore/health`. The behavioral HITL proof remains the runner
  test/artifact, not a screenshot or simulated browser route.
- **Limits:** there is no authentication, role policy, multi-worker durable session store, or
  CSRF/audit hardening, so this remains a lab reviewer UI.

### 7.7 Mermaid graph export

- **Baseline/change:** diagrams could drift when copied by hand. `make export-graph` calls
  `build_graph().get_graph().draw_mermaid()` and writes normalized UTF-8 text.
- **Check/evidence:** `outputs/graph.mmd` contains the actual 11 application nodes and 19 compiled
  edge instances; `tests/test_diagram.py` proves repeatability and exact compiled-graph sourcing.
- **Limits:** Mermaid proves static topology, not which conditional route ran or whether a
  particular external side effect succeeded.

## 8. Improvement plan

The single highest-priority production step is authenticated, policy-enforced reviewer
authorization with an immutable approval audit. It comes first because the real interrupt,
durable checkpoint, and UI now prove orchestration mechanics, but a caller-supplied reviewer
alias still cannot establish who authorized a consequential action or whether they had the
right to do so.
