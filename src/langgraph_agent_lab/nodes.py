"""Node functions for the LangGraph workflow.

Each function receives AgentState and returns a partial state update dict.
Do NOT mutate input state — return new values only.

LLM REQUIREMENT:
- classify_node MUST use a real LLM call (structured output for intent classification)
- answer_node MUST use a real LLM call (grounded response generation)
- evaluate_node SHOULD use LLM-as-judge (bonus points; heuristic acceptable for base score)
"""

from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel, Field

from .llm import get_llm, is_nvidia_chat_model
from .state import AgentState, make_event


class IntentClassification(BaseModel):
    """Structured response expected from the intent-classification model."""

    route: Literal["simple", "tool", "missing_info", "risky", "error"] = Field(
        description="The single route selected for the user's query."
    )
    risk_level: Literal["low", "high"] = Field(
        description="Use high only when the selected route is risky; otherwise use low."
    )


def _content(response: object) -> str:
    """Extract plain text from a LangChain chat response."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join(str(part) for part in content).strip()
    return str(content).strip()


# ─── EXAMPLE: working node (provided for reference) ──────────────────
def intake_node(state: AgentState) -> dict:
    """Normalize raw query. This node is provided as a working example."""
    query = state.get("query", "").strip()
    return {
        "query": query,
        "messages": [f"intake:{query[:40]}"],
        "events": [make_event("intake", "completed", "query normalized")],
    }


# ─── Workflow nodes ───────────────────────────────────────────────


def classify_node(state: AgentState) -> dict:
    """Classify the query into a route using an LLM.

    *** MUST use a real LLM call — keyword-only heuristics will lose points. ***

    Use .with_structured_output() or equivalent to get reliable enum classification.
    The LLM should classify into one of: simple, tool, missing_info, risky, error.

    Hints:
    - See llm.py for the get_llm() helper
    - Use Pydantic model or TypedDict with .with_structured_output()
    - Set risk_level to "high" for risky routes, "low" otherwise
    - Priority guide: risky > tool > missing_info > error > simple

    Return: {"route": str, "risk_level": str, "events": [make_event(...)]}
    """
    query = state.get("query", "").strip()
    prompt = f"""Classify this user request into exactly one route.

Routes, in strict precedence order when more than one could apply:
1. risky: an action with consequential side effects, including deletion, cancellation,
   refund, purchase, sending a message, or changing access/data.
2. tool: an information lookup, search, status, tracking, or retrieval request.
3. missing_info: a request cannot be completed safely because essential context is absent.
   A generic action with an unresolved reference (such as it, this, or that) and no
   identified object, system, symptom, or desired outcome MUST use missing_info.
   Do not use simple merely because a request is short.
4. error: the request reports or asks about a system failure, outage, crash, or timeout.
5. simple: a self-contained general question that needs no external tool or side effect.

Return one JSON object with route and risk_level. risk_level must be high exactly for risky,
otherwise low.
User request: {query!r}"""
    llm = get_llm()
    if is_nvidia_chat_model(llm):
        classifier = llm.with_structured_output(IntentClassification, method="json_mode")
    else:
        classifier = llm.with_structured_output(IntentClassification)
    result = classifier.invoke(prompt)
    route = getattr(result, "route", None)
    risk_level = getattr(result, "risk_level", None)
    if isinstance(result, dict):
        route = result.get("route")
        risk_level = result.get("risk_level")
    if route not in {"simple", "tool", "missing_info", "risky", "error"}:
        raise ValueError("LLM classification returned an unsupported route")
    if risk_level not in {"low", "high"}:
        raise ValueError("LLM classification returned an unsupported risk level")
    return {
        "route": route,
        "risk_level": "high" if route == "risky" else "low",
        "events": [make_event("classify", "completed", f"classified as {route}")],
    }


def tool_node(state: AgentState) -> dict:
    """Execute a mock tool call.

    Simulate transient failures for error-route scenarios to test retry loops.

    Requirements:
    - Read current attempt count from state
    - If route is "error" and attempt < 2: return error result (string containing "ERROR")
    - Otherwise: return a mock success result string
    - Append result to tool_results list

    Return: {"tool_results": [result_string], "events": [make_event(...)]}
    """
    attempt = state.get("attempt", 0)
    attempt = attempt if isinstance(attempt, int) and not isinstance(attempt, bool) else 0
    if state.get("route") == "error" and attempt < 2:
        result = f"ERROR: transient tool outage (attempt {attempt + 1})"
        event_type = "failed"
    else:
        result = f"Tool result for: {state.get('query', '').strip()}"
        event_type = "completed"
    return {
        "tool_results": [result],
        "events": [make_event("tool", event_type, result)],
    }


def evaluate_node(state: AgentState) -> dict:
    """Evaluate tool results — the retry-loop gate.

    Check whether the latest tool result is satisfactory or needs retry.

    SHOULD use LLM-as-judge for bonus points. Heuristic (e.g., check for "ERROR" substring)
    is acceptable for base score.

    Requirements:
    - Read the latest entry from tool_results
    - Set evaluation_result to "needs_retry" or "success"
    - This field drives route_after_evaluate conditional edge

    Note: You may need to add 'evaluation_result' to AgentState if not present.

    Return: {"evaluation_result": str, "events": [make_event(...)]}
    """
    results = state.get("tool_results", [])
    latest = results[-1] if results else "ERROR: no tool result returned"
    evaluation_result = "needs_retry" if "ERROR" in latest.upper() else "success"
    return {
        "evaluation_result": evaluation_result,
        "events": [
            make_event("evaluate", "completed", f"tool result evaluated as {evaluation_result}")
        ],
    }


def answer_node(state: AgentState) -> dict:
    """Generate a final response using an LLM.

    *** MUST use a real LLM call — hardcoded strings will lose points. ***

    The LLM should generate a helpful response grounded in available context:
    - tool_results (if any)
    - approval decision (if risky route)
    - original query

    Return: {"final_answer": str, "events": [make_event(...)]}
    """
    query = state.get("query", "").strip()
    tool_results = state.get("tool_results", [])
    approval = state.get("approval")
    prompt = f"""Answer the user's request using only the supplied workflow context.
If the context is insufficient, say what is missing rather than inventing facts.

User request: {query}
Tool results: {tool_results}
Approval decision: {approval}
"""
    answer = _content(get_llm().invoke(prompt))
    return {
        "final_answer": answer,
        "events": [make_event("answer", "completed", "generated grounded answer")],
    }


def ask_clarification_node(state: AgentState) -> dict:
    """Ask for missing information instead of hallucinating.

    Generate a specific clarification question based on the vague/incomplete query.

    Note: You may need to add 'pending_question' to AgentState if not present.

    Return: {"pending_question": str, "final_answer": str, "events": [make_event(...)]}
    """
    query = state.get("query", "").strip()
    question = f"To help with '{query}', what specific outcome and key details should I use?"
    return {
        "pending_question": question,
        "final_answer": question,
        "events": [make_event("clarify", "completed", "requested missing information")],
    }


def risky_action_node(state: AgentState) -> dict:
    """Prepare a risky action for human approval.

    Describe the proposed action and why it requires approval.

    Note: You may need to add 'proposed_action' to AgentState if not present.

    Return: {"proposed_action": str, "events": [make_event(...)]}
    """
    proposed_action = (
        f"Proposed action: {state.get('query', '').strip()}. "
        "This may have consequential side effects and requires human approval."
    )
    return {
        "proposed_action": proposed_action,
        "events": [make_event("risky_action", "pending_approval", "prepared action for review")],
    }


def approval_node(state: AgentState) -> dict:
    """Human-in-the-loop approval step.

    Default behavior: mock approval (approved=True) so tests and CI run offline.
    Extension: if env LANGGRAPH_INTERRUPT=true, use langgraph.types.interrupt() for real HITL.

    Return: approval mapping and one event.
    """
    proposed_action = state.get("proposed_action", "")
    if os.getenv("LANGGRAPH_INTERRUPT", "").lower() == "true":
        from langgraph.types import interrupt

        response = interrupt(
            {"proposed_action": proposed_action, "instruction": "Approve or reject."}
        )
        response = response if isinstance(response, dict) else {}
        approval = {
            "approved": response.get("approved") is True,
            "reviewer": str(response.get("reviewer", "human-reviewer")),
            "comment": str(response.get("comment", "Human review completed.")),
        }
    else:
        approval = {
            "approved": True,
            "reviewer": "mock-reviewer",
            "comment": "Automatically approved for offline workflow execution.",
        }
    message = "action approved" if approval["approved"] else "action rejected"
    return {
        "approval": approval,
        "events": [make_event("approval", "completed", message)],
    }


def retry_or_fallback_node(state: AgentState) -> dict:
    """Record a retry attempt.

    Increment the attempt counter and log the transient failure.

    Requirements:
    - Read current attempt from state, increment by 1
    - Add an error message to errors list
    - Return updated attempt count

    Return: {"attempt": int, "errors": [str], "events": [make_event(...)]}
    """
    attempt = state.get("attempt", 0)
    attempt = attempt if isinstance(attempt, int) and not isinstance(attempt, bool) else 0
    next_attempt = attempt + 1
    error = f"Retry {next_attempt} requested after an unsatisfactory tool result."
    return {
        "attempt": next_attempt,
        "errors": [error],
        "events": [make_event("retry", "retrying", error)],
    }


def dead_letter_node(state: AgentState) -> dict:
    """Handle unresolvable failures after max retries exceeded.

    This is the third layer: retry → fallback → dead letter.
    Log the failure and set a final_answer explaining that the request could not be completed.

    Return: {"final_answer": str, "events": [make_event(...)]}
    """
    final_answer = "The request could not be completed after the allowed retry attempts."
    return {
        "final_answer": final_answer,
        "events": [make_event("dead_letter", "failed", "retry limit reached")],
    }


def finalize_node(state: AgentState) -> dict:
    """Emit a final audit event. All routes must pass through here before END.

    Return: {"events": [make_event("finalize", "completed", "workflow finished")]}
    """
    return {"events": [make_event("finalize", "completed", "workflow finished")]}
