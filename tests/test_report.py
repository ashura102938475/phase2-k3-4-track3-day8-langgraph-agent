"""Behavioral tests for the human-readable lab report."""

from __future__ import annotations

from langgraph_agent_lab.metrics import MetricsReport, ScenarioMetric
from langgraph_agent_lab.report import render_report


def _metrics() -> MetricsReport:
    return MetricsReport(
        total_scenarios=2,
        success_rate=0.5,
        avg_nodes_visited=4.5,
        total_retries=3,
        total_interrupts=1,
        resume_success=False,
        scenario_metrics=[
            ScenarioMetric(
                scenario_id="simple-1",
                success=True,
                expected_route="simple",
                actual_route="simple",
                nodes_visited=3,
                latency_ms=120,
            ),
            ScenarioMetric(
                scenario_id="risky-2",
                success=False,
                expected_route="risky",
                actual_route="error",
                nodes_visited=6,
                retry_count=3,
                interrupt_count=1,
                approval_required=True,
                approval_observed=True,
                latency_ms=875,
                errors=["tool timeout"],
            ),
        ],
    )


def test_render_report_contains_summary_and_per_scenario_metrics() -> None:
    report = render_report(_metrics())

    assert "2" in report
    assert "50%" in report or "0.5" in report
    assert "3" in report  # total retries
    assert "1" in report  # total interrupts
    assert "simple-1" in report and "risky-2" in report
    assert "simple" in report and "error" in report
    assert "120" in report and "875" in report


def test_render_report_explains_architecture_state_failures_and_persistence_caveat() -> None:
    report = render_report(_metrics()).lower()

    assert "architecture" in report
    assert "state" in report
    assert "reducer" in report
    assert "retry" in report and ("tool" in report or "timeout" in report)
    assert "approval" in report and ("risk" in report or "risky" in report)
    assert "persistence" in report
    assert "caveat" in report or "limitation" in report or "not persisted" in report


def test_render_report_distinguishes_safe_rejection_from_action_success() -> None:
    report = render_report(_metrics()).lower()

    assert "safe workflow completion" in report
    assert "does not mean" in report and "action" in report and "approved" in report


def test_render_report_does_not_invent_active_checkpointer_evidence() -> None:
    report = render_report(_metrics()).lower()

    assert "metrics" in report and "do not record" in report and "checkpointer" in report


def test_render_report_escapes_control_characters_and_markdown_pipes() -> None:
    metrics = _metrics()
    metrics.scenario_metrics[0].scenario_id = "a\\|b\r\nc"

    report = render_report(metrics)

    assert "\r" not in report
    assert r"a\\\|b  c" in report
