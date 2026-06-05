"""LangGraph workflow — the multi-agent pipeline orchestrator.

Graph topology:

    ┌─────────────────────────┐
    │  Requirement & Design    │  (combined: analyst + test designer — 1 LLM call)
    │  (Node 1)                │
    └──────────┬──────────────┘
               │
    ┌──────────▼──────────────┐
    │  Code Generator          │
    │  (Node 2)                │
    └──────────┬──────────────┘
               │
    ┌──────────▼──────────────┐
    │  Execution               │◄──────────┐
    │  (Node 3)                │           │
    └──────────┬──────────────┘           │
               │                           │
          ┌────▼────┐                      │
          │ should   │   "debug"           │
          │ retry?   │────────►┌───────────┴──┐
          └────┬────┘         │  Debug Loop   │
               │ "done"       │  (Node 4)     │
               │              └──────────────┘
    ┌──────────▼──────────────┐
    │  Finalise                │
    │  (Terminal)              │
    └─────────────────────────┘
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langgraph.graph import StateGraph, END

from config.settings import paths, exec_config
from .state import PipelineState
from .requirement_and_design import requirement_and_design_node
from .code_generator import code_generator_node
from .execution_debug import execution_node
from .checkpoint import save_checkpoint, load_checkpoint
from integrations.trello import push_bugs_to_trello

logger = logging.getLogger(__name__)

# Linear pipeline: plan (1 LLM call) -> render (deterministic) -> execute -> finalise.
# The Debug agent was removed — the deterministic renderer never produces broken
# code, so there is nothing to auto-fix, and dropping it keeps runs to 1 LLM call.
NODE_ORDER = [
    "requirement_and_design",
    "code_generator",
    "execution",
    "finalise",
]


# ---------------------------------------------------------------------------
# Live status — writes progress to a JSON file for the Streamlit dashboard
# ---------------------------------------------------------------------------

def _write_live_status(run_id: str, node_name: str, event: str, state_dict: dict) -> None:
    """Write current pipeline state to live_status.json for the dashboard to poll."""
    status_file = paths.generated_reports / "live_status.json"
    status_file.parent.mkdir(parents=True, exist_ok=True)

    # Load existing status or start fresh
    if status_file.exists():
        try:
            current = json.loads(status_file.read_text())
        except Exception:
            current = {}
    else:
        current = {}

    now = datetime.now(timezone.utc).isoformat()

    node_statuses = current.get("node_statuses", {n: "pending" for n in NODE_ORDER})
    node_completed_at = current.get("node_completed_at", {})

    if event == "init":
        node_statuses = {n: "pending" for n in NODE_ORDER}
        node_completed_at = {}
    elif event == "start":
        node_statuses[node_name] = "running"
    elif event == "done":
        node_statuses[node_name] = "done"
        node_completed_at[node_name] = now
    elif event == "failed":
        node_statuses[node_name] = "failed"
        node_completed_at[node_name] = now

    # Extract key metrics from state for the dashboard summary
    req_analysis = state_dict.get("requirement_analysis", {})
    test_plan = state_dict.get("test_plan", {})
    exec_result = state_dict.get("execution_result", {})
    debug_analysis = state_dict.get("debug_analysis", {})
    generated_code = state_dict.get("generated_code", {})

    tests = test_plan.get("tests", [])
    test_cases_count = len(tests)

    summary = {
        "feature_name": req_analysis.get("feature_name", ""),
        "test_suites": 1 if tests else 0,
        "test_cases": test_cases_count,
        "generated_files": len(generated_code.get("written_files", [])),
        "failed_files": exec_result.get("failed_files", []),
        "all_passed": exec_result.get("all_passed", False),
        "retry_count": state_dict.get("retry_count", 0),
        "max_retries": state_dict.get("max_retries", 1),
        "bug_reports": debug_analysis.get("bug_reports", []),
    }

    status = {
        "run_id": run_id,
        "requirement": state_dict.get("raw_requirement", current.get("requirement", "")),
        "started_at": current.get("started_at", now),
        "pipeline_status": state_dict.get("pipeline_status", current.get("pipeline_status", "running")),
        "current_node": node_name if event == "start" else current.get("current_node", ""),
        "node_statuses": node_statuses,
        "node_completed_at": node_completed_at,
        "summary": summary,
        "test_plan": test_plan,
        "execution_result": exec_result,
    }

    # Atomic write — write to temp then rename to avoid partial reads
    tmp = status_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(status, indent=2, default=str))
    tmp.rename(status_file)


# ---------------------------------------------------------------------------
# Checkpoint-saving wrapper
# ---------------------------------------------------------------------------

def _make_checkpointing_node(node_name: str, node_fn, run_id: str):
    """Wrap a node so it writes live status before/after and saves a checkpoint after."""

    def wrapped(state: PipelineState) -> dict[str, Any]:
        state_dict = state.dict() if hasattr(state, "dict") else dict(state)

        # Signal to dashboard that this node is now running
        _write_live_status(run_id, node_name, "start", state_dict)

        result = node_fn(state)

        # Merge result for accurate status snapshot
        merged = {**state_dict}
        if isinstance(result, dict):
            merged.update(result)

        _write_live_status(run_id, node_name, "done", merged)
        save_checkpoint(run_id, node_name, merged)
        return result

    wrapped.__name__ = node_fn.__name__
    wrapped.__doc__ = node_fn.__doc__
    return wrapped


# ---------------------------------------------------------------------------
# Terminal node — write final report
# ---------------------------------------------------------------------------

def _parse_test_results(raw_outputs: dict) -> list[tuple[str, str]]:
    """Extract (test_function_name, status) pairs from pytest -v output."""
    import re
    results: list[tuple[str, str]] = []
    seen: set[str] = set()
    for out in raw_outputs.values():
        for m in re.finditer(r"(test_\d+_\w+)\s+(PASSED|FAILED|ERROR)", out):
            if m.group(1) not in seen:
                seen.add(m.group(1))
                results.append((m.group(1), m.group(2)))
    return results


def _humanize(name: str) -> str:
    return str(name).replace("_", " ").strip().lower()


def _step_to_english(step: dict) -> str:
    """Render a plan step as a plain-English reproduction step."""
    a = str(step.get("action", "")); t = step.get("target", ""); v = step.get("value", "")
    if a == "open_section":
        sec = str(t).replace("SIDEBAR_", "").replace("_", " ").title()
        return f"Open the {sec} section from the sidebar"
    if a == "goto":
        return f"Navigate to {t}"
    if a == "click":
        return f"Click the {_humanize(t)}"
    if a == "fill":
        return f"Type \"{v}\" into the {_humanize(t)}"
    if a == "press":
        return f"Press \"{v}\" on the {_humanize(t)}"
    if a == "expect_visible":
        return f"Check the {_humanize(t)} is visible"
    if a == "expect_not_visible":
        return f"Check the {_humanize(t)} is NOT visible"
    if a == "expect_enabled":
        return f"Check the {_humanize(t)} is enabled"
    if a == "expect_text":
        return f"Check the {_humanize(t)} shows text \"{v}\""
    if a == "expect_count_gt":
        return f"Check there are more than {v} {_humanize(t)}"
    return f"{a} {t}".strip()


def _failure_detail(out: str, func: str) -> tuple[str, str]:
    """Pull the failing assertion line + the error message for `func` from the
    pytest traceback. Returns (failing_check, error_message)."""
    import re
    lines = out.splitlines()
    start = next((i for i, l in enumerate(lines) if "___" in l and func in l), None)
    if start is None:
        return "", ""
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if "___" in lines[j] and "test_" in lines[j]:
            end = j
            break
    code_line, errs = "", []
    for l in lines[start:end]:
        s = l.strip()
        if s.startswith("E "):
            errs.append(s[1:].strip())
        elif any(k in s for k in ("expect(", "assert ", ".click(", ".fill(", "open_section(")) \
                and not s.startswith("E"):
            code_line = s
    err = " ".join(errs[:4])
    err = re.sub(r"\s+", " ", err).strip()
    return code_line, err[:500]


def _expected_from_check(code_line: str) -> str:
    """Derive the expected behaviour from the failing assertion line."""
    import re
    if not code_line:
        return ""
    m = re.search(r"locator\(([\w.]+)\)", code_line)
    name = _humanize(m.group(1).split(".")[-1]) if m else "element"
    if "to_be_hidden" in code_line or "not_to_be_visible" in code_line:
        return f"The {name} should NOT be visible"
    if "to_be_visible" in code_line:
        return f"The {name} should be visible"
    if "to_contain_text" in code_line:
        return f"The {name} should contain the expected text"
    if "to_be_enabled" in code_line:
        return f"The {name} should be enabled"
    if ".count(" in code_line:
        return f"There should be more {name}"
    return ""


def finalise_node(state: PipelineState) -> dict[str, Any]:
    """Summarise findings (failing checks), write a report, file Trello cards."""
    import re

    exec_result = state.execution_result
    all_passed = exec_result.get("all_passed", False)
    feature_name = state.requirement_analysis.get("feature_name", "unknown")
    plan_tests = (state.test_plan or {}).get("tests", [])

    raw_text = "\n".join(exec_result.get("raw_outputs", {}).values())
    results = _parse_test_results(exec_result.get("raw_outputs", {}))
    passed = [n for n, s in results if s == "PASSED"]
    failed = [n for n, s in results if s in ("FAILED", "ERROR")]

    # Map each failing test function (test_<NNN>_...) back to its plan, build a
    # concrete, reproducible finding with the exact failing check + error.
    findings: list[dict[str, Any]] = []
    for name in failed:
        m = re.match(r"test_0*(\d+)_", name)
        idx = (int(m.group(1)) - 1) if m else -1
        plan_test = plan_tests[idx] if 0 <= idx < len(plan_tests) else {}
        title = plan_test.get("title") or name

        steps = [_step_to_english(s) for s in plan_test.get("steps", [])]
        failing_check, error_msg = _failure_detail(raw_text, name)

        # Expected behaviour, derived from the ACTUAL failing assertion so it
        # matches the evidence (falls back to the last check step).
        expected = _expected_from_check(failing_check)
        if not expected:
            check_steps = [s for s in steps if s.startswith("Check")]
            expected = check_steps[-1] if check_steps else "The verified condition holds on the live app."

        findings.append({
            "title": f"[{feature_name}] {title}",
            "severity": "medium",
            "steps_to_reproduce": steps or ["(see test plan)"],
            "expected": expected,
            "actual": error_msg or "The check failed against the live app.",
            "evidence": failing_check or f"pytest: {name} FAILED",
            "test": name,
        })

    status = "PASSED" if all_passed else f"FINDINGS: {len(failed)} failing check(s)"

    report_lines = [
        "# Test Pipeline Report",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        f"**Feature:** {feature_name}",
        f"**Status:** {status}",
        "",
        "## Execution Summary",
        f"- Tests run: {len(results)}",
        f"- Passed: {len(passed)}",
        f"- Failing checks (findings): {len(failed)}",
        "",
    ]

    if findings:
        report_lines.append("## Findings (failing checks → potential bugs)")
        for f in findings:
            report_lines.append(f"- **{f['title']}**  _(test: {f['test']})_")
        report_lines.append("")

    trello_cards = push_bugs_to_trello(findings, feature_name=feature_name)
    if trello_cards:
        report_lines.append("## Trello Cards Created")
        for card in trello_cards:
            report_lines.append(f"- [{card['name']}]({card['url']})")
        report_lines.append("")

    if state.error_log:
        report_lines.append("## Pipeline Notes")
        for err in state.error_log:
            report_lines.append(f"- {err}")

    reports_dir = paths.generated_reports
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    (reports_dir / f"report_{ts}.md").write_text("\n".join(report_lines))

    return {
        "debug_analysis": {"bug_reports": findings},
        "pipeline_status": "passed" if all_passed else "failed",
    }


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph(run_id: str, entry_point: str = "requirement_and_design") -> StateGraph:
    """Construct and compile the LangGraph workflow."""
    nodes = {
        "requirement_and_design": _make_checkpointing_node("requirement_and_design", requirement_and_design_node, run_id),
        "code_generator": _make_checkpointing_node("code_generator", code_generator_node, run_id),
        "execution": _make_checkpointing_node("execution", execution_node, run_id),
        "finalise": _make_checkpointing_node("finalise", finalise_node, run_id),
    }

    graph = StateGraph(PipelineState)

    for name, fn in nodes.items():
        graph.add_node(name, fn)

    graph.set_entry_point(entry_point)

    entry_idx = NODE_ORDER.index(entry_point)
    linear_edges = [
        ("requirement_and_design", "code_generator"),
        ("code_generator", "execution"),
        ("execution", "finalise"),
    ]
    for src, dst in linear_edges:
        if NODE_ORDER.index(src) >= entry_idx:
            graph.add_edge(src, dst)

    graph.add_edge("finalise", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def run_pipeline(
    requirement: str,
    *,
    max_retries: int | None = None,
    resume_run_id: str | None = None,
) -> dict[str, Any]:
    """Run the full testing pipeline for a given feature requirement."""
    entry_point = "requirement_and_design"
    initial_kwargs: dict[str, Any] = {
        "raw_requirement": requirement,
        "max_retries": max_retries or exec_config.max_retries,
        "pipeline_status": "running",
    }
    run_id = resume_run_id or uuid.uuid4().hex[:12]

    if resume_run_id:
        checkpoint = load_checkpoint(resume_run_id)
        if checkpoint:
            last_node = checkpoint["last_completed_node"]
            _legacy_map = {
                "requirement_analyst": "requirement_and_design",
                "test_designer": "requirement_and_design",
            }
            last_node = _legacy_map.get(last_node, last_node)

            last_idx = NODE_ORDER.index(last_node) if last_node in NODE_ORDER else -1

            if last_idx + 1 < len(NODE_ORDER):
                entry_point = NODE_ORDER[last_idx + 1]
            else:
                entry_point = "finalise"

            initial_kwargs = checkpoint["state"]
            initial_kwargs["pipeline_status"] = "running"

            logger.info(
                "Resuming run '%s' from node '%s' (last completed: '%s')",
                resume_run_id, entry_point, last_node,
            )
        else:
            logger.warning("No checkpoint found for run_id '%s'. Starting fresh.", resume_run_id)

    # Initialise live status file so dashboard shows all nodes as pending
    _write_live_status(run_id, entry_point, "init", {
        "raw_requirement": requirement,
        "pipeline_status": "running",
        "max_retries": initial_kwargs.get("max_retries", 1),
    })

    graph = build_graph(run_id=run_id, entry_point=entry_point)
    initial_state = PipelineState(**initial_kwargs)
    final_state = graph.invoke(initial_state)

    return final_state
