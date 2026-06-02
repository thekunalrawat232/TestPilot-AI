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
from .execution_debug import execution_node, debug_node, should_retry
from .checkpoint import save_checkpoint, load_checkpoint
from integrations.trello import push_bugs_to_trello

logger = logging.getLogger(__name__)

NODE_ORDER = [
    "requirement_and_design",
    "code_generator",
    "execution",
    "debug",
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

    test_suites = test_plan.get("test_suites", [])
    test_cases_count = sum(len(s.get("test_cases", [])) for s in test_suites)

    summary = {
        "feature_name": req_analysis.get("feature_name", ""),
        "test_suites": len(test_suites),
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

def finalise_node(state: PipelineState) -> dict[str, Any]:
    """Produce a human-readable summary report and persist it."""
    exec_result = state.execution_result
    debug = state.debug_analysis
    all_passed = exec_result.get("all_passed", False)

    status = "PASSED" if all_passed else "FAILED"
    if not all_passed and state.retry_count >= state.max_retries:
        status = "FAILED (retries exhausted)"

    report_lines = [
        f"# Test Pipeline Report",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        f"**Feature:** {state.requirement_analysis.get('feature_name', 'unknown')}",
        f"**Status:** {status}",
        f"**Retries used:** {state.retry_count} / {state.max_retries}",
        "",
        "## Execution Summary",
        f"- Total test files: {exec_result.get('total_files', 0)}",
        f"- Failed files: {exec_result.get('failed_files', [])}",
        "",
    ]

    bug_reports = debug.get("bug_reports", [])
    feature_name = state.requirement_analysis.get("feature_name", "unknown")
    if bug_reports:
        report_lines.append("## Real Bugs Found")
        for br in bug_reports:
            report_lines.append(f"### {br.get('title', 'Untitled')}")
            report_lines.append(f"**Severity:** {br.get('severity', 'unknown')}")
            report_lines.append(f"**Expected:** {br.get('expected', '')}")
            report_lines.append(f"**Actual:** {br.get('actual', '')}")
            report_lines.append("")

    trello_cards = push_bugs_to_trello(bug_reports, feature_name=feature_name)
    if trello_cards:
        report_lines.append("## Trello Cards Created")
        for card in trello_cards:
            report_lines.append(f"- [{card['name']}]({card['url']})")
        report_lines.append("")

    if state.error_log:
        report_lines.append("## Pipeline Errors")
        for err in state.error_log:
            report_lines.append(f"- {err}")

    report_text = "\n".join(report_lines)

    reports_dir = paths.generated_reports
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = reports_dir / f"report_{ts}.md"
    report_path.write_text(report_text)

    return {
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
        "debug": _make_checkpointing_node("debug", debug_node, run_id),
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
    ]
    for src, dst in linear_edges:
        if NODE_ORDER.index(src) >= entry_idx:
            graph.add_edge(src, dst)

    graph.add_conditional_edges(
        "execution",
        should_retry,
        {"debug": "debug", "done": "finalise"},
    )
    graph.add_edge("debug", "execution")
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
