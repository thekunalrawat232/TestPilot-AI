"""Node 4 & 5 — Execution & Debug Agent (combined with conditional retry loop)."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage

from config.settings import get_llm, exec_config, paths
from prompts.execution_debug import EXECUTION_DEBUG_PROMPT
from .state import PipelineState
from .utils import extract_json, trim_context_to_fit
from .llm_cache import cached_llm_invoke


def _run_tests(scripts_dir: Path) -> dict[str, str]:
    """Execute all test files in the scripts directory and collect outputs."""
    results: dict[str, str] = {}

    test_files = sorted(scripts_dir.glob("test_*.py"))
    if not test_files:
        return {"_no_tests": "No test files found in generated scripts directory."}

    for test_file in test_files:
        cmd = [
            "python3", "-m", "pytest",
            str(test_file),
            "--tb=short",
            "-v",
            "--timeout=60",
        ]

        project_root = Path(__file__).parent.parent
        pythonpath = f"{scripts_dir}{os.pathsep}{project_root}"
        env = {
            **os.environ,
            "BASE_URL": exec_config.target_base_url,
            "HEADLESS": str(exec_config.headless).lower(),
            "PYTHONPATH": pythonpath,
        }

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=180,
                env=env,
                cwd=str(scripts_dir),
            )
            output = proc.stdout + "\n" + proc.stderr
            results[test_file.name] = output[-6000:]
        except subprocess.TimeoutExpired:
            results[test_file.name] = "TIMEOUT: Test execution exceeded 180 seconds."
        except Exception as exc:
            results[test_file.name] = f"EXECUTION ERROR: {exc}"

    return results


def _apply_fixes(fixed_files: list[dict[str, str]], scripts_dir: Path) -> list[str]:
    """Write corrected files back to disk."""
    written: list[str] = []
    for fix in fixed_files:
        fname = fix.get("file_name", "")
        code = fix.get("code", "")
        if fname and code:
            # Handle page objects subdirectory
            if fname.endswith("_page.py"):
                fpath = scripts_dir / "page_objects" / fname
            else:
                fpath = scripts_dir / fname
            fpath.parent.mkdir(parents=True, exist_ok=True)
            fpath.write_text(code)
            written.append(str(fpath))
    return written


# ---------------------------------------------------------------------------
# Node 4: Execute tests
# ---------------------------------------------------------------------------

def execution_node(state: PipelineState) -> dict[str, Any]:
    """Run all generated test scripts and capture results.

    Inputs from state:
    - generated_code (from Node 3, or fixed code from Node 5)

    Outputs:
    - execution_result: raw test outputs keyed by file name
    """
    scripts_dir = paths.generated_scripts

    raw_results = _run_tests(scripts_dir)

    # Build a summary
    total_files = len(raw_results)
    failed_files = [
        fname for fname, output in raw_results.items()
        if "FAILED" in output or "ERROR" in output or "TIMEOUT" in output
    ]

    return {
        "execution_result": {
            "raw_outputs": raw_results,
            "total_files": total_files,
            "failed_files": failed_files,
            "all_passed": len(failed_files) == 0,
        },
        "pipeline_status": "passed" if len(failed_files) == 0 else (
            "failed" if state.retry_count >= state.max_retries else "running"
        ),
    }


# ---------------------------------------------------------------------------
# Node 5: Debug & fix failures
# ---------------------------------------------------------------------------

def debug_node(state: PipelineState) -> dict[str, Any]:
    """Analyse failures, classify them, and produce fixes.

    Inputs from state:
    - execution_result (from Node 4)
    - generated_code (current code on disk)

    Outputs:
    - debug_analysis: classification + fixes
    - retry_count: incremented
    """
    llm = get_llm()

    exec_result = state.execution_result
    raw_outputs = exec_result.get("raw_outputs", {})
    failed_files = exec_result.get("failed_files", [])

    # Collect the source code of failing test files for the LLM
    scripts_dir = paths.generated_scripts
    file_sources: dict[str, str] = {}
    for fname in failed_files:
        fpath = scripts_dir / fname
        if fpath.exists():
            file_sources[fname] = fpath.read_text()

    # Truncate individual outputs and source files to keep the prompt compact
    truncated_outputs = {f: raw_outputs.get(f, "")[-3000:] for f in failed_files}
    truncated_sources = {f: src[-3000:] for f, src in file_sources.items()}

    failure_report = json.dumps({
        "failed_files": failed_files,
        "outputs": truncated_outputs,
        "source_code": truncated_sources,
    }, indent=2)

    retry_text = f"## Retry Attempt: {state.retry_count + 1} of {state.max_retries}"

    failure_report = trim_context_to_fit(
        system_prompt=EXECUTION_DEBUG_PROMPT,
        user_content_parts=[retry_text],
        context=failure_report,
    )

    messages = [
        SystemMessage(content=EXECUTION_DEBUG_PROMPT),
        HumanMessage(content=(
            f"## Test Execution Failures\n```json\n{failure_report}\n```\n\n"
            f"{retry_text}"
        )),
    ]

    response = cached_llm_invoke(llm, messages, node_name="debug")

    try:
        analysis = extract_json(response.content)
    except (json.JSONDecodeError, Exception) as exc:
        return {
            "debug_analysis": {"raw_response": response.content},
            "error_log": [f"DebugAgent JSON parse error: {exc}"],
            "retry_count": state.retry_count + 1,
        }

    # Apply fixes to disk if the agent produced them
    fixed_files = analysis.get("fixed_files", [])
    if fixed_files:
        written = _apply_fixes(fixed_files, scripts_dir)
        analysis["applied_fixes"] = written

    return {
        "debug_analysis": analysis,
        "retry_count": state.retry_count + 1,
    }


# ---------------------------------------------------------------------------
# Conditional edge: should we retry?
# ---------------------------------------------------------------------------

def should_retry(state: PipelineState) -> str:
    """Routing function for the conditional edge after execution.

    Returns:
    - "debug"  → failures exist and retries remain
    - "done"   → all tests passed OR retries exhausted
    """
    exec_result = state.execution_result
    all_passed = exec_result.get("all_passed", False)

    if all_passed:
        return "done"

    if state.retry_count >= state.max_retries:
        return "done"

    return "debug"
