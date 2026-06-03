"""Node 4 & 5 — Execution & Debug Agent (combined with conditional retry loop)."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from langchain_core.messages import SystemMessage, HumanMessage

from config.settings import get_llm, exec_config, paths
from prompts.execution_debug import EXECUTION_DEBUG_PROMPT
from .state import PipelineState
from .utils import extract_json, trim_context_to_fit
from .llm_cache import cached_llm_invoke


def _sanitize_generated_code(scripts_dir: Path) -> list[str]:
    """Cheap, deterministic guards against collection-killing codegen bugs.

    Fixes (no API calls), applied before pytest collection:
    1. Self-referential imports — a module importing a name from its OWN module
       path (e.g. ``form_page.py`` doing ``from page_objects.form_page import X``).
       Always a circular import; the line is removed.
    2. Broken multi-line ``assert`` — ``assert cond,\n    msg`` is a SyntaxError
       in Python; the message is pulled back onto the assert line.
    """
    import re as _re

    fixed: list[str] = []
    py_files = list(scripts_dir.glob("*.py")) + list((scripts_dir / "page_objects").glob("*.py"))

    # assert <cond>,  <newline>  <indented msg>   →   assert <cond>, <msg>
    assert_break = _re.compile(r"(^[ \t]*assert\b[^\n]*,)[ \t]*\n[ \t]+(?=\S)", _re.MULTILINE)

    for f in py_files:
        try:
            text = f.read_text()
        except Exception:
            continue
        mod = f.stem  # e.g. "form_page"
        # Self-import: from form_page import X | from page_objects.form_page import X
        #              from .form_page import X | import form_page
        self_import = _re.compile(
            rf"^[ \t]*(?:from\s+(?:[\w.]*\.)?{_re.escape(mod)}\s+import\s+.*|import\s+{_re.escape(mod)})\s*$",
            _re.MULTILINE,
        )

        new_text = self_import.sub("", text)
        # Join broken asserts (run twice in case of stacked continuations).
        for _ in range(2):
            new_text = assert_break.sub(r"\1 ", new_text)

        if new_text != text:
            f.write_text(new_text)
            fixed.append(f.name)
    return fixed


def _run_tests(scripts_dir: Path) -> dict[str, dict[str, Any]]:
    """Execute all test files in the scripts directory and collect outputs.

    Returns a dict keyed by file name, each value ``{"output": str,
    "returncode": int}``. The pytest exit code is authoritative for pass/fail
    (0 = passed; non-zero = failure, including collection/import errors and
    "no tests collected" which is code 5). String-matching the output is
    unreliable — e.g. an ``ImportError`` in conftest yields no "FAILED"/"ERROR"
    token yet means nothing ran.
    """
    results: dict[str, dict[str, Any]] = {}

    test_files = sorted(scripts_dir.glob("test_*.py"))
    if not test_files:
        return {"_no_tests": {"output": "No test files found in generated scripts directory.", "returncode": 5}}

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
                timeout=900,
                env=env,
                cwd=str(scripts_dir),
            )
            output = proc.stdout + "\n" + proc.stderr
            results[test_file.name] = {"output": output[-6000:], "returncode": proc.returncode}
        except subprocess.TimeoutExpired:
            results[test_file.name] = {"output": "TIMEOUT: Test execution exceeded 900 seconds.", "returncode": 124}
        except Exception as exc:
            results[test_file.name] = {"output": f"EXECUTION ERROR: {exc}", "returncode": 1}

    return results


def _apply_fixes(fixed_files: list[dict[str, str]], scripts_dir: Path) -> list[str]:
    """Write corrected files back to disk.

    A fix is only applied if it parses as valid Python. The debug LLM sometimes
    returns truncated/garbled code; writing that would CLOBBER a working file and
    break collection. Invalid fixes are skipped so good code is never overwritten.
    """
    import ast as _ast

    written: list[str] = []
    for fix in fixed_files:
        fname = fix.get("file_name", "")
        code = fix.get("code", "")
        if not (fname and code):
            continue
        if fname.endswith(".py"):
            try:
                _ast.parse(code)
            except SyntaxError:
                logger.warning("Skipping debug fix for %s — proposed code is not valid Python.", fname)
                continue
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

    # Deterministic pre-flight fix for common collection-killing codegen bugs.
    _sanitize_generated_code(scripts_dir)

    run_results = _run_tests(scripts_dir)

    # pytest exit code is authoritative: 0 = passed, anything else = failure
    # (1 failed, 2 interrupted, 3 internal, 4 usage, 5 no tests collected).
    raw_outputs = {fname: r["output"] for fname, r in run_results.items()}
    failed_files = [fname for fname, r in run_results.items() if r.get("returncode", 1) != 0]

    # A run with no collectable tests (e.g. conftest import error → code 5, or the
    # "_no_tests" sentinel) is NOT a pass.
    ran_something = bool(run_results) and "_no_tests" not in run_results
    all_passed = ran_something and len(failed_files) == 0

    total_files = len(run_results)

    return {
        "execution_result": {
            "raw_outputs": raw_outputs,
            "total_files": total_files,
            "failed_files": failed_files,
            "all_passed": all_passed,
        },
        "pipeline_status": "passed" if all_passed else (
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
