"""Node 3 — Automation Code Generator Agent."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage

from config.settings import get_llm, paths
from prompts.code_generator import CODE_GENERATOR_PROMPT
from .state import PipelineState
from .utils import extract_json, trim_context_to_fit
from .llm_cache import cached_llm_invoke


def _write_generated_files(code_output: dict[str, Any]) -> list[str]:
    """Persist all generated code to the file system.

    Returns list of file paths written.
    """
    written: list[str] = []
    scripts_dir = paths.generated_scripts
    scripts_dir.mkdir(parents=True, exist_ok=True)

    # Remove stale test files from previous runs before writing new ones
    for stale in scripts_dir.glob("test_*.py"):
        stale.unlink()

    # Page objects
    po_dir = scripts_dir / "page_objects"
    po_dir.mkdir(exist_ok=True)
    # Ensure page_objects is a package
    (po_dir / "__init__.py").write_text("")

    for po in code_output.get("page_objects", []):
        fpath = po_dir / Path(po["file_name"]).name
        fpath.write_text(po["code"])
        written.append(str(fpath))

    # Copy base_page.py from project root AFTER LLM files so it always wins
    project_root = Path(__file__).parent.parent
    src_base = project_root / "page_objects" / "base_page.py"
    if src_base.exists():
        shutil.copy2(src_base, po_dir / "base_page.py")
        written.append(str(po_dir / "base_page.py"))

    # Playwright test files
    for tf in code_output.get("playwright_tests", []):
        fpath = scripts_dir / tf["file_name"]
        fpath.write_text(tf["code"])
        written.append(str(fpath))

    # Selenium test files
    for tf in code_output.get("selenium_tests", []):
        fpath = scripts_dir / tf["file_name"]
        fpath.write_text(tf["code"])
        written.append(str(fpath))

    # conftest.py
    conftest = code_output.get("conftest")
    if conftest:
        fpath = scripts_dir / conftest["file_name"]
        fpath.write_text(conftest["code"])
        written.append(str(fpath))

    return written


def code_generator_node(state: PipelineState) -> dict[str, Any]:
    """Generate Playwright + Selenium automation scripts from the test plan.

    Inputs from state:
    - test_plan (from Node 2)
    - retrieved_context (RAG context)

    Outputs:
    - generated_code: the full code output dict + list of written file paths
    """
    llm = get_llm()

    test_plan_json = json.dumps(state.test_plan, indent=2)
    test_plan_part = f"## Test Plan\n```json\n{test_plan_json}\n```"

    # Feed the retrieved project context (real locators / page objects / existing
    # tests from the RAG knowledge base) to the code generator so generated
    # scripts reuse actual selectors instead of guessing. Trim to fit the window.
    context = trim_context_to_fit(
        system_prompt=CODE_GENERATOR_PROMPT,
        user_content_parts=[test_plan_part],
        context=state.retrieved_context or "",
    )

    user_content = (
        f"{test_plan_part}\n\n"
        f"## Project Context (real locators / page objects / existing tests — reuse these)\n"
        f"{context}"
    )

    messages = [
        SystemMessage(content=CODE_GENERATOR_PROMPT),
        HumanMessage(content=user_content),
    ]

    response = cached_llm_invoke(llm, messages, node_name="code_generator")

    try:
        code_output = extract_json(response.content)
    except (json.JSONDecodeError, Exception) as exc:
        return {
            "generated_code": {"raw_response": response.content},
            "error_log": [f"CodeGenerator JSON parse error: {exc}"],
        }

    # Write files to disk
    try:
        written_files = _write_generated_files(code_output)
        code_output["written_files"] = written_files
    except Exception as exc:
        return {
            "generated_code": code_output,
            "error_log": [f"CodeGenerator file write error: {exc}"],
        }

    return {"generated_code": code_output}
