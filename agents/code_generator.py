"""Node 3 — Automation Code Generator Agent."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage

import re

from config.settings import get_llm, paths, rag_config
from prompts.code_generator import CODE_GENERATOR_PROMPT
from rag import ProjectRetriever
from .state import PipelineState
from .utils import extract_json, trim_context_to_fit
from .llm_cache import cached_llm_invoke

_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "page", "list",
    "bug", "bugs", "hunt", "find", "found", "out", "all", "section", "first",
    "click", "new", "add", "user", "able", "verify", "test", "tests", "do", "not",
    "go", "to", "on", "of", "a", "an", "as", "is", "are", "it", "its",
}
_SKIP_DIRS = {"node_modules", "__pycache__", ".git", ".pytest_cache", ".venv", "venv"}


def _feature_keywords(feature: str, requirement: str) -> set[str]:
    """Extract significant, singularised keywords from the feature/requirement."""
    words = re.findall(r"[a-zA-Z]{3,}", f"{feature} {requirement}".lower())
    kws = set()
    for w in words:
        if w in _STOPWORDS:
            continue
        kws.add(w)
        if w.endswith("s"):
            kws.add(w[:-1])  # crude singular
    return kws


# Locator files that are near-universal prerequisites (every admin test logs in
# first), so they are injected regardless of the feature being tested.
_PREREQUISITE_LOCATOR_HINTS = ("login", "signin", "sign_in", "auth")


def _collect_locator_files(feature: str, requirement: str) -> str:
    """Deterministically pull locator/page-object files for this feature.

    Embeddings rank a flat ``NAME = "selector"`` constants file poorly, so we
    bypass semantic search: scan the configured external project dirs for
    ``*locator*.py`` files whose name matches a feature keyword (or is a login/
    auth prerequisite) and inject their FULL content. Guarantees the real
    selectors — including the login flow every test depends on — reach the model.
    """
    keywords = _feature_keywords(feature, requirement)
    blocks: list[str] = []
    seen: set[str] = set()
    for raw in rag_config.external_context_dirs:
        base = Path(raw).expanduser()
        if not base.exists():
            continue
        for f in sorted(base.rglob("*.py")):
            if any(p in _SKIP_DIRS for p in f.parts):
                continue
            name = f.name.lower()
            if "locator" not in name:
                continue
            stem = name.replace("_locators.py", "").replace("_locator.py", "").replace("locators.py", "").replace(".py", "")
            stem_words = set(re.findall(r"[a-z]{3,}", stem))
            is_feature = bool(stem_words & keywords)
            is_prerequisite = any(h in name for h in _PREREQUISITE_LOCATOR_HINTS)
            # Include feature-matching locators OR login/auth prerequisites.
            if not (is_feature or is_prerequisite):
                continue
            try:
                content = f.read_text(errors="replace")
            except Exception:
                continue
            if f.name in seen:
                continue
            seen.add(f.name)
            label = "PREREQUISITE (login/auth)" if is_prerequisite and not is_feature else "feature"
            blocks.append(f"# FILE: {f.name}  [{label}]\n{content[:8000]}")
    return "\n\n".join(blocks)


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
    # Remove stale page objects from previous runs (base_page.py is re-copied
    # below). Leftover objects from an earlier feature confuse debugging.
    for stale in po_dir.glob("*.py"):
        stale.unlink()
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

    # Playwright test files (Playwright-only project — no Selenium)
    for tf in code_output.get("playwright_tests", []):
        fpath = scripts_dir / tf["file_name"]
        fpath.write_text(tf["code"])
        written.append(str(fpath))

    # conftest.py
    conftest = code_output.get("conftest")
    if conftest:
        fpath = scripts_dir / conftest["file_name"]
        fpath.write_text(conftest["code"])
        written.append(str(fpath))

    # Always write a pytest.ini so unregistered custom marks don't warn and the
    # cache is disabled (deterministic — independent of the LLM output).
    pytest_ini = (
        "[pytest]\n"
        "addopts = -p no:cacheprovider\n"
        "filterwarnings =\n"
        "    ignore::pytest.PytestUnknownMarkWarning\n"
    )
    (scripts_dir / "pytest.ini").write_text(pytest_ini)
    written.append(str(scripts_dir / "pytest.ini"))

    return written


def code_generator_node(state: PipelineState) -> dict[str, Any]:
    """Generate Playwright automation scripts from the test plan.

    Inputs from state:
    - test_plan (from Node 2)
    - retrieved_context (RAG context)

    Outputs:
    - generated_code: the full code output dict + list of written file paths
    """
    llm = get_llm()

    test_plan_json = json.dumps(state.test_plan, indent=2)
    test_plan_part = f"## Test Plan\n```json\n{test_plan_json}\n```"

    # Targeted retrieval for locators/page objects. The requirement-level context
    # stored in state often ranks test files above the locator definitions, so we
    # run a second retrieval explicitly aimed at selector sources for this feature
    # and put it FIRST (trim_context_to_fit keeps the head), so the real locators
    # survive truncation and the generator can reuse them verbatim.
    feature = (
        state.requirement_analysis.get("feature_name")
        or state.test_plan.get("feature_name")
        or ""
    )
    locator_query = (
        f"{feature} {state.raw_requirement} "
        "page object locators selectors css xpath data-testid input button table"
    )
    try:
        locator_context = ProjectRetriever().query_formatted(locator_query, k=12)
    except Exception:
        locator_context = ""

    # Deterministic, embedding-free injection of the real locator files for this
    # feature. Placed FIRST so it always survives context trimming.
    injected_locators = _collect_locator_files(feature, state.raw_requirement)

    parts: list[str] = []
    if injected_locators:
        parts.append(
            "### REAL LOCATOR DEFINITIONS for this feature — COPY THESE SELECTORS VERBATIM\n"
            "These are the project's actual locators. Use them exactly; do NOT invent data-testid selectors.\n\n"
            + injected_locators
        )
    if locator_context and locator_context not in (state.retrieved_context or ""):
        parts.append("### Related page objects & tests (reference)\n" + locator_context)
    if state.retrieved_context:
        parts.append("### Additional project context\n" + state.retrieved_context)
    combined_context = "\n\n---\n\n".join(parts) if parts else "(No project context found.)"

    context = trim_context_to_fit(
        system_prompt=CODE_GENERATOR_PROMPT,
        user_content_parts=[test_plan_part],
        context=combined_context,
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

    code_output = None
    try:
        code_output = extract_json(response.content)
    except (json.JSONDecodeError, Exception):
        # Embedding code in JSON often breaks parsing (unescaped quotes inside a
        # "code" value, etc.). Retry ONCE with a strict reminder before giving up.
        retry_messages = messages + [
            HumanMessage(content=(
                "Your previous response was not valid JSON and could not be parsed. "
                "Return ONLY the JSON object. For EVERY \"code\" field, delimit the value "
                "with triple-single-quotes like \"code\": '''<python here>''' so the inner "
                "double quotes in the code do not break the JSON. No markdown fences, no "
                "prose. Output must start with '{' and end with '}'."
            )),
        ]
        try:
            response = cached_llm_invoke(llm, retry_messages, node_name="code_generator_retry")
            code_output = extract_json(response.content)
        except (json.JSONDecodeError, Exception) as exc:
            return {
                "generated_code": {"raw_response": response.content},
                "error_log": [f"CodeGenerator JSON parse error (after retry): {exc}"],
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
