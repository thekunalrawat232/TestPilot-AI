"""Node 1 — Planner. Requirement -> small STRUCTURED test plan (1 LLM call).

The model never writes code. It picks locator NAMES from the real catalog and
emits a plan of steps from a fixed action vocabulary. A deterministic renderer
(agents/render.py) turns the plan into runnable Playwright tests.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage

from config.settings import get_llm
from prompts.requirement_and_design import REQUIREMENT_AND_DESIGN_PROMPT
from .render import extract_locator_catalog, format_locator_catalog
from .state import PipelineState
from .utils import extract_json
from .llm_cache import cached_llm_invoke


def _fallback_plan(requirement: str, catalog: dict[str, dict]) -> dict[str, Any]:
    """A deterministic minimal plan so the pipeline NEVER crashes on a bad LLM
    response: navigate to the likely page and assert a heading-ish locator."""
    req = requirement.lower()
    path = "/forms" if "form" in req else "/"
    heading = None
    for info in catalog.values():
        for name in info["selectors"]:
            if "HEADING" in name or "TITLE" in name or "PAGE" in name:
                heading = name
                break
        if heading:
            break
    steps: list[dict] = [{"action": "goto", "target": path}]
    if heading:
        steps.append({"action": "expect_visible", "target": heading})
    return {
        "feature_name": "fallback_plan",
        "summary": "Fallback plan (LLM plan unavailable).",
        "tests": [{"id": "TC_001", "title": "Page loads", "steps": steps}],
    }


def requirement_and_design_node(state: PipelineState) -> dict[str, Any]:
    """Produce a structured test plan from the requirement (single LLM call)."""
    catalog = extract_locator_catalog("", state.raw_requirement)
    catalog_str = format_locator_catalog(catalog)

    llm = get_llm()
    user = (
        f"## Feature Requirement\n{state.raw_requirement}\n\n"
        f"## LOCATOR CATALOG (reference these NAMES only)\n{catalog_str}"
    )
    messages = [
        SystemMessage(content=REQUIREMENT_AND_DESIGN_PROMPT),
        HumanMessage(content=user),
    ]

    plan: dict[str, Any] | None = None
    err: str | None = None
    try:
        plan = extract_json(cached_llm_invoke(llm, messages, node_name="planner").content)
    except Exception:
        # One strict retry, then a deterministic fallback — never crash.
        retry = messages + [HumanMessage(content=(
            "Your previous response was not valid JSON. Return ONLY the JSON plan object "
            "per the schema — no prose, no fences. Start with '{' and end with '}'."
        ))]
        try:
            plan = extract_json(cached_llm_invoke(llm, retry, node_name="planner_retry").content)
        except Exception as exc:
            err = f"Planner JSON parse error (after retry): {exc}"

    if not isinstance(plan, dict) or not plan.get("tests"):
        plan = _fallback_plan(state.raw_requirement, catalog)
        err = err or "Planner returned no usable tests; used fallback plan."

    return {
        "requirement_analysis": {
            "feature_name": plan.get("feature_name", ""),
            "summary": plan.get("summary", ""),
        },
        "test_plan": plan,
        "retrieved_context": catalog_str,
        "error_log": [err] if err else [],
        "pipeline_status": "running",
    }
