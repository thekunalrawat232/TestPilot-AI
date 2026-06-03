"""Node 1 — Combined Requirement Analyst + Test Designer (single LLM call)."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage

from config.settings import get_llm
from prompts.requirement_and_design import REQUIREMENT_AND_DESIGN_PROMPT
from rag import ProjectRetriever
from .state import PipelineState
from .utils import extract_json, trim_context_to_fit
from .llm_cache import cached_llm_invoke


def requirement_and_design_node(state: PipelineState) -> dict[str, Any]:
    """Analyse the requirement AND design the test plan in a single LLM call.

    Replaces the two separate requirement_analyst + test_designer nodes,
    cutting LLM calls from 3 to 2 per pipeline run.
    """
    retriever = ProjectRetriever()

    context = retriever.query_formatted(state.raw_requirement, k=10)

    llm = get_llm()

    requirement_text = f"## Feature Requirement\n{state.raw_requirement}"
    context = trim_context_to_fit(
        system_prompt=REQUIREMENT_AND_DESIGN_PROMPT,
        user_content_parts=[requirement_text],
        context=context,
    )

    messages = [
        SystemMessage(content=REQUIREMENT_AND_DESIGN_PROMPT),
        HumanMessage(content=(
            f"{requirement_text}\n\n"
            f"## Project Context (from knowledge base)\n{context}"
        )),
    ]

    response = cached_llm_invoke(llm, messages, node_name="requirement_and_design")

    combined = None
    try:
        combined = extract_json(response.content)
    except (json.JSONDecodeError, Exception):
        # The model occasionally returns prose, a truncated runaway, or otherwise
        # non-JSON output. Retry ONCE with a strict reminder before giving up —
        # otherwise the empty plan makes the code generator improvise off-scope.
        retry_messages = messages + [
            HumanMessage(content=(
                "Your previous response could not be parsed as JSON. Respond AGAIN with "
                "ONLY the single JSON object specified in the system prompt: no prose, no "
                "markdown fences, no code copied from the context, every string value short "
                "and single-line. Output must start with '{' and end with '}'."
            )),
        ]
        try:
            response = cached_llm_invoke(llm, retry_messages, node_name="requirement_and_design_retry")
            combined = extract_json(response.content)
        except (json.JSONDecodeError, Exception) as exc:
            return {
                "requirement_analysis": {"raw_response": response.content},
                "test_plan": {},
                "retrieved_context": context,
                "error_log": [f"RequirementAndDesign JSON parse error (after retry): {exc}"],
                "pipeline_status": "running",
            }

    return {
        "requirement_analysis": combined.get("requirement_analysis", {}),
        "test_plan": combined.get("test_plan", {}),
        "retrieved_context": context,
        "pipeline_status": "running",
    }
