"""Shared utilities for all agent nodes."""

from __future__ import annotations

import json
import logging
from typing import Any

import tiktoken
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage

from config.settings import llm_config, get_context_limit

logger = logging.getLogger(__name__)

# Tiktoken encoding — cl100k_base is a reasonable token-count approximation
# for Gemini (which has no public tiktoken encoding).
_ENCODING: tiktoken.Encoding | None = None


def _get_encoding() -> tiktoken.Encoding:
    global _ENCODING
    if _ENCODING is None:
        try:
            _ENCODING = tiktoken.encoding_for_model(llm_config.model)
        except KeyError:
            _ENCODING = tiktoken.get_encoding("cl100k_base")
    return _ENCODING


def count_tokens(text: str) -> int:
    """Count tokens in a string using tiktoken."""
    return len(_get_encoding().encode(text))


def count_message_tokens(messages: list[BaseMessage]) -> int:
    """Count total tokens across a list of LangChain messages.

    Adds a small overhead per message for role/formatting tokens.
    """
    total = 0
    for msg in messages:
        total += 4  # role + formatting overhead per message
        total += count_tokens(msg.content if isinstance(msg.content, str) else "")
    total += 2  # reply priming
    return total


def trim_context_to_fit(
    system_prompt: str,
    user_content_parts: list[str],
    context: str,
    max_tokens: int | None = None,
) -> str:
    """Trim the RAG context so the full prompt fits within the context window.

    Parameters
    ----------
    system_prompt:
        The system message content.
    user_content_parts:
        Non-context portions of the user message (e.g., requirement text,
        test plan JSON). These are never trimmed.
    context:
        The RAG-retrieved context string. This will be truncated if needed.
    max_tokens:
        Override the auto-detected context limit.

    Returns
    -------
    The (possibly truncated) context string.
    """
    limit = max_tokens or get_context_limit()

    # Tokens used by everything except context
    fixed_tokens = count_tokens(system_prompt)
    for part in user_content_parts:
        fixed_tokens += count_tokens(part)
    fixed_tokens += 20  # message formatting overhead

    available_for_context = limit - fixed_tokens
    if available_for_context <= 0:
        logger.warning(
            "Prompt without context already exceeds limit (%d tokens vs %d limit). "
            "Sending with empty context.",
            fixed_tokens, limit,
        )
        return "(Context omitted — prompt too large.)"

    context_tokens = count_tokens(context)
    if context_tokens <= available_for_context:
        logger.info(
            "Context fits: %d tokens (limit %d, used %d for prompt).",
            context_tokens, limit, fixed_tokens,
        )
        return context

    # Truncate context by splitting on document separators and keeping as many
    # complete chunks as possible.
    enc = _get_encoding()
    encoded = enc.encode(context)
    truncated = enc.decode(encoded[:available_for_context])

    # Try to cut at the last clean separator to avoid mid-document cuts
    last_sep = truncated.rfind("\n\n---\n\n")
    if last_sep > len(truncated) // 2:
        truncated = truncated[:last_sep]

    trimmed_tokens = count_tokens(truncated)
    logger.warning(
        "Context trimmed: %d → %d tokens (limit %d, prompt uses %d).",
        context_tokens, trimmed_tokens, limit, fixed_tokens,
    )
    return truncated + "\n\n... [context truncated to fit context window]"


def _sanitize_llm_json(text: str) -> str:
    """Fix common LLM JSON generation mistakes before parsing.

    Handles:
    - Triple-quoted VALUES: ``"code": '''...'''`` or ``"code": \"\"\"...\"\"\"``.
      LLMs frequently delimit big code blobs with triple quotes (which are NOT
      valid JSON), and the code itself often contains ''' Python docstrings.
      We rewrite each triple-quoted value into a properly escaped JSON string.
    - JS string repeat expressions: ``"x".repeat(N)`` → ``"xxx..."`` (N chars)
    """
    import re

    # Convert triple-quoted values to valid JSON strings. Restrict to the known
    # multi-line fields ("code", "requirements_txt") so a ``: '''`` that appears
    # INSIDE the embedded Python (dict literals, etc.) is never mistaken for a
    # JSON value boundary — matching those mangles the whole structure.
    # The closing delimiter is the triple-quote followed by JSON punctuation
    # (, } ]), so nested Python docstrings (followed by code, not punctuation) are
    # correctly skipped by the non-greedy match + trailing anchor.
    triple_value = re.compile(
        r'("(?:code|requirements_txt)"\s*:\s*)("""|\'\'\')(.*?)\2(\s*[,}\]])',
        re.DOTALL,
    )

    def _to_json_string(m: re.Match) -> str:
        body = m.group(3)
        return m.group(1) + json.dumps(body) + m.group(4)

    # Run a few passes in case of adjacent values the first pass consumed around.
    for _ in range(3):
        new_text = triple_value.sub(_to_json_string, text)
        if new_text == text:
            break
        text = new_text

    # Replace "char".repeat(N) with an actual repeated string literal.
    def _expand_repeat(m: re.Match) -> str:
        char = m.group(1)   # the character inside quotes
        count = int(m.group(2))
        # Cap at a safe length to avoid giant strings in test data
        return '"' + (char * min(count, 256)) + '"'

    text = re.sub(r'"(.)"\s*\.\s*repeat\s*\(\s*(\d+)\s*\)', _expand_repeat, text)

    return text


def extract_json(text: str) -> dict[str, Any]:
    """Robustly extract JSON from LLM response that may contain markdown fences."""
    import re

    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        start = 1 if lines[0].startswith("```") else 0
        end = -1 if lines[-1].strip() == "```" else len(lines)
        text = "\n".join(lines[start:end]).strip()

    # strict=False tolerates literal control characters (newlines/tabs) inside
    # string values — a very common LLM mistake in large multi-line responses
    # that otherwise raises "Invalid control character".
    try:
        return json.loads(text, strict=False)
    except json.JSONDecodeError:
        pass

    # Apply LLM-specific sanitisation and retry.
    fixed = _sanitize_llm_json(text)
    try:
        return json.loads(fixed, strict=False)
    except json.JSONDecodeError:
        pass

    # Last resort: extract the outermost {...} block and try again.
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        candidate = _sanitize_llm_json(match.group(0))
        return json.loads(candidate, strict=False)

    raise json.JSONDecodeError("No valid JSON object found", text, 0)
