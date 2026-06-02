"""Disk-based LLM response cache to avoid redundant API calls."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent.parent / ".llm_cache"


class _CachedResponse:
    """Minimal response object that matches the .content interface nodes expect."""
    def __init__(self, content: str):
        self.content = content


def cached_llm_invoke(llm, messages: list[BaseMessage], node_name: str = ""):
    """Invoke the LLM with transparent disk caching.

    On a cache hit, returns the cached response without making an API call.
    On a cache miss, calls the LLM, caches the result, and returns it.
    """
    raw = node_name + "".join(
        (getattr(m, "type", "") + (m.content if isinstance(m.content, str) else ""))
        for m in messages
    )
    cache_key = hashlib.md5(raw.encode()).hexdigest()

    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"{cache_key}.json"

    if cache_file.exists():
        cached = json.loads(cache_file.read_text())
        logger.info("[llm_cache] HIT  node='%s' key=%s", node_name, cache_key[:8])
        print(f"[cache] Hit for '{node_name}' — skipping LLM call")
        return _CachedResponse(cached["content"])

    response = llm.invoke(messages)
    cache_file.write_text(json.dumps({"content": response.content}, ensure_ascii=False))
    logger.info("[llm_cache] MISS node='%s' — saved to %s", node_name, cache_key[:8])
    return response
