"""Central configuration for the AI Test Agent system."""

from __future__ import annotations

import os
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

# Module-level tracker for Google RPM pacing (5 RPM = 1 call per 12s)
_last_google_call_time: float = 0.0
_GOOGLE_MIN_INTERVAL: float = 13.0  # 12s + 1s buffer

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class LLMConfig:
    # Gemini is the only supported provider.
    provider: str = "google"
    model: str = os.getenv("LLM_MODEL", "gemini-2.5-flash")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))


def _parse_external_dirs() -> tuple[str, ...]:
    """Read EXTERNAL_CONTEXT_DIRS env var into a tuple of paths.

    Accepts multiple directories separated by the OS path separator (':' on
    Linux/macOS, ';' on Windows) or by commas. These directories are ingested
    into the RAG knowledge base read-only — nothing in them is ever modified.
    """
    raw = os.getenv("EXTERNAL_CONTEXT_DIRS", "")
    parts: list[str] = []
    for chunk in raw.replace(os.pathsep, ",").split(","):
        cleaned = chunk.strip()
        if cleaned:
            parts.append(cleaned)
    return tuple(parts)


@dataclass(frozen=True)
class RAGConfig:
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", str(PROJECT_ROOT / "rag_store"))
    collection_name: str = "project_context"
    chunk_size: int = 1500
    chunk_overlap: int = 200
    retrieval_k: int = 8
    # Extra (external) project folders to ingest read-only, e.g. another repo's
    # locators / page objects. Configured via EXTERNAL_CONTEXT_DIRS in .env.
    external_context_dirs: tuple[str, ...] = field(default_factory=_parse_external_dirs)


@dataclass(frozen=True)
class ExecutionConfig:
    target_base_url: str = os.getenv("TARGET_BASE_URL", "http://localhost:3000")
    headless: bool = os.getenv("HEADLESS_BROWSER", "true").lower() == "true"
    max_retries: int = int(os.getenv("MAX_RETRY_ATTEMPTS", "3"))
    debug: bool = os.getenv("DEBUG_MODE", "true").lower() == "true"
    test_timeout_ms: int = 30_000
    navigation_timeout_ms: int = 15_000


@dataclass(frozen=True)
class PathConfig:
    generated_tests: Path = field(default_factory=lambda: PROJECT_ROOT / "generated" / "test_cases")
    generated_scripts: Path = field(default_factory=lambda: PROJECT_ROOT / "generated" / "automation_scripts")
    generated_reports: Path = field(default_factory=lambda: PROJECT_ROOT / "generated" / "reports")
    page_objects: Path = field(default_factory=lambda: PROJECT_ROOT / "page_objects")
    context_codebase: Path = field(default_factory=lambda: PROJECT_ROOT / "context" / "codebase")
    context_api: Path = field(default_factory=lambda: PROJECT_ROOT / "context" / "api_schemas")
    context_docs: Path = field(default_factory=lambda: PROJECT_ROOT / "context" / "docs")
    context_tests: Path = field(default_factory=lambda: PROJECT_ROOT / "context" / "existing_tests")


llm_config = LLMConfig()
rag_config = RAGConfig()
exec_config = ExecutionConfig()
paths = PathConfig()

# Context window limits per Gemini model (input tokens)
MODEL_CONTEXT_LIMITS: dict[str, int] = {
    "gemini-2.5-flash": 1_048_576,
    "gemini-2.0-flash": 1_048_576,
    "gemini-1.5-flash": 1_048_576,
    "gemini-1.5-pro": 2_097_152,
}

# Reserve tokens for the LLM response
RESPONSE_TOKEN_RESERVE = 4_096


def get_context_limit() -> int:
    """Return the max input tokens for the configured Gemini model."""
    return MODEL_CONTEXT_LIMITS.get(llm_config.model, 1_048_576) - RESPONSE_TOKEN_RESERVE


def get_llm():
    """Return the Gemini chat model (the only supported provider).

    Wraps ``ChatGoogleGenerativeAI`` with 5-RPM pacing and automatic backoff
    for transient 503 (UNAVAILABLE) and 429 (RESOURCE_EXHAUSTED) errors.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI
    import re as _re

    class _RetryGoogleLLM(ChatGoogleGenerativeAI):
        def invoke(self, messages, *args, **kwargs):
            global _last_google_call_time

            # RPM pacing: enforce minimum interval between calls
            elapsed = time.time() - _last_google_call_time
            if _last_google_call_time > 0 and elapsed < _GOOGLE_MIN_INTERVAL:
                wait = _GOOGLE_MIN_INTERVAL - elapsed
                print(f"\n⏳ RPM pacing — waiting {wait:.1f}s (5 RPM limit)...")
                time.sleep(wait)

            for attempt in range(1, 5):
                try:
                    _last_google_call_time = time.time()
                    return super().invoke(messages, *args, **kwargs)
                except Exception as e:
                    err = str(e)
                    if "503" in err or "UNAVAILABLE" in err:
                        wait = 15 * attempt
                        print(f"\n⏳ Gemini 503 — waiting {wait}s before retry (attempt {attempt}/4)...")
                        time.sleep(wait)
                        if attempt == 4:
                            raise
                    elif "429" in err or "RESOURCE_EXHAUSTED" in err:
                        m = _re.search(r'retryDelay.*?(\d+)s', err)
                        wait = int(m.group(1)) + 5 if m else 65
                        print(f"\n⏳ Gemini 429 — waiting {wait}s before retry (attempt {attempt}/4)...")
                        time.sleep(wait)
                        if attempt == 4:
                            raise
                    else:
                        raise

    return _RetryGoogleLLM(
        model=llm_config.model,
        temperature=llm_config.temperature,
        max_retries=2,
    )
