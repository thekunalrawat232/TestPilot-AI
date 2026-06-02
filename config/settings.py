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
    provider: str = os.getenv("LLM_PROVIDER", "openai")
    model: str = os.getenv("LLM_MODEL", "gpt-4o")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))


@dataclass(frozen=True)
class RAGConfig:
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", str(PROJECT_ROOT / "rag_store"))
    collection_name: str = "project_context"
    chunk_size: int = 1500
    chunk_overlap: int = 200
    retrieval_k: int = 8


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

# Context window limits per model family (input tokens)
MODEL_CONTEXT_LIMITS: dict[str, int] = {
    # OpenAI
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 8_192,
    "gpt-3.5-turbo": 16_385,
    # Anthropic
    "claude-opus-4-20250514": 200_000,
    "claude-sonnet-4-20250514": 200_000,
    "claude-sonnet-4-6": 200_000,
    "claude-sonnet-4-5-20241022": 200_000,
    "claude-haiku-4-5-20251001": 200_000,
    # Google Gemini
    "gemini-2.5-flash": 1_048_576,
    "gemini-2.0-flash": 1_048_576,
    "gemini-1.5-flash": 1_048_576,
    "gemini-1.5-pro": 2_097_152,
    # Groq
    "llama-3.3-70b-versatile": 128_000,
    "llama-3.1-8b-instant": 128_000,
    "mixtral-8x7b-32768": 32_768,
}

# Reserve tokens for the LLM response
RESPONSE_TOKEN_RESERVE = 4_096


# Groq free tier TPM cap — keep input well under 12k to leave room for output
GROQ_FREE_TIER_INPUT_LIMIT = 6_000


def get_context_limit() -> int:
    """Return the max input tokens for the configured model."""
    model_limit = MODEL_CONTEXT_LIMITS.get(llm_config.model, 16_000) - RESPONSE_TOKEN_RESERVE
    if llm_config.provider.lower() == "groq":
        return min(model_limit, GROQ_FREE_TIER_INPUT_LIMIT)
    return model_limit


def _groq_rate_limit_aware_invoke(llm):
    """Wrap a ChatGroq instance so invoke() auto-retries on rate-limit errors."""
    from langchain_groq import ChatGroq

    class _RateLimitGroq(ChatGroq):
        def invoke(self, messages, *args, **kwargs):
            for attempt in range(1, 4):
                try:
                    return super().invoke(messages, *args, **kwargs)
                except Exception as e:
                    msg = str(e)
                    if "413" in msg or "429" in msg or "rate_limit_exceeded" in msg or "RESOURCE_EXHAUSTED" in msg:
                        wait = 62
                        logger.warning(
                            "Groq rate limit hit (attempt %d/3). Waiting %ds before retry...",
                            attempt, wait,
                        )
                        print(f"\n⏳ Groq rate limit hit — waiting {wait}s before retry (attempt {attempt}/3)...")
                        time.sleep(wait)
                    else:
                        raise
                    if attempt == 3:
                        raise
            return super().invoke(messages, *args, **kwargs)

    wrapped = _RateLimitGroq(
        model=llm.model_name,
        temperature=llm.temperature,
        max_retries=llm.max_retries,
        max_tokens=llm.max_tokens,
    )
    return wrapped


def get_llm():
    """Factory that returns the right LangChain chat model based on LLM_PROVIDER."""
    provider = llm_config.provider.lower()

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=llm_config.model,
            temperature=llm_config.temperature,
            max_retries=2,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=llm_config.model,
            temperature=llm_config.temperature,
            max_retries=2,
        )

    if provider == "google":
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

    if provider == "groq":
        from langchain_groq import ChatGroq
        llm = ChatGroq(
            model=llm_config.model,
            temperature=llm_config.temperature,
            max_retries=2,
            max_tokens=4096,
        )
        return _groq_rate_limit_aware_invoke(llm)

    raise ValueError(
        f"Unsupported LLM_PROVIDER: '{provider}'. Use 'openai', 'anthropic', 'google', or 'groq'."
    )
