"""ChromaDB vector store management for project context."""

from __future__ import annotations

import hashlib
import logging
import shutil
from pathlib import Path
from typing import Sequence

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config.settings import rag_config, paths
from .embeddings import get_embedding_model

logger = logging.getLogger(__name__)

# File extensions we know how to ingest
_CODE_EXTS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".html", ".css",
    ".json", ".yaml", ".yml", ".toml", ".md", ".txt",
    ".feature", ".gherkin", ".robot",
}

# Directory names to skip when scanning (dependency/cache/artifact noise).
_SKIP_DIRS = {
    "node_modules", "__pycache__", ".git", ".pytest_cache", ".venv", "venv",
    ".mypy_cache", ".ruff_cache", "dist", "build", ".next", ".cache",
    "screenshots", "reports", "logs", ".playwright-mcp", "test-results",
    "playwright-report", "allure-results",
}


def _collect_files(directory: Path) -> list[Path]:
    """Recursively collect indexable files from a directory.

    Skips dependency/cache/artifact directories (see ``_SKIP_DIRS``) so that
    ingesting a real project doesn't pull in node_modules or test reports.
    """
    if not directory.exists():
        return []
    collected: list[Path] = []
    for f in directory.rglob("*"):
        if not f.is_file():
            continue
        if f.suffix not in _CODE_EXTS:
            continue
        if any(part in _SKIP_DIRS for part in f.parts):
            continue
        try:
            if f.stat().st_size >= 500_000:
                continue
        except OSError:
            continue
        collected.append(f)
    return collected


def _file_to_document(path: Path, source_label: str) -> Document:
    """Convert a file to a LangChain Document with metadata."""
    text = path.read_text(errors="replace")
    return Document(
        page_content=text,
        metadata={
            "source": str(path),
            "source_type": source_label,
            "filename": path.name,
            "extension": path.suffix,
            "content_hash": hashlib.sha256(text.encode()).hexdigest()[:16],
        },
    )


def _load_external_documents() -> list[Document]:
    """Load documents from external project folders (read-only).

    Configured via EXTERNAL_CONTEXT_DIRS. Used to ingest another repo's
    locators / page objects / existing tests without copying or modifying it.
    """
    docs: list[Document] = []
    for raw in rag_config.external_context_dirs:
        directory = Path(raw).expanduser()
        if not directory.exists():
            logger.warning("EXTERNAL_CONTEXT_DIRS entry not found, skipping: %s", directory)
            continue
        files = _collect_files(directory)
        logger.info("Ingesting %d files from external context dir: %s", len(files), directory)
        for fpath in files:
            docs.append(_file_to_document(fpath, "external_project"))
    return docs


def _load_all_context_documents() -> list[Document]:
    """Load documents from every context source directory."""
    sources = [
        (paths.context_codebase, "codebase"),
        (paths.context_api, "api_schema"),
        (paths.context_docs, "documentation"),
        (paths.context_tests, "existing_test"),
    ]
    docs: list[Document] = []
    for directory, label in sources:
        for fpath in _collect_files(directory):
            docs.append(_file_to_document(fpath, label))

    # External project folders (e.g. your QA automation framework's locators)
    docs.extend(_load_external_documents())
    return docs


def _split_documents(docs: list[Document]) -> list[Document]:
    """Chunk documents for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=rag_config.chunk_size,
        chunk_overlap=rag_config.chunk_overlap,
        separators=["\nclass ", "\ndef ", "\n\n", "\n", " "],
    )
    return splitter.split_documents(docs)


def build_vectorstore(force_rebuild: bool = False) -> Chroma:
    """Build or load the Chroma vector store.

    If the store already exists on disk and ``force_rebuild`` is False,
    it is loaded directly.  Otherwise all context directories are
    scanned, chunked, embedded and persisted.
    """
    persist_dir = Path(rag_config.persist_dir)
    embeddings = get_embedding_model()

    if persist_dir.exists() and not force_rebuild:
        return Chroma(
            collection_name=rag_config.collection_name,
            persist_directory=str(persist_dir),
            embedding_function=embeddings,
        )

    # Force rebuild: wipe the existing collection first. Without this,
    # Chroma.from_documents APPENDS to the existing collection, leaving stale
    # chunks behind and duplicating everything on every rebuild.
    if force_rebuild and persist_dir.exists():
        try:
            existing = Chroma(
                collection_name=rag_config.collection_name,
                persist_directory=str(persist_dir),
                embedding_function=embeddings,
            )
            existing.delete_collection()
            logger.info("Cleared existing '%s' collection before rebuild.", rag_config.collection_name)
        except Exception as exc:
            logger.warning("Could not delete existing collection (%s); rebuilding fresh dir.", exc)
            shutil.rmtree(persist_dir, ignore_errors=True)

    raw_docs = _load_all_context_documents()
    if not raw_docs:
        # Return empty store so downstream code still works
        return Chroma(
            collection_name=rag_config.collection_name,
            persist_directory=str(persist_dir),
            embedding_function=embeddings,
        )

    chunks = _split_documents(raw_docs)

    store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=rag_config.collection_name,
        persist_directory=str(persist_dir),
    )
    return store


def add_documents_to_store(
    store: Chroma, docs: Sequence[Document]
) -> None:
    """Incrementally add new documents to an existing store."""
    chunks = _split_documents(list(docs))
    if chunks:
        store.add_documents(chunks)
