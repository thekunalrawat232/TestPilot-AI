"""Node 2 — Render (DETERMINISTIC, no LLM call).

Turns the structured test plan into a runnable Playwright suite: copies the
fixed harness (conftest + support), writes the real locator package, and renders
the test file. Output is always valid Python regardless of the plan's quality.
"""

from __future__ import annotations

import re
import shutil
from typing import Any

from config.settings import paths
from .render import (
    extract_locator_catalog, render_test_file, write_harness, write_locator_package,
)
from .state import PipelineState


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")
    return s[:40] or "suite"


def _clean_scripts_dir(scripts_dir) -> None:
    """Remove stale generated artifacts from previous runs."""
    for stale in scripts_dir.glob("test_*.py"):
        stale.unlink()
    for name in ("conftest.py", "_support.py"):
        f = scripts_dir / name
        if f.exists():
            f.unlink()
    for sub in ("page_objects", "locators", "__pycache__", ".pytest_cache"):
        d = scripts_dir / sub
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)


def code_generator_node(state: PipelineState) -> dict[str, Any]:
    """Render the plan into a Playwright suite (deterministic)."""
    scripts_dir = paths.generated_scripts
    scripts_dir.mkdir(parents=True, exist_ok=True)
    _clean_scripts_dir(scripts_dir)

    feature = state.requirement_analysis.get("feature_name") or ""
    catalog = extract_locator_catalog(feature, state.raw_requirement)

    written: list[str] = []
    try:
        written += write_harness(scripts_dir)
        written += write_locator_package(scripts_dir, feature, state.raw_requirement)
        code = render_test_file(state.test_plan or {}, catalog)
        test_path = scripts_dir / f"test_{_slug(feature)}_pw.py"
        test_path.write_text(code)
        written.append(str(test_path))
    except Exception as exc:
        return {
            "generated_code": {"written_files": written},
            "error_log": [f"Render error: {exc}"],
        }

    return {"generated_code": {"written_files": written, "test_file": str(test_path)}}
