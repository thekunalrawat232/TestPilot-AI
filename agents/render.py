"""Deterministic render layer (replaces LLM code generation).

The LLM only produces a small structured test plan. This module:
  1. extracts the real locator catalog from the external framework (to tell the
     planner which locator NAMES exist), and
  2. writes the fixed harness + a real ``locators/`` package + the rendered
     Playwright test file — all deterministically, so output is ALWAYS valid
     Python regardless of what the model proposes.
"""
from __future__ import annotations

import ast
import json
import re
import shutil
from pathlib import Path

from config.settings import rag_config

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_HARNESS_DIR = _PROJECT_ROOT / "harness"

_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "page", "list",
    "bug", "bugs", "hunt", "find", "found", "out", "all", "section", "first",
    "click", "new", "add", "user", "able", "verify", "test", "tests", "do", "not",
    "go", "to", "on", "of", "a", "an", "as", "is", "are", "it", "its", "discovery",
}
_SKIP_DIRS = {"node_modules", "__pycache__", ".git", ".pytest_cache", ".venv", "venv"}
_LOGIN_HINTS = ("login", "signin", "sign_in", "auth")


# ---------------------------------------------------------------------------
# Locator catalog (real selectors from the external framework)
# ---------------------------------------------------------------------------

def _keywords(feature: str, requirement: str) -> set[str]:
    words = re.findall(r"[a-zA-Z]{3,}", f"{feature} {requirement}".lower())
    kws: set[str] = set()
    for w in words:
        if w in _STOPWORDS:
            continue
        kws.add(w)
        if w.endswith("s"):
            kws.add(w[:-1])
    return kws


def _feature_locator_files(feature: str, requirement: str) -> list[Path]:
    """Locator files whose name matches the feature (login files excluded — the
    harness handles auth)."""
    keywords = _keywords(feature, requirement)
    out: list[Path] = []
    seen: set[str] = set()
    for raw in rag_config.external_context_dirs:
        base = Path(raw).expanduser()
        if not base.exists():
            continue
        for f in sorted(base.rglob("*.py")):
            if any(p in _SKIP_DIRS for p in f.parts):
                continue
            # Only use admin-portal locators — member-portal locators have
            # different UI elements and cause wrong assertions on the admin app.
            if "member" in f.parts:
                continue
            # Skip ledger-specific locator files unless the requirement explicitly
            # mentions ledger — they add confusing sub-flow locators to the catalog.
            if "ledger" in f.name.lower() and "ledger" not in requirement.lower():
                continue
            name = f.name.lower()
            if "locator" not in name or f.name in seen:
                continue
            if any(h in name for h in _LOGIN_HINTS):
                continue
            stem = re.sub(r"_?locators?\.py$", "", name)
            if set(re.findall(r"[a-z]{3,}", stem)) & keywords:
                seen.add(f.name)
                out.append(f)
    return out


def extract_locator_catalog(feature: str, requirement: str) -> dict[str, dict]:
    """Parse the feature locator files via AST → ::

        {ClassName: {"module": <stem>, "selectors": {ATTR: selector}}}

    AST only (no import/exec), so a malformed file can never crash us.
    Returns empty dict if no locator files match — the caller must handle this.
    """
    catalog: dict[str, dict] = {}
    for f in _feature_locator_files(feature, requirement):
        try:
            tree = ast.parse(f.read_text(errors="replace"))
        except Exception:
            continue
        module = f.stem
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            sels: dict[str, str] = {}
            for stmt in node.body:
                if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 \
                        and isinstance(stmt.targets[0], ast.Name) \
                        and isinstance(stmt.value, ast.Constant) \
                        and isinstance(stmt.value.value, str):
                    sels[stmt.targets[0].id] = stmt.value.value
            if sels:
                catalog[node.name] = {"module": module, "selectors": sels}
    return catalog


def format_locator_catalog(catalog: dict[str, dict], max_per_class: int = 80) -> str:
    """Human-readable catalog for the planner prompt (NAME -> selector)."""
    if not catalog:
        return "(no locators found)"
    lines: list[str] = []
    for cls, info in catalog.items():
        lines.append(f"## {cls}")
        for i, (name, sel) in enumerate(info["selectors"].items()):
            if i >= max_per_class:
                lines.append("  ... (more)")
                break
            lines.append(f"  - {name}: {sel}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plan -> Playwright code (deterministic; output is always valid Python)
# ---------------------------------------------------------------------------

def _pystr(s) -> str:
    """Safe Python string literal (json.dumps yields a valid double-quoted str)."""
    return json.dumps(str(s))


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")
    return s[:50] or "case"


def _render_step(step: dict, name_to_class: dict[str, str]) -> list[str]:
    """Render ONE plan step to code line(s). Unknown action/locator -> a comment
    (never broken code)."""
    action = str(step.get("action", "")).strip()
    target = step.get("target", "")
    value = step.get("value", "")

    if action == "goto":
        return [f"    app_goto(page, {_pystr(target)})"]

    cls = name_to_class.get(str(target))
    if action not in {"open_section", "click", "fill", "press", "expect_visible",
                      "expect_not_visible", "expect_enabled", "expect_text", "expect_count_gt"}:
        return [f"    # skipped: unknown action {action!r}"]
    if not cls:
        return [f"    # skipped: unknown locator name {target!r}"]

    loc = f"page.locator({cls}.{target}).first"
    if action == "open_section":
        return [f"    open_section(page, {cls}.{target})"]
    if action == "click":
        return [f"    {loc}.click()"]
    if action == "fill":
        return [f"    {loc}.fill({_pystr(value)})"]
    if action == "press":
        return [f"    {loc}.press({_pystr(value or 'Enter')})"]
    if action == "expect_visible":
        return [f"    expect({loc}).to_be_visible()"]
    if action == "expect_not_visible":
        return [f"    expect({loc}).to_be_hidden()"]
    if action == "expect_enabled":
        return [f"    expect({loc}).to_be_enabled()"]
    if action == "expect_text":
        return [f"    expect({loc}).to_contain_text(re.compile(re.escape({_pystr(value)}), re.I))"]
    if action == "expect_count_gt":
        try:
            n = int(value)
        except Exception:
            n = 0
        return [f"    assert page.locator({cls}.{target}).count() > {n}, "
                f"{_pystr('expected more than ' + str(n) + ' of ' + str(target))}"]
    return [f"    # skipped: unhandled action {action!r}"]


def render_test_file(plan: dict, catalog: dict[str, dict]) -> str:
    """Render the full Playwright test file from a plan + locator catalog."""
    # NAME -> ClassName (first class that defines it wins)
    name_to_class: dict[str, str] = {}
    class_to_module: dict[str, str] = {}
    for cls, info in catalog.items():
        class_to_module[cls] = info["module"]
        for name in info["selectors"]:
            name_to_class.setdefault(name, cls)

    tests = plan.get("tests") or []
    used_classes: set[str] = set()
    bodies: list[str] = []
    used_names: set[str] = set()

    for i, t in enumerate(tests, 1):
        tid = _slug(t.get("id") or f"tc_{i:03d}")
        title = str(t.get("title") or tid)
        fn = f"def test_{i:03d}_{_slug(t.get('title') or tid)}(authenticated_page):"
        lines = [fn, f"    '''{title.replace(chr(39), '')}'''", "    page = authenticated_page"]
        steps = t.get("steps") or []
        rendered_any = False
        for step in steps:
            for ln in _render_step(step, name_to_class):
                lines.append(ln)
                if not ln.strip().startswith("#"):
                    rendered_any = True
            tgt = step.get("target")
            if tgt in name_to_class:
                used_classes.add(name_to_class[tgt])
                used_names.add(tgt)
        if not rendered_any:
            lines.append("    pass  # no renderable steps")
        bodies.append("\n".join(lines))

    imports = [
        "import re",
        "import pytest",
        "from playwright.sync_api import expect",
        "from _support import app_goto, open_section, dismiss_frill, BASE_URL",
    ]
    for cls in sorted(used_classes):
        imports.append(f"from locators.{class_to_module[cls]} import {cls}")

    header = (
        '"""Auto-generated Playwright tests (deterministically rendered from the test plan).\n'
        "Edits here are overwritten on each run.\n"
        '"""\n'
    )
    return header + "\n".join(imports) + "\n\n\n" + "\n\n\n".join(bodies) + "\n"


# ---------------------------------------------------------------------------
# Disk writers
# ---------------------------------------------------------------------------

def write_locator_package(scripts_dir: Path, feature: str, requirement: str) -> list[str]:
    """Write the real locator files into an importable ``locators/`` package."""
    written: list[str] = []
    loc_dir = scripts_dir / "locators"
    loc_dir.mkdir(parents=True, exist_ok=True)
    (loc_dir / "__init__.py").write_text("")
    written.append(str(loc_dir / "__init__.py"))
    for f in _feature_locator_files(feature, requirement):
        try:
            (loc_dir / f.name).write_text(f.read_text(errors="replace"))
            written.append(str(loc_dir / f.name))
        except Exception:
            continue
    return written


def write_harness(scripts_dir: Path) -> list[str]:
    """Copy the fixed, hand-written harness (conftest + support) into the suite."""
    written: list[str] = []
    scripts_dir.mkdir(parents=True, exist_ok=True)
    for name in ("_support.py", "conftest.py"):
        src = _HARNESS_DIR / name
        if src.exists():
            shutil.copy2(src, scripts_dir / name)
            written.append(str(scripts_dir / name))
    return written
