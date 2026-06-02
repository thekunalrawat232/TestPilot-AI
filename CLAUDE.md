# CLAUDE.md — TestPilot AI (AI Test Agent)

Context file for Claude Code. Read this first; it captures the architecture so you don't have to re-read the whole tree each session.

## What this is

An end-to-end AI QA pipeline. Input: a plain-English feature requirement. Output: generated Playwright + Selenium pytest suites, executed against a live app, with failures auto-debugged and real bugs filed as Trello cards.

Built on **LangGraph** (orchestration) + **ChromaDB RAG** + multi-provider LLM (OpenAI / Anthropic / Google / Groq). Whole run is **2–3 LLM calls**; repeated runs hit the disk cache and cost 0 calls.

## Run it

```bash
python3 main.py "As a user I can log in with email and password"   # run pipeline
python3 main.py --resume latest          # resume crashed run from last checkpoint
python3 main.py --resume <run_id>
python3 main.py --list-checkpoints
python3 main.py --retries 3 "..."        # override retry limit
python3 main.py --rebuild-rag "..."      # rebuild RAG index from context/ files
streamlit run dashboard.py               # live web dashboard (polls live_status.json)
```

Config is all env-driven via `.env` (see `.env.example`). Key vars: `LLM_PROVIDER`, `LLM_MODEL`, `TARGET_BASE_URL`, `HEADLESS_BROWSER`, `MAX_RETRY_ATTEMPTS`, `TRELLO_*`.

## Pipeline (LangGraph) — `agents/graph.py`

Linear with one retry loop. `NODE_ORDER = [requirement_and_design, code_generator, execution, debug, finalise]`.

```
requirement_and_design → code_generator → execution → should_retry?
                                              ↑              ├─ "debug" → debug → (back to execution)
                                              └──────────────┘
                                                          └─ "done" → finalise → END
```

| Node | File | LLM | Does |
|------|------|-----|------|
| `requirement_and_design` | `agents/requirement_and_design.py` | 1 | RAG-retrieves context, then one call produces BOTH `requirement_analysis` + `test_plan`. (Merged from two old nodes to save a call.) |
| `code_generator` | `agents/code_generator.py` | 1 | Turns `test_plan` into Playwright/Selenium files + page objects + conftest, writes to `generated/automation_scripts/`. Receives `retrieved_context` (RAG locators/page objects) and is instructed to reuse real selectors. |
| `execution` | `agents/execution_debug.py` | 0 | Runs each `test_*.py` via `pytest --timeout=60` subprocess; marks files failed if output contains FAILED/ERROR/TIMEOUT. |
| `debug` | `agents/execution_debug.py` | 1 (per retry) | Sends failing source + output to LLM; classifies test-bug vs real-bug, writes `fixed_files` back to disk, increments `retry_count`. |
| `finalise` | `agents/graph.py` | 0 | Writes `report_<ts>.md`, pushes real bugs to Trello. |

`should_retry()` routes to `debug` only if failures exist AND `retry_count < max_retries`; else `done`.

## Key mechanics

- **State** (`agents/state.py`): `PipelineState` Pydantic model passed through all nodes. Dict fields use reducers (shallow-merge / list-append) so nodes emit partial updates without clobbering. Main fields: `requirement_analysis`, `test_plan`, `generated_code`, `execution_result`, `debug_analysis`, `retry_count`, `max_retries`, `pipeline_status`, `error_log`.
- **Checkpointing** (`agents/checkpoint.py`): every node is wrapped by `_make_checkpointing_node` in graph.py — writes `live_status.json` (start/done) AND saves a full-state checkpoint to `generated/reports/checkpoints/<run_id>.json` after each node. Resume reconstructs `entry_point` as the node AFTER the last completed one.
- **LLM cache** (`agents/llm_cache.py`): `cached_llm_invoke()` — MD5 of (node_name + message contents) → `.llm_cache/<hash>.json`. Hit = no API call. Delete `.llm_cache/` to force fresh calls.
- **LLM factory** (`config/settings.py` `get_llm()`): picks provider from env. Wraps Google + Groq clients with custom retry/backoff for 429/503/rate-limit, and enforces Google 5-RPM pacing (`_GOOGLE_MIN_INTERVAL`). `MODEL_CONTEXT_LIMITS` + `get_context_limit()` drive context trimming.
- **Context trimming** (`agents/utils.py`): `trim_context_to_fit()` uses tiktoken to keep RAG context under the model's window (system prompt + user parts are never trimmed). `extract_json()` robustly parses LLM JSON (strips fences, fixes `"""` docstrings and `"x".repeat(N)`).

## RAG — `rag/`

- `vectorstore.py`: scans `context/{codebase,api_schemas,docs,existing_tests}/` **plus any folders in `EXTERNAL_CONTEXT_DIRS`** (read-only ingestion of an external repo's locators/page objects, tagged `source_type=external_project`). Chunks (1500/200, code-aware separators), embeds, persists to ChromaDB at `rag_store/`. Loads existing store unless `force_rebuild`. `_collect_files` skips noise dirs (`_SKIP_DIRS`: node_modules, caches, reports, screenshots, …) and files ≥500KB. Run `--rebuild-rag` after changing context/external dirs.
- `embeddings.py`: local HuggingFace `all-MiniLM-L6-v2` (free, no API).
- `retriever.py`: `ProjectRetriever.query_formatted()` is what nodes call. Supports `source_type` filter.

## Layout

```
main.py              CLI entry point
dashboard.py         Streamlit live dashboard (spawns main.py as subprocess)
config/settings.py   all config dataclasses, LLM factory, rate limiting, token limits
agents/              graph.py, state.py, the 3 node modules, checkpoint.py, llm_cache.py, utils.py
prompts/             system prompts (one per node module)
rag/                 vectorstore, retriever, embeddings
integrations/trello.py   board/list lookup + card creation, dedups by card name, severity→label color
page_objects/base_page.py   PlaywrightBasePage + SeleniumBasePage — copied into generated/ each run
context/             RAG knowledge base inputs (add files here to improve generation)
generated/
  automation_scripts/    test_<feat>_pw.py, _se.py, conftest.py, page_objects/
  reports/               report_<ts>.md, state_<ts>.json, live_status.json, checkpoints/
.llm_cache/          disk LLM cache
rag_store/           ChromaDB persistence
tools/               UNUSED legacy helpers (browser/terminal/filesystem/test_parser) — not imported by the pipeline
```

## Gotchas / notes

- `tools/` is dead code — nothing outside `tools/` imports it. Don't assume it's wired in.
- `code_generator` deletes stale `test_*.py` before writing, and always copies the project's `page_objects/base_page.py` over any LLM-generated one (so base classes stay canonical). This fixes the common `No module named 'page_objects.base_page'` error.
- `debug` routes fixed files ending in `_page.py` into `page_objects/`, everything else into the scripts dir root.
- Tests run with `cwd=generated/automation_scripts` and `PYTHONPATH` = scripts_dir + project_root; env passes `BASE_URL` and `HEADLESS`.
- Resume has a legacy node-name map (`requirement_analyst`/`test_designer` → `requirement_and_design`) for old checkpoints.
- `.env.example` is stale on provider docs (says openai/anthropic only); the code supports `google` and `groq` too — README is the accurate reference.
- `build_vectorstore(force_rebuild=True)` wipes the existing Chroma collection before rebuilding. (Before the fix it appended, duplicating every chunk and leaving stale data on each `--rebuild-rag`.)
- `context/` sample files (login_page.html, test_auth_smoke.py, auth_api.yaml, testing_standards.md) were removed — they seeded the RAG with fake auth locators. Real locators come from `EXTERNAL_CONTEXT_DIRS` (Easyshul framework). Dirs kept but empty.
- No test suite / linter config in the repo. Verify changes by running `main.py` against a target URL, or inspect generated files.
```
