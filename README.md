# AI Test Agent

An end-to-end AI-powered QA pipeline that takes a plain-English feature requirement and autonomously generates test cases, writes Playwright and Selenium automation scripts, executes them against a live application, debugs failures, and files real bugs as Trello cards.

Built with **LangGraph**, **RAG (ChromaDB)**, and multi-provider LLM support.

---

## What It Does

```
"As a user I can log in with email and password"
             ↓
  Requirement & Test Design  (1 LLM call)
             ↓
  Code Generation            (Playwright + Selenium scripts written to disk)
             ↓
  Execution                  (pytest runs against your live app)
             ↓
  Debug Loop                 (AI fixes test bugs, retries up to N times)
             ↓
  Trello Cards               (real bugs → cards with severity + steps to reproduce)
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) |
| LLM providers | OpenAI / Anthropic / Google Gemini / Groq |
| RAG / Vector store | [ChromaDB](https://www.trychroma.com/) + HuggingFace embeddings |
| Browser automation | [Playwright](https://playwright.dev/python/) + [Selenium](https://selenium-python.readthedocs.io/) |
| Bug tracking | [Trello API](https://developer.atlassian.com/cloud/trello/) |
| LLM response caching | Disk-based MD5 cache (saves API quota) |

---

## Prerequisites

- Python 3.10+
- At least one LLM API key (OpenAI, Anthropic, Google, or Groq)
- A running web application to test against

---

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd ai_test_agent
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Playwright browsers

```bash
playwright install chromium
```

### 4. Configure environment

```bash
cp .env.example .env
```

Edit `.env` with your values:

```env
# Pick one LLM provider
LLM_PROVIDER=google                        # openai | anthropic | google | groq
LLM_MODEL=gemini-2.5-flash                 # see .env.example for model options
LLM_TEMPERATURE=0.1

# API keys — only the one you use is required
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...
GROQ_API_KEY=gsk_...

# Your application's base URL
TARGET_BASE_URL=https://your-app.com

# Execution settings
HEADLESS_BROWSER=true
MAX_RETRY_ATTEMPTS=1

# Trello (optional — skip if you don't need bug cards)
TRELLO_API_KEY=...
TRELLO_TOKEN=...
TRELLO_BOARD_NAME=AI Test Agent
TRELLO_LIST_NAME=To Do
```

### 5. (Optional) Add project context for RAG

Drop relevant files into the `context/` directories so the AI can generate more accurate tests:

```
context/
├── codebase/        ← relevant source code snippets
├── api_schemas/     ← API docs, OpenAPI specs
├── docs/            ← feature documentation
└── existing_tests/  ← example tests from your project
```

---

## Usage

### Run the pipeline on a requirement

```bash
python3 main.py "As a user I can log in with email and password"
```

```bash
python3 main.py "Explore the fundraising section and find bugs"
```

### Resume from a checkpoint (after a crash or rate limit)

```bash
python3 main.py --resume latest
python3 main.py --resume <run_id>
```

### List all saved checkpoints

```bash
python3 main.py --list-checkpoints
```

### Override retry limit

```bash
python3 main.py --retries 3 "As a user I can reset my password"
```

### Rebuild the RAG index (after updating context files)

```bash
python3 main.py --rebuild-rag "requirement text"
```

---

## How the Pipeline Works

### Nodes

| Node | What it does | LLM calls |
|------|-------------|-----------|
| **Requirement & Design** | Parses the requirement into test cases (positive, negative, edge cases, security) | 1 |
| **Code Generator** | Writes Playwright + Selenium Python test files and page objects to disk | 1 |
| **Execution** | Runs all generated tests with pytest, captures output | 0 |
| **Debug** (optional) | Analyses failures, classifies real bugs vs test bugs, rewrites broken tests | 1 |
| **Finalise** | Writes a Markdown report, creates Trello cards for real bugs | 0 |

**Total: 2–3 LLM calls per pipeline run.** Repeated runs of the same requirement use the disk cache and cost 0 API calls.

### Outputs

```
generated/
├── automation_scripts/
│   ├── page_objects/          ← generated page object classes
│   ├── test_<feature>_pw.py   ← Playwright test suite
│   ├── test_<feature>_se.py   ← Selenium test suite
│   └── conftest.py
└── reports/
    ├── report_<timestamp>.md  ← human-readable summary
    ├── state_<timestamp>.json ← full pipeline state
    └── checkpoints/           ← resume state per node
```

### Test coverage generated per requirement

- **Positive paths** — happy flow, valid inputs
- **Negative paths** — invalid credentials, wrong formats
- **Edge cases** — empty inputs, boundary values, special characters
- **Security** — XSS payloads, SQL injection strings
- **UI checks** — element visibility, navigation links

---

## Project Structure

```
ai_test_agent/
├── main.py                        # CLI entry point
├── requirements.txt
├── .env.example                   # configuration template
│
├── agents/                        # LangGraph pipeline nodes
│   ├── graph.py                   # pipeline orchestrator + checkpointing
│   ├── state.py                   # shared Pydantic state schema
│   ├── requirement_and_design.py  # Node 1: analyse requirement + design tests
│   ├── code_generator.py          # Node 2: generate automation code
│   ├── execution_debug.py         # Node 3+4: execute tests + debug failures
│   ├── checkpoint.py              # pause/resume support
│   └── llm_cache.py               # disk-based LLM response cache
│
├── prompts/                       # system prompts for each agent node
│   ├── requirement_and_design.py
│   ├── code_generator.py
│   └── execution_debug.py
│
├── rag/                           # Retrieval-Augmented Generation
│   ├── vectorstore.py             # ChromaDB ingestion + chunking
│   ├── retriever.py               # query interface
│   └── embeddings.py              # HuggingFace local embeddings
│
├── config/
│   └── settings.py                # all config, LLM factory, rate limiting
│
├── integrations/
│   └── trello.py                  # bug card creation
│
├── page_objects/
│   └── base_page.py               # Playwright + Selenium base classes
│
├── tools/                         # utility tools (filesystem, browser, terminal)
│
└── context/                       # RAG knowledge base (add your files here)
    ├── codebase/
    ├── api_schemas/
    ├── docs/
    └── existing_tests/
```

---

## API Quota Guide (Gemini 2.5 Flash free tier)

| Limit | Value | Usage per run |
|-------|-------|---------------|
| RPD (requests/day) | 20 | 2–3 (cached runs = 0) |
| RPM (requests/min) | 5 | 1–2 |
| TPM (tokens/min) | 250,000 | ~14,000 |

The built-in cache means re-running the same requirement costs **0 API calls**. At 2–3 calls per fresh run, the free tier supports **6–10 full pipeline runs per day**.

---

## Trello Integration

When the debug agent identifies a **real application bug** (not a test code issue), it automatically creates a Trello card with:

- Severity label (Critical / High / Medium / Low)
- Steps to reproduce
- Expected vs actual behaviour
- Evidence from test output

Duplicate cards are automatically skipped.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'page_objects.base_page'`**
Run a fresh pipeline — the code generator copies `base_page.py` automatically on each run. Or copy it manually:
```bash
cp page_objects/base_page.py generated/automation_scripts/page_objects/
```

**`503 UNAVAILABLE` from Gemini**
The model is temporarily overloaded. The pipeline retries automatically with backoff. Wait a minute and re-run, or use `--resume latest` to continue from the last checkpoint.

**`429 RESOURCE_EXHAUSTED` from Gemini**
Daily quota exceeded (20 RPD on free tier). Wait until quota resets (midnight Pacific), or use a paid API key.

**Groq rate limit**
Groq's free tier is 100k tokens/day. Switch to a different provider in `.env` if exhausted.

**Pipeline crashed mid-run**
```bash
python3 main.py --resume latest
```
Checkpoints are saved after every node — you won't lose progress.

---

## Contributing

1. Add project context files to `context/` for better test generation
2. Adjust `MAX_RETRY_ATTEMPTS` and `HEADLESS_BROWSER` in `.env` for your workflow
3. The LLM cache lives in `.llm_cache/` — delete it to force fresh LLM calls

---

## License

MIT
