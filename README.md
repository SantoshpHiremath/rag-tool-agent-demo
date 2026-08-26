# RAG + Tool-Routing Agent Demo

A small agentic AI project: an LLM-driven agent that decides, per question,
whether to **retrieve** an answer from a document (RAG), **calculate** it,
or **answer directly** — rather than always following a single fixed pipeline.

Built to demonstrate hands-on experience with Retrieval-Augmented Generation
(RAG) and Agentic AI concepts, using a real, working document as the
knowledge source: reference notes on the FordA time-series dataset (the
same dataset used in my Smart Motor project).

Runs **fully locally via [Ollama](https://ollama.com)** — no API key, no
cloud calls, no cost.

## A note on this file's history (read this first)

This project's source files (`agent.py`, `rag_tool.py`,
`calculator_tool.py`, `data/forda_dataset_notes.md`) went missing at some
point -- only the tests and this README survived. Rather than leave a
gap or quietly re-describe the project without disclosing that, the
source was rebuilt from scratch, driven directly by the existing test
suite: `tests/test_calculator_tool.py` and `tests/test_agent_routing.py`
were not modified, and the rebuilt `calculator_tool.py`, `rag_tool.py`,
and `agent.py` were written to satisfy them as-is. All **29 tests pass
live against the rebuilt code** (confirmed in a clean virtual
environment: `python -m venv venv && pip install -r requirements.txt -r
requirements-dev.txt && pytest tests/ -v` → 29 passed).

What that does and doesn't prove: the 29 tests cover safe-arithmetic
correctness/security (21 tests, fully deterministic, no LLM involved)
and the agent's tool-registration and orchestration wiring against a
**mocked** LLM response (8 tests) -- i.e., "if the model decides to call
tool X, does the surrounding code correctly invoke it and return its
result." None of the 29 tests, old or new, call a live Ollama model --
that was true of the original test suite design too (see "Testing"
below, unchanged from before: it deliberately keeps CI fast and
non-flaky by mocking the LLM call). Whether `agent.py` actually produces
good tool-routing *decisions* when pointed at a real local Ollama model
is a live-execution result that needs to be run and recorded locally
(see the "real run against local Ollama" note in the Testing section,
which is currently an open item, not yet backed by a captured run)
rather than something the automated test suite itself checks.

## Why this project exists

Most "I know RAG" claims stop at a single retrieval-then-answer chain. This
project goes one step further: the agent is given **two tools** (a document
search tool and a calculator) and has to **reason about which one a given
question actually needs** — the same tool-choosing behavior behind agent
platforms like Microsoft Copilot Studio's "topics and actions," just built
from first principles with LangChain instead of a low-code UI.

## Architecture

```
                    ┌─────────────────────┐
   user question -> │   Agent (LLM)        │
                    │   decides: which tool?│
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                 │
        search_notes      calculator       (answers directly,
        (RAG over a       (safe arithmetic  no tool needed)
        local .md file    via Python's ast
        + FAISS vector    module, not eval())
        index, embedded
        locally via
        Ollama)
```

- **`rag_tool.py`** — loads `data/forda_dataset_notes.md`, splits it into
  chunks, embeds the chunks locally with Ollama's `nomic-embed-text` model,
  stores them in a local FAISS vector index, and answers questions by
  retrieving the most relevant chunks and composing them with the LLM using
  LCEL (LangChain Expression Language) — the retriever, prompt, and model
  are chained together explicitly rather than hidden behind a one-line
  helper class.
- **`calculator_tool.py`** — evaluates arithmetic expressions safely using
  Python's `ast` module (never `eval()` on raw input).
- **`agent.py`** — the orchestration layer, built with LangChain's
  `create_agent`. Defines the agent's system prompt and gives it both
  tools; the LLM (`llama3.2`, running locally via Ollama) decides
  per-question which tool (if any) to call.
- **`test_offline.py`** — sanity checks that run without Ollama needing to
  be active, to confirm the project structure and safe-calculator logic
  are correct before running the live demo.

## Setup

**1. Install Ollama** (one-time): download from https://ollama.com/download
and install it. It runs as a background app/service.

**2. Pull the two models this project uses:**

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

**3. Set up the Python environment:**

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

No API key and no `.env` file are needed — everything runs locally.

## Run it

```bash
# 1. Confirm the project structure and safe-calculator logic first
python test_offline.py

# 2. Make sure the Ollama app is running, then run the live demo
python agent.py
```

`agent.py` runs four example questions and prints which tool the agent
chose for each one, plus its final answer:

1. *"What is the FordA dataset used for, and who created it?"*
   → expected: routes to `search_notes` (RAG)
2. *"What preprocessing steps are commonly applied to the FordA signals?"*
   → expected: routes to `search_notes` (RAG)
3. *"Compute 1320 / (3601 + 1320) to find the test-set fraction."*
   → expected: routes to `calculator`
4. *"What's the difference between supervised and unsupervised learning?"*
   → expected: answered directly, no tool needed

## Testing

Beyond `test_offline.py`'s manual sanity checks, the project includes a
`pytest` test suite (`tests/`) covering the parts of the system that can be
tested deterministically, without requiring a live Ollama instance:

- **`tests/test_calculator_tool.py`** (21 tests) — unit tests for
  `safe_calculate()`, the AST-based arithmetic evaluator behind the
  calculator tool. Covers correctness across all supported operators,
  operator precedence and parentheses, and — since this function is
  invoked by an LLM agent on free-text input — a dedicated security test
  class confirming it rejects code-injection attempts (`__import__`,
  attribute access, function calls) rather than falling back to
  `eval()`-style execution. One test documents a real, verified edge case:
  the evaluator's `ast.Constant` check also matches string literals, so
  `safe_calculate("'a' + 'b'")` returns `"ab"` rather than raising — not a
  security issue (no code execution is possible), but worth knowing if a
  stricter numeric-only version is needed later.

- **`tests/test_agent_routing.py`** (8 tests) — verifies the agent's
  tool-routing wiring: that `calculator` and `search_notes` are registered
  under the names the system prompt refers to, and — using a mocked LLM
  response rather than a live model call — that the agent's orchestration
  logic correctly invokes the right tool and unwraps the final answer for
  calculator questions, dataset questions, and general-knowledge
  questions. This separates orchestration correctness (deterministic,
  unit-testable) from LLM reasoning quality, which needs a different kind
  of evaluation, such as an eval set run against a live model.

Run the suite:

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest
```

All 29 tests pass against the current implementation (confirmed live,
see the note at the top of this README).

### Real run against local Ollama

Unlike every other `RealLLMRunner`/`RealAgentRunner`-style class
elsewhere in this portfolio (`rag-tool-fastapi`, `rag-tool-api-docker`,
`underwriting-llm-risk-extraction`, `agentic-embedded-firmware-assistant`
-- all honestly disclosed as never executed against a live model, since
no live LLM was available in the sandbox those were built in), this
project's `agent.py` genuinely can run against a real, locally-running
Ollama instance (`llama3.2` + `nomic-embed-text`) on a machine where
Ollama is installed -- the cloud sandbox this README's rebuild happened
in has no route to `localhost:11434` or to Ollama's model registry, so
a live run has to happen on local hardware, not here.

**Status, stated precisely:** `python agent.py` has been run locally
against real Ollama before, but the printed output from that run (which
tool the model chose per question, and its final answers) was not
captured or saved at the time, so it isn't reproduced here. Rather than
reconstruct or approximate what that output probably looked like, this
section is left as an open item: the next real run's output should be
pasted here verbatim, unedited, the same way every other "real run"
artifact in this portfolio is sourced from an actual execution rather
than written from memory or expectation.

To produce it:
```bash
ollama pull llama3.2
ollama pull nomic-embed-text
python agent.py
```
and paste the full terminal output below, unedited.

<!-- REAL_OLLAMA_RUN_OUTPUT_PLACEHOLDER -->

## What this demonstrates

- Building a working RAG pipeline end-to-end (chunking, embedding, vector
  search, grounded generation) rather than just describing the concept.
- Agentic AI: an LLM that reasons about tool selection per-query, not a
  fixed single-path pipeline.
- Safe tool design — the calculator explicitly avoids `eval()` in favor of
  a restricted `ast`-based evaluator, backed by unit tests that verify it.
- Testable agent orchestration — routing logic is covered by mocked-LLM
  unit tests, separating deterministic wiring from LLM reasoning quality.
- Practical engineering habits: local-model setup, offline sanity tests,
  clear module boundaries, no secrets to manage.

## Relationship to sibling projects

This project is the source of truth for the actual RAG + tool-routing
agent logic. Two other projects wrap it rather than duplicate it:

- [`rag-tool-api-docker`](../rag-tool-api-docker/) — a Flask HTTP API
  wrapping this agent (`AGENT_MODE=real` imports and calls this
  project's `agent.run_agent()`), containerized with Docker (multi-stage
  build, non-root user, healthcheck). 16 tests, all passing, verified
  live over real HTTP.
- [`rag-tool-mcp-server`](../rag-tool-mcp-server/) — the same two
  underlying capabilities (retrieval, calculator) exposed as a real
  [Model Context Protocol](https://modelcontextprotocol.io) server using
  the official `mcp` Python SDK, so any MCP-compatible client can call
  them directly. 11 tests, all passing, independently verified over the
  real MCP protocol (handshake, tool discovery, tool invocation).

## Possible extensions

- Swap the local FAISS index for a hosted vector DB (e.g. Qdrant) to
  demonstrate production-scale retrieval.
- Swap Ollama for a hosted LLM API (OpenAI, Anthropic) to compare quality
  and demonstrate both local and cloud-based deployment.
- Add a third tool (e.g. a web search tool) to broaden the agent's reasoning.
- Wrap the agent in a small Streamlit or FastAPI front end.
