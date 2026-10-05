# RAG + Tool-Routing Agent Demo

A small agentic AI project: an LLM-driven agent that decides, per question,
whether to **retrieve** an answer from a document (RAG), **calculate** it,
or **answer directly**, rather than following a single fixed pipeline.

It uses a real working document as its knowledge source: reference notes on
the FordA time-series dataset, the same dataset used in my Smart Motor
project. Everything runs **locally via [Ollama](https://ollama.com)**: no
API key, no cloud calls, no cost.

## What it does

The agent has two tools, a document search tool and a calculator, and
reasons about which one each question needs:

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

- **`rag_tool.py`** loads `data/forda_dataset_notes.md`, splits it into
  chunks, embeds them locally with Ollama's `nomic-embed-text` model,
  stores them in a local FAISS vector index, and answers questions by
  retrieving the most relevant chunks and composing them with the LLM using
  LCEL (LangChain Expression Language). The retriever, prompt, and model
  are chained together explicitly.
- **`calculator_tool.py`** evaluates arithmetic expressions safely with
  Python's `ast` module, never `eval()` on raw input.
- **`agent.py`** is the orchestration layer, built with LangChain's
  `create_agent`. It defines the system prompt and gives the agent both
  tools; the LLM (`llama3.2`, running locally via Ollama) decides
  per question which tool, if any, to call.
- **`test_offline.py`** runs sanity checks without Ollama needing to be
  active, confirming the project structure and calculator logic before the
  live demo.

## Setup

**1. Install Ollama** (one-time): download from https://ollama.com/download.
It runs as a background app/service.

**2. Pull the two models:**

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

No API key and no `.env` file are needed.

## Run it

```bash
# 1. Confirm the project structure and calculator logic
python test_offline.py

# 2. With the Ollama app running, start the live demo
python agent.py
```

`agent.py` runs four example questions and prints which tool the agent
chose for each one, plus its final answer:

1. *"What is the FordA dataset used for, and who created it?"*
   routes to `search_notes` (RAG)
2. *"What preprocessing steps are commonly applied to the FordA signals?"*
   routes to `search_notes` (RAG)
3. *"Compute 1320 / (3601 + 1320) to find the test-set fraction."*
   routes to `calculator`
4. *"What's the difference between supervised and unsupervised learning?"*
   answered directly, no tool needed

## Tests

The project has a `pytest` suite of **29 tests, all passing**:

- **`tests/test_calculator_tool.py`** (21 tests) covers `safe_calculate()`,
  the AST-based arithmetic evaluator: correctness across all supported
  operators, operator precedence and parentheses, and a dedicated security
  test class confirming that code-injection attempts (`__import__`,
  attribute access, function calls) are rejected. One test documents an
  edge case: the `ast.Constant` check also matches string literals, so
  `safe_calculate("'a' + 'b'")` returns `"ab"`. No code execution is
  possible, and a stricter numeric-only version is an easy extension.
- **`tests/test_agent_routing.py`** (8 tests) verifies the routing wiring:
  `calculator` and `search_notes` are registered under the names the
  system prompt refers to, and, with a mocked LLM response, the
  orchestration logic invokes the right tool and returns the final answer
  for calculator questions, dataset questions, and general-knowledge
  questions. Mocking the model keeps CI fast and deterministic.

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest
```

The routing quality of the live model is seen by running `python agent.py`
against local Ollama, which prints the tool chosen for each of the four
example questions.

## Highlights

- A working RAG pipeline end to end: chunking, embedding, vector search,
  and grounded generation.
- Agentic AI: an LLM that reasons about tool selection per query, in the
  spirit of the topics-and-actions pattern used by agent platforms, built
  from first principles with LangChain.
- Safe tool design: the calculator uses a restricted `ast`-based evaluator
  and is backed by unit tests.
- Testable agent orchestration: routing logic is covered by mocked-LLM unit
  tests.
- Clean engineering habits: local-model setup, offline sanity tests, clear
  module boundaries, and no secrets to manage.

## Related projects

This project holds the core RAG + tool-routing agent logic. Two other
projects build on it:

- [`rag-tool-api`](https://github.com/SantoshpHiremath/rag-tool-api): a Flask HTTP API around
  this agent (`AGENT_MODE=real` calls this project's `agent.run_agent()`),
  containerized with Docker (multi-stage build, non-root user,
  healthcheck). 16 tests.
- [`rag-tool-mcp-server`](../rag-tool-mcp-server/): the same two
  capabilities (retrieval, calculator) exposed as a
  [Model Context Protocol](https://modelcontextprotocol.io) server with the
  official `mcp` Python SDK, so any MCP-compatible client can call them.
  11 tests.

## Possible extensions

- Swap the local FAISS index for a hosted vector DB (e.g. Qdrant) for
  production-scale retrieval.
- Swap Ollama for a hosted LLM API (OpenAI, Anthropic) to compare quality
  across local and cloud deployments.
- Add a third tool, such as web search, to broaden the agent's reasoning.
- Wrap the agent in a small Streamlit or FastAPI front end.
