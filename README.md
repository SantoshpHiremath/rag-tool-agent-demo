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

## What this demonstrates

- Building a working RAG pipeline end-to-end (chunking, embedding, vector
  search, grounded generation) rather than just describing the concept.
- Agentic AI: an LLM that reasons about tool selection per-query, not a
  fixed single-path pipeline.
- Safe tool design — the calculator explicitly avoids `eval()` in favor of
  a restricted `ast`-based evaluator.
- Practical engineering habits: local-model setup, offline sanity tests,
  clear module boundaries, no secrets to manage.

## Possible extensions

- Swap the local FAISS index for a hosted vector DB (e.g. Qdrant) to
  demonstrate production-scale retrieval.
- Swap Ollama for a hosted LLM API (OpenAI, Anthropic) to compare quality
  and demonstrate both local and cloud-based deployment.
- Add a third tool (e.g. a web search tool) to broaden the agent's reasoning.
- Wrap the agent in a small Streamlit or FastAPI front end.
