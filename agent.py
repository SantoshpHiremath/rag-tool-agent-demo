"""
agent.py
--------

The orchestration layer: builds a LangGraph ReAct-style agent (via
langgraph.prebuilt.create_react_agent, imported here as create_agent)
over the two tools this project defines (search_notes, calculator), and
lets the LLM (llama3.2, running locally via Ollama) decide per-question
which tool -- if any -- to call.

This is genuinely agentic in the sense that matters for this project:
the routing decision (retrieve vs. calculate vs. answer directly) is
made by the model reasoning over the question and the tools' names and
descriptions, not by a hand-rolled if/else dispatcher (that pattern is
used instead in the StubAgentRunner/StubLLMRunner classes elsewhere in
this portfolio, precisely to make routing testable without a live LLM --
see rag-tool-api-docker/agent_runner.py for that contrast).
"""
from __future__ import annotations

import os

from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent as create_agent

from calculator_tool import calculator
from rag_tool import search_notes

TOOLS = [calculator, search_notes]

SYSTEM_PROMPT = """You are a helpful assistant with access to two tools:

- search_notes: use this for any question about the FordA dataset --
  what it is, where it came from, its structure, or how it's typically
  used. Always ground your answer in what this tool returns.
- calculator: use this whenever the question requires computing a
  numeric result (arithmetic on numbers given in the question).

If a question needs neither tool -- e.g. a general knowledge question
unrelated to FordA or arithmetic -- answer it directly from your own
knowledge instead of forcing a tool call.

Be concise. State clearly which approach you used (search_notes,
calculator, or direct) at the start of your answer."""


def _build_agent_graph():
    model = ChatOllama(model=os.environ.get("OLLAMA_MODEL", "llama3.2"), temperature=0)
    return create_agent(model, TOOLS, prompt=SYSTEM_PROMPT)


def run_agent(question: str) -> str:
    """Runs the agent graph on a single question and returns the final
    answer text (not the raw LangGraph state dict)."""
    graph = _build_agent_graph()
    result = graph.invoke({"messages": [("user", question)]})
    return result["messages"][-1].content


if __name__ == "__main__":
    questions = [
        "What is the FordA dataset used for, and who created it?",
        "What preprocessing steps are commonly applied to the FordA signals?",
        "Compute 1320 / (3601 + 1320) to find the test-set fraction.",
        "What's the difference between supervised and unsupervised learning?",
    ]
    for q in questions:
        print("=" * 70)
        print("Q:", q)
        print("-" * 70)
        print(run_agent(q))
        print()
