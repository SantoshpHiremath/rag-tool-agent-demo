"""
agent.py
--------
The agentic layer: an LLM-driven agent that, given a user question, decides
for itself which of three paths to take:

  1. search_notes    -> look something up via RAG (retrieval-augmented generation)
  2. calculator      -> perform an arithmetic computation
  3. answer directly -> respond from general knowledge, no tool needed

This is the piece that distinguishes an "agent" from a plain RAG pipeline:
a RAG pipeline always retrieves, then answers. An agent first REASONS about
what kind of question it's been asked, and chooses an action accordingly,
which is the core idea behind Agentic AI as used in tools like Microsoft
Copilot Studio's "topics and actions" model.

Note on LangChain versions: this targets LangChain 1.x. The agent is built
with `langchain.agents.create_agent`, the current tool-calling agent
constructor (the older `create_tool_calling_agent` + `AgentExecutor`
combo, and later the `langgraph.prebuilt.create_react_agent` alias, have
both been superseded by this). It's simple: give it an LLM, a list of
tools, and a system prompt, and it returns a ready-to-run graph you can
`.invoke()`.

This version runs the LLM fully locally via Ollama (https://ollama.com) -
no API key, no cloud calls, no cost. Pull the model first with
`ollama pull llama3.2` (and make sure the Ollama app/service is running).

Run this file directly for an interactive demo, or import `run_agent(question)`
to use it programmatically.
"""

from langchain_ollama import ChatOllama
from langchain.agents import create_agent

from rag_tool import search_notes
from calculator_tool import calculator

CHAT_MODEL = "llama3.2"

SYSTEM_PROMPT = """You are a helpful assistant with access to two tools:

1. search_notes: use this for factual questions about the FordA dataset
   (its structure, purpose, preprocessing steps, or known challenges).
2. calculator: use this for any question that requires an arithmetic
   computation.

If a question needs neither tool (e.g. it's a general knowledge question
unrelated to the dataset, or simple conversation), answer directly without
calling a tool. Always explain briefly which tool you used and why, or
say that you answered directly."""

TOOLS = [search_notes, calculator]


def build_agent():
    llm = ChatOllama(model=CHAT_MODEL, temperature=0)
    return create_agent(llm, TOOLS, system_prompt=SYSTEM_PROMPT)


def run_agent(question: str) -> str:
    agent = build_agent()
    result = agent.invoke({"messages": [("human", question)]})
    # The final answer is the content of the last message in the conversation.
    final_message = result["messages"][-1]
    return final_message.content


DEMO_QUESTIONS = [
    # Should route to search_notes (RAG) -- factual lookup on the source doc
    "What is the FordA dataset used for, and who created it?",
    # Should route to search_notes (RAG) -- preprocessing details
    "What preprocessing steps are commonly applied to the FordA signals before feature engineering?",
    # Should route to calculator -- pure arithmetic
    "The training set has 3601 instances and the test set has 1320. Using the calculator tool, "
    "compute 1320 / (3601 + 1320) to find the fraction of the total data used for testing.",
    # Should answer directly -- general knowledge, not in the notes, not a calculation
    "In one sentence, what is the difference between supervised and unsupervised learning?",
]


if __name__ == "__main__":
    print("=" * 70)
    print("RAG + Tool-Routing Agent Demo")
    print("=" * 70)
    for q in DEMO_QUESTIONS:
        print(f"\nQUESTION: {q}\n")
        answer = run_agent(q)
        print(f"\nFINAL ANSWER:\n{answer}")
        print("-" * 70)
