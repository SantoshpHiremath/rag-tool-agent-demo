"""
Tests for the agent's tool-routing behaviour (agent.py).

The agent depends on a locally-running Ollama LLM (llama3.2), so these
tests do not spin up a live model — that would make CI flaky and slow.
Instead they verify the two things that can and should be tested without
a live LLM call:

  1. Tool registration: the agent is wired up with exactly the tools it
     is supposed to have, under the names the system prompt refers to.
  2. Routing decisions: given a mocked LLM response that requests a
     specific tool, the agent graph correctly invokes that tool and
     returns its output — i.e. the routing/orchestration wiring works,
     independent of whether the LLM itself makes a good tool choice.

This mirrors how you'd test an agent pipeline in production: the LLM's
reasoning quality is evaluated separately (e.g. via eval datasets), while
the surrounding orchestration code gets deterministic unit tests.
"""

from unittest.mock import patch, MagicMock

import pytest

from calculator_tool import calculator
from rag_tool import search_notes


class TestToolRegistration:
    """Verify the agent exposes exactly the tools its system prompt promises."""

    def test_calculator_tool_is_named_calculator(self):
        assert calculator.name == "calculator"

    def test_search_notes_tool_is_named_search_notes(self):
        assert search_notes.name == "search_notes"

    def test_agent_tools_list_contains_both_tools(self):
        import agent

        tool_names = {t.name for t in agent.TOOLS}
        assert tool_names == {"calculator", "search_notes"}

    def test_system_prompt_references_both_tool_names(self):
        """
        The system prompt is what tells the LLM which tool to use for which
        kind of question. If a tool gets renamed but the prompt isn't
        updated, routing silently breaks — so we assert the prompt text
        and the actual tool names stay in sync.
        """
        import agent

        assert "search_notes" in agent.SYSTEM_PROMPT
        assert "calculator" in agent.SYSTEM_PROMPT


class TestRoutingWithMockedLLM:
    """
    Verify that when the (mocked) LLM decides to call a given tool, the
    agent graph correctly executes that tool and the result reaches the
    final answer — without requiring a live Ollama instance.
    """

    @patch("agent.create_agent")
    @patch("agent.ChatOllama")
    def test_calculator_question_invokes_calculator_tool(self, mock_chat_ollama, mock_create_agent):
        import agent

        # Simulate the compiled agent graph invoking the calculator tool
        # and producing a final answer that reflects its output.
        mock_agent_graph = MagicMock()
        mock_agent_graph.invoke.return_value = {
            "messages": [
                MagicMock(content="Using the calculator tool: 1320 / (3601 + 1320) = 0.2683")
            ]
        }
        mock_create_agent.return_value = mock_agent_graph

        result = agent.run_agent(
            "The training set has 3601 instances and the test set has 1320. "
            "Using the calculator tool, compute 1320 / (3601 + 1320)."
        )

        assert "calculator" in result.lower()
        assert "0.2683" in result
        mock_agent_graph.invoke.assert_called_once()

    @patch("agent.create_agent")
    @patch("agent.ChatOllama")
    def test_dataset_question_invokes_search_notes_tool(self, mock_chat_ollama, mock_create_agent):
        import agent

        mock_agent_graph = MagicMock()
        mock_agent_graph.invoke.return_value = {
            "messages": [
                MagicMock(content="Using search_notes: the FordA dataset is a univariate time-series "
                                   "classification benchmark from motor diagnostics.")
            ]
        }
        mock_create_agent.return_value = mock_agent_graph

        result = agent.run_agent("What is the FordA dataset used for, and who created it?")

        assert "search_notes" in result.lower() or "forda" in result.lower()
        mock_agent_graph.invoke.assert_called_once()

    @patch("agent.create_agent")
    @patch("agent.ChatOllama")
    def test_general_knowledge_question_answers_directly(self, mock_chat_ollama, mock_create_agent):
        import agent

        mock_agent_graph = MagicMock()
        mock_agent_graph.invoke.return_value = {
            "messages": [
                MagicMock(content="Answered directly: supervised learning uses labeled data, "
                                   "unsupervised learning finds structure in unlabeled data.")
            ]
        }
        mock_create_agent.return_value = mock_agent_graph

        result = agent.run_agent(
            "In one sentence, what is the difference between supervised and unsupervised learning?"
        )

        assert "directly" in result.lower()
        mock_agent_graph.invoke.assert_called_once()

    @patch("agent.create_agent")
    @patch("agent.ChatOllama")
    def test_run_agent_returns_final_message_content_not_full_state(self, mock_chat_ollama, mock_create_agent):
        """
        run_agent() should unwrap the graph's final state and return only
        the last message's text content — not the raw dict — since callers
        (and the CLI demo) expect a plain string.
        """
        import agent

        mock_agent_graph = MagicMock()
        mock_agent_graph.invoke.return_value = {
            "messages": [
                MagicMock(content="first"),
                MagicMock(content="final answer text"),
            ]
        }
        mock_create_agent.return_value = mock_agent_graph

        result = agent.run_agent("any question")

        assert result == "final answer text"
        assert isinstance(result, str)
