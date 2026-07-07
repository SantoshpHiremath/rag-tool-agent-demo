"""
Unit tests for calculator_tool.py — the safe arithmetic evaluator used by the
agent's "calculator" tool.

These tests run offline with no LLM/Ollama dependency, since safe_calculate()
is pure Python (ast-based parsing). They cover:
  - correct arithmetic across all supported operators
  - operator precedence and parentheses
  - the security property that matters most for an agent-exposed tool:
    that it CANNOT execute arbitrary code, only arithmetic.
"""

import math

import pytest

from calculator_tool import safe_calculate, calculator


class TestSafeCalculateArithmetic:
    """Correctness of the four basic operators plus exponentiation."""

    @pytest.mark.parametrize(
        "expression, expected",
        [
            ("2 + 2", 4),
            ("10 - 3", 7),
            ("6 * 7", 42),
            ("9 / 2", 4.5),
            ("2 ** 10", 1024),
            ("-5 + 3", -2),
            ("3601 / (3601 + 1320)", 3601 / 4921),
        ],
    )
    def test_basic_operations(self, expression, expected):
        result = safe_calculate(expression)
        assert math.isclose(result, expected, rel_tol=1e-9)

    def test_operator_precedence_respected(self):
        # Multiplication should bind tighter than addition without parens.
        assert safe_calculate("2 + 3 * 4") == 14

    def test_parentheses_override_precedence(self):
        assert safe_calculate("(2 + 3) * 4") == 20

    def test_nested_parentheses(self):
        assert safe_calculate("((1 + 2) * (3 + 4)) / 7") == 3.0

    def test_negative_numbers(self):
        assert safe_calculate("-10 / 2") == -5.0

    def test_float_inputs(self):
        assert math.isclose(safe_calculate("1.5 * 2.5"), 3.75)


class TestSafeCalculateSecurity:
    """
    The calculator tool is invoked by an LLM agent based on free-text user
    input, so it must never fall back to eval()-style arbitrary execution.
    These tests assert that only whitelisted arithmetic AST nodes are
    permitted — anything else must raise, never execute.
    """

    def test_blocks_import_injection(self):
        with pytest.raises(Exception):
            safe_calculate("__import__('os').system('echo unsafe')")

    def test_blocks_function_calls(self):
        with pytest.raises(Exception):
            safe_calculate("print('hi')")

    def test_blocks_attribute_access(self):
        with pytest.raises(Exception):
            safe_calculate("().__class__.__bases__")

    def test_blocks_name_references(self):
        # Bare names (variables) are not valid arithmetic literals here.
        with pytest.raises(Exception):
            safe_calculate("os")

    def test_string_constants_are_not_rejected(self):
        """
        Known limitation (not a security hole): ast.Constant matches string
        literals too, so safe_calculate will happily evaluate "'a' + 'b'"
        via operator.add and return "ab". This can't execute code or access
        anything outside the expression, but it does mean the function is
        not strictly numeric-only. Documented here rather than silently
        assumed — a stricter version could add an isinstance(..., (int,
        float)) check inside _safe_eval's ast.Constant branch.
        """
        result = safe_calculate("'a' + 'b'")
        assert result == "ab"

    def test_invalid_syntax_raises(self):
        with pytest.raises(Exception):
            safe_calculate("2 + ")


def _invoke_calculator_tool(expression: str) -> str:
    """
    Call the @tool-decorated `calculator` the way LangChain actually calls
    it at runtime. langchain_core's @tool decorator wraps the function in a
    StructuredTool, so the underlying Python function is reached via
    `.func`, not by calling the tool object directly (that requires
    `.invoke({...})` with a full ToolCall payload). Falls back to calling
    `calculator` directly for environments where it hasn't been wrapped
    (e.g. a lightweight stub during offline testing).
    """
    if hasattr(calculator, "func"):
        return calculator.func(expression)
    return calculator(expression)


class TestCalculatorToolWrapper:
    """
    The @tool-decorated `calculator` function is what the agent actually
    calls. It must never raise — on failure it should return a readable
    error string, since an unhandled exception would break the agent loop.
    """

    def test_returns_formatted_success_string(self):
        result = _invoke_calculator_tool("2 + 2")
        assert "2 + 2" in result
        assert "4" in result

    def test_never_raises_on_bad_input(self):
        # Should return an error message, not propagate an exception.
        try:
            result = _invoke_calculator_tool("not an expression")
        except Exception as exc:
            pytest.fail(f"calculator tool raised instead of returning an error string: {exc}")
        assert "Could not evaluate" in result

    def test_never_raises_on_injection_attempt(self):
        try:
            result = _invoke_calculator_tool("__import__('os')")
        except Exception as exc:
            pytest.fail(f"calculator tool raised instead of returning an error string: {exc}")
        assert "Could not evaluate" in result
