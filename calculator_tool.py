"""
calculator_tool.py
-------------------

A safe arithmetic tool for the agent. Deliberately does NOT use eval() on
raw text -- the input comes from an LLM parsing free-text user questions,
so it must be restricted to a real arithmetic grammar (+, -, *, /, **,
parentheses, numeric literals, unary minus) via Python's ast module, not
arbitrary code execution.
"""
from __future__ import annotations

import ast
import operator

from langchain_core.tools import tool

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARYOPS = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node: ast.AST):
    if isinstance(node, ast.Constant):
        # Deliberately permissive here (see tests/test_calculator_tool.py's
        # test_string_constants_are_not_rejected): ast.Constant also
        # matches string literals, so "'a' + 'b'" evaluates via
        # operator.add to "ab" rather than raising. This is not a security
        # hole -- no code execution is possible either way -- but it does
        # mean this function is not strictly numeric-only. Documented
        # rather than silently assumed; a stricter numeric-only version
        # would add an isinstance(node.value, (int, float)) check here.
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"Unsupported expression element: {ast.dump(node)}")


def safe_calculate(expression: str):
    """Parse and evaluate a restricted arithmetic expression. Raises on
    anything outside the whitelisted grammar -- imports, function calls,
    attribute access, and bare names are all rejected, never executed."""
    tree = ast.parse(expression, mode="eval")
    return _eval_node(tree.body)


@tool
def calculator(expression: str) -> str:
    """Evaluate a numeric arithmetic expression (supports +, -, *, /, **,
    parentheses, and unary minus). Use this whenever the user's question
    requires computing a numeric result rather than looking up information."""
    try:
        result = safe_calculate(expression)
    except Exception as exc:  # noqa: BLE001 - deliberately broad, surfaced to caller
        return f"Could not evaluate '{expression}': {exc}"
    return f"{expression} = {result}"
