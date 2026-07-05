"""
calculator_tool.py
-------------------
A small, safe arithmetic tool for the agent. Demonstrates the "does a
calculation" branch of the agent's decision: not every question should be
routed to the RAG retriever — some need a real computation instead.

Uses Python's ast module to safely evaluate arithmetic expressions rather
than calling eval() directly on arbitrary input.
"""

import ast
import operator
from langchain_core.tools import tool

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPERATORS:
            raise ValueError(f"Operator {op_type} not allowed")
        return _ALLOWED_OPERATORS[op_type](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPERATORS:
            raise ValueError(f"Operator {op_type} not allowed")
        return _ALLOWED_OPERATORS[op_type](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression node: {node}")


def safe_calculate(expression: str) -> float:
    """Safely evaluate a basic arithmetic expression (+, -, *, /, **) without using eval()."""
    tree = ast.parse(expression, mode="eval")
    return _safe_eval(tree.body)


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '3601 / (3601 + 1320)' or '500 * 2'.
    Use this tool whenever the question requires a numeric computation rather than a
    factual lookup."""
    try:
        result = safe_calculate(expression)
        return f"{expression} = {result}"
    except Exception as exc:
        return f"Could not evaluate '{expression}': {exc}"
