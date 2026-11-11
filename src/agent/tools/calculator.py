"""A safe arithmetic evaluator built on the `ast` module (no eval/exec)."""

import ast
import math
import operator

from langchain_core.tools import tool

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {"abs": abs, "round": round, "sqrt": math.sqrt, "min": min, "max": max}
_MAX_EXPONENT = 100
_MAX_LENGTH = 200


class UnsafeExpression(ValueError):
    """Raised when an expression contains anything other than plain arithmetic."""


def safe_eval(expression: str) -> float | int:
    if len(expression) > _MAX_LENGTH:
        raise UnsafeExpression("expression too long")
    tree = ast.parse(expression.replace("^", "**"), mode="eval")
    return _eval(tree.body)


def _eval(node: ast.AST) -> float | int:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
            raise UnsafeExpression("exponent too large")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval(node.operand))
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCS
        and not node.keywords
    ):
        return _FUNCS[node.func.id](*(_eval(arg) for arg in node.args))
    raise UnsafeExpression(f"unsupported syntax: {type(node).__name__}")


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression exactly, e.g. "(42 - 30) / 30 * 100".

    Supports + - * / // % ** (or ^), parentheses and abs, round, sqrt, min, max.
    Always use this instead of doing math in your head.
    """
    try:
        result = safe_eval(expression)
    except (UnsafeExpression, SyntaxError, ZeroDivisionError, TypeError, ValueError) as exc:
        return f"Error: {exc}"
    if isinstance(result, float):
        result = round(result, 6)
    return f"{expression} = {result}"
