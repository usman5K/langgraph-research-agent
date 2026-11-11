import pytest

from agent.tools.calculator import UnsafeExpression, calculator, safe_eval


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("2 + 3 * 4", 14),
        ("(42 - 30) / 30 * 100", 40.0),
        ("-2 ** 2", -4),
        ("2 ^ 10", 1024),
        ("sqrt(16) + abs(-1)", 5.0),
        ("max(1, 7, 3)", 7),
        ("7 // 2 + 7 % 2", 4),
    ],
)
def test_arithmetic(expr, expected):
    assert safe_eval(expr) == expected


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('echo hi')",
        "open('/etc/passwd')",
        "(1).__class__",
        "x + 1",
        "[1, 2]",
        "lambda: 1",
        "'a' * 3",
        "round(1.5, ndigits=1)",
    ],
)
def test_rejects_anything_but_arithmetic(expr):
    with pytest.raises(UnsafeExpression):
        safe_eval(expr)


def test_rejects_huge_exponents_and_long_input():
    with pytest.raises(UnsafeExpression):
        safe_eval("9 ** 999999")
    with pytest.raises(UnsafeExpression):
        safe_eval("1+" * 200 + "1")


def test_tool_returns_errors_as_text():
    assert calculator.invoke({"expression": "1/0"}).startswith("Error")
    assert calculator.invoke({"expression": "import os"}).startswith("Error")
    assert calculator.invoke({"expression": "10 / 4"}) == "10 / 4 = 2.5"
