"""Recompute finite numeric answers using a restricted arithmetic AST."""

from __future__ import annotations

import ast
import math
from typing import Any, List, Mapping, Union

Number = Union[int, float]
_BIN_OPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
}
_UNARY_OPS = {ast.UAdd: lambda a: +a, ast.USub: lambda a: -a}
_OPS = {"argmin", "argmax", "rank", "count_ge", "count_le", "count_gt", "count_lt"}


class ExpressionError(ValueError):
    """An expression is malformed, nonfinite or outside the arithmetic whitelist."""


def _number(value: Any, label: str) -> Number:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("%s must be a finite real number" % label)
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError("%s must be a finite real number" % label)
    return value


def _tree(expr: str) -> ast.Expression:
    if not isinstance(expr, str) or not expr.strip() or len(expr) > 10_000:
        raise ExpressionError("expression must be nonempty text of at most 10000 characters")
    try:
        tree = ast.parse(expr, mode="eval")
    except (SyntaxError, RecursionError) as exc:
        raise ExpressionError("could not parse expression %r: %s" % (expr, exc)) from exc
    if sum(1 for _node in ast.walk(tree)) > 1000:
        raise ExpressionError("expression exceeds 1000 AST nodes")
    return tree


def evaluate_expr(expr: str, fields: Mapping[str, Any]) -> Number:
    if not isinstance(fields, Mapping):
        raise ExpressionError("expression fields must be a mapping")
    try:
        return _eval_node(_tree(expr).body, fields, expr)
    except (ArithmeticError, RecursionError, ValueError) as exc:
        if isinstance(exc, ExpressionError):
            raise
        raise ExpressionError("arithmetic error in %r: %s" % (expr, exc)) from exc


def expr_fields(expr: str) -> set:
    return {node.id for node in ast.walk(_tree(expr)) if isinstance(node, ast.Name)}


def _eval_node(node: ast.AST, fields: Mapping[str, Any], expr: str) -> Number:
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        value = _BIN_OPS[type(node.op)](
            _eval_node(node.left, fields, expr), _eval_node(node.right, fields, expr)
        )
    elif isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        value = _UNARY_OPS[type(node.op)](_eval_node(node.operand, fields, expr))
    elif isinstance(node, ast.Constant):
        value = node.value
    elif isinstance(node, ast.Name):
        if node.id not in fields:
            raise ExpressionError("unknown field %r in %r" % (node.id, expr))
        value = fields[node.id]
    else:
        raise ExpressionError(
            "disallowed expression element %s in %r" % (type(node).__name__, expr)
        )
    return _number(value, "expression value")


def resolve_field(item: Mapping[str, Any], field: str, derived: Mapping[str, str]) -> Any:
    if field in item:
        return item[field]
    if field in derived:
        return evaluate_expr(derived[field], item)
    raise KeyError("field %r is neither a raw nor a derived field" % field)


def validate_compute_spec(spec: Mapping[str, Any], n_items: int) -> None:
    """Apply the same operation guards to public compute entry points."""
    if (
        not isinstance(spec, Mapping)
        or not isinstance(spec.get("op"), str)
        or spec["op"] not in _OPS
    ):
        raise ValueError("unknown or missing compute operation")
    if not isinstance(spec.get("field"), str) or not spec["field"].strip():
        raise ValueError("compute field must be nonempty text")
    op = spec["op"]
    if op == "rank":
        k = spec.get("k")
        if type(k) is not int or not 1 <= k <= n_items:
            raise ValueError("rank k must be an integer within the item range")
        if spec.get("order", "asc") not in ("asc", "desc"):
            raise ValueError("rank order must be asc/desc")
    if op.startswith("count_"):
        _number(spec.get("threshold"), "count threshold")


def _numeric(
    items: List[Mapping[str, Any]], field: str, derived: Mapping[str, str]
) -> List[Number]:
    return [_number(resolve_field(item, field, derived), "target value") for item in items]


def compute_answer_value(data: Mapping[str, Any], spec: Mapping[str, Any]) -> Any:
    """Compute from source; ties select the earliest source item."""
    if (
        not isinstance(data, Mapping)
        or not isinstance(data.get("items"), list)
        or not data["items"]
    ):
        raise ValueError("compute data requires a nonempty items list")
    items = data["items"]
    if any(
        not isinstance(item, Mapping)
        or not isinstance(item.get("id"), str)
        or not item["id"].strip()
        for item in items
    ):
        raise ValueError("every compute item requires a string id")
    if len({item["id"] for item in items}) != len(items):
        raise ValueError("compute item ids must be unique")
    derived = data.get("derived", {})
    if not isinstance(derived, Mapping):
        raise ValueError("derived fields must be a mapping")
    validate_compute_spec(spec, len(items))
    op = spec["op"]
    values = _numeric(items, spec["field"], derived)
    if op in ("argmin", "argmax"):
        chooser = min if op == "argmin" else max
        return items[chooser(range(len(items)), key=lambda i: values[i])]["id"]
    if op == "rank":
        indices = sorted(
            range(len(items)), key=lambda i: values[i], reverse=spec.get("order", "asc") == "desc"
        )
        return items[indices[spec["k"] - 1]]["id"]
    threshold = spec["threshold"]
    compare = {
        "count_ge": lambda value: value >= threshold,
        "count_le": lambda value: value <= threshold,
        "count_gt": lambda value: value > threshold,
        "count_lt": lambda value: value < threshold,
    }[op]
    return sum(1 for value in values if compare(value))


def recompute(data: Mapping[str, Any], question: Mapping[str, Any]) -> str:
    if not isinstance(question, Mapping) or not isinstance(question.get("choices"), list):
        raise ValueError("question requires a compute spec and choices list")
    value = compute_answer_value(data, question.get("compute"))
    matches = []
    for choice in question["choices"]:
        if not isinstance(choice, Mapping) or not isinstance(choice.get("id"), str):
            raise ValueError("question choices require string ids")
        if choice.get("value") == value:
            matches.append(choice["id"])
    if len(matches) != 1:
        raise ValueError(
            "computed answer %r for question %r must match exactly one choice"
            % (value, question.get("id"))
        )
    return matches[0]
