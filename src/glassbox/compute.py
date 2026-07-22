"""The recompute engine — every answer key is re-derived from source, never trusted.

Two pieces:

* ``evaluate_expr`` — a *safe* arithmetic evaluator over a restricted AST (numbers,
  field names, and the operators ``+ - * / // % **`` only). It never calls ``eval``
  and never touches attributes, calls, subscripts, or comprehensions, so a scenario
  file cannot smuggle in executable code. This is what lets "derived" fields such as
  ``total_cost = monthly_payment * term_months + fees`` live in data while staying
  inert. Exponentiation is deliberately excluded so no expression can trigger
  unbounded integer growth (a ``9**9**9`` denial of service).

* ``recompute`` — given a scenario's data and one question, computes the canonical
  answer purely from the source and returns the id of the choice that represents it.
  The validator (and tests) compare that against the authored ``answer`` so content
  and answer key can never silently diverge.
"""

from __future__ import annotations

import ast
from typing import Any, Dict, List, Mapping, Union

Number = Union[int, float]

# --- safe arithmetic over a whitelisted AST ---------------------------------

# Exponentiation is intentionally omitted: `**` on constants can grow without bound
# (`9**9**9`) and hang the process. The shipped derived fields never need it.
_BIN_OPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
}
_UNARY_OPS = {
    ast.UAdd: lambda a: +a,
    ast.USub: lambda a: -a,
}


class ExpressionError(ValueError):
    """A derived-field expression is malformed or uses a disallowed construct."""


def evaluate_expr(expr: str, fields: Mapping[str, Any]) -> Number:
    """Evaluate ``expr`` against ``fields`` using only whitelisted arithmetic.

    Names resolve to values in ``fields`` (which must be numeric). Any construct
    outside the whitelist raises :class:`ExpressionError`.
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:  # malformed expression
        raise ExpressionError("could not parse expression %r: %s" % (expr, exc))
    try:
        return _eval_node(tree.body, fields, expr)
    except ArithmeticError as exc:  # e.g. division by zero on an in-whitelist operator
        raise ExpressionError("arithmetic error in %r: %s" % (expr, exc))


def expr_fields(expr: str) -> "set":
    """The set of field names an expression references (its inputs)."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError("could not parse expression %r: %s" % (expr, exc))
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}


def _eval_node(node: ast.AST, fields: Mapping[str, Any], expr: str) -> Number:
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _eval_node(node.left, fields, expr)
        right = _eval_node(node.right, fields, expr)
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand, fields, expr))
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ExpressionError("non-numeric constant %r in %r" % (node.value, expr))
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in fields:
            raise ExpressionError("unknown field %r in %r" % (node.id, expr))
        value = fields[node.id]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ExpressionError("field %r is not numeric (%r)" % (node.id, value))
        return value
    raise ExpressionError("disallowed expression element %s in %r"
                          % (type(node).__name__, expr))


# --- field resolution -------------------------------------------------------

def resolve_field(item: Mapping[str, Any], field: str,
                  derived: Mapping[str, str]) -> Any:
    """Return ``field`` for ``item`` — a raw field, or a derived one evaluated now."""
    if field in item:
        return item[field]
    if field in derived:
        return evaluate_expr(derived[field], item)
    raise KeyError("field %r is neither a raw nor a derived field" % field)


# --- compute operations -----------------------------------------------------

def _numeric(items: List[Mapping[str, Any]], field: str,
             derived: Mapping[str, str]) -> List[Number]:
    return [resolve_field(it, field, derived) for it in items]


def compute_answer_value(data: Mapping[str, Any], spec: Mapping[str, Any]) -> Any:
    """Compute the canonical answer *value* for a question's ``compute`` spec.

    Returns an item id (for argmin/argmax/rank) or an integer (for count_*).
    Ties break to the earliest item in source order — scenarios should avoid ties.
    """
    op = spec["op"]
    items: List[Mapping[str, Any]] = data["items"]
    derived: Mapping[str, str] = data.get("derived", {})

    if op in ("argmin", "argmax"):
        values = _numeric(items, spec["field"], derived)
        chooser = min if op == "argmin" else max
        best_idx = chooser(range(len(items)), key=lambda i: values[i])
        return items[best_idx]["id"]

    if op == "rank":
        # k is 1-indexed; order "asc" (default) ranks smallest first.
        k = int(spec["k"])
        order = spec.get("order", "asc")
        values = _numeric(items, spec["field"], derived)
        order_idx = sorted(range(len(items)),
                           key=lambda i: values[i],
                           reverse=(order == "desc"))
        if not 1 <= k <= len(items):
            raise ValueError("rank k=%d out of range for %d items" % (k, len(items)))
        return items[order_idx[k - 1]]["id"]

    if op in ("count_ge", "count_le", "count_gt", "count_lt"):
        threshold = spec["threshold"]
        values = _numeric(items, spec["field"], derived)
        cmp = {
            "count_ge": lambda v: v >= threshold,
            "count_le": lambda v: v <= threshold,
            "count_gt": lambda v: v > threshold,
            "count_lt": lambda v: v < threshold,
        }[op]
        return sum(1 for v in values if cmp(v))

    raise ValueError("unknown compute op %r" % op)


def recompute(data: Mapping[str, Any], question: Mapping[str, Any]) -> str:
    """Recompute a question's correct choice id from source data alone.

    Maps the computed answer value to the choice whose ``value`` equals it.
    Raises if no choice matches — an unrepresentable answer is a scenario bug.
    """
    value = compute_answer_value(data, question["compute"])
    for choice in question["choices"]:
        if choice.get("value") == value:
            return choice["id"]
    raise ValueError(
        "computed answer %r for question %r matches no choice value"
        % (value, question.get("id"))
    )
