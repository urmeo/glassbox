from __future__ import annotations

import ast
import copy
import hashlib
import html
import json
import math
import operator
import os
import re
import signal
import struct
import subprocess
import tempfile
import time
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Tuple, Union

import yaml


__version__ = "0.1.1"


# Resources

_KINDS = ("scenarios", "studies", "anchors")


class ResourceError(ValueError):
    pass


def _directory(kind: str):
    if not isinstance(kind, str) or kind not in _KINDS:
        raise ResourceError("unknown resource kind %r" % (kind,))
    return Path(__file__).resolve().parent.joinpath("_data").joinpath(kind)


def _filename(kind: str, name: str) -> str:
    _directory(kind)
    if (
        not isinstance(name, str)
        or not name.strip()
        or name in (".", "..")
        or any(char in name for char in ("/", "\\", "\0", ":"))
    ):
        raise ResourceError("resource name must be a basename")
    return name if name.endswith(".yaml") else name + ".yaml"


def resource_names(kind: str) -> Tuple[str, ...]:
    try:
        return tuple(
            sorted(
                file.name
                for file in _directory(kind).iterdir()
                if file.is_file() and file.name.endswith(".yaml")
            )
        )
    except OSError as exc:
        raise ResourceError("missing packaged resource directory %r" % kind) from exc


def resource_bytes(kind: str, name: str) -> bytes:
    filename = _filename(kind, name)
    try:
        return _directory(kind).joinpath(filename).read_bytes()
    except OSError as exc:
        raise ResourceError("missing packaged resource %s/%s" % (kind, filename)) from exc


def resource_text(kind: str, name: str) -> str:
    try:
        return resource_bytes(kind, name).decode("utf-8")
    except UnicodeError as exc:
        raise ResourceError("resource %s/%s is not UTF-8 text" % (kind, name)) from exc


class _YamlLoader(yaml.SafeLoader):
    def __init__(self, stream):
        super().__init__(stream)
        self._depth = 0

    def compose_node(self, parent, index):
        if self.check_event(yaml.AliasEvent):
            raise ResourceError("YAML aliases are not supported")
        if self._depth > 100:
            raise ResourceError("YAML nesting exceeds 100 levels")
        self._depth += 1
        try:
            return super().compose_node(parent, index)
        finally:
            self._depth -= 1

    def construct_object(self, node, deep=False):
        value = super().construct_object(node, deep=deep)
        if type(value) in (int, float) and not _finite(value):
            raise ResourceError("YAML numeric values must be finite")
        return value

    def construct_mapping(self, node, deep=False):
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ResourceError("YAML object keys must be strings")
            if key in result:
                raise ResourceError("duplicate YAML key %r" % key)
            result[key] = self.construct_object(value_node, deep=deep)
        return result


def decode_yaml(text: str, label: str = "YAML") -> Any:
    try:
        return finite_json(yaml.load(text, Loader=_YamlLoader), ResourceError)
    except (yaml.YAMLError, ValueError, TypeError, RecursionError) as exc:
        raise ResourceError("%s: invalid YAML: %s" % (label, exc)) from None


def encode_yaml(value: Any) -> str:
    return yaml.safe_dump(finite_json(value), sort_keys=False, allow_unicode=True, width=100)


def resource_yaml(kind: str, name: str) -> Any:
    return decode_yaml(resource_text(kind, name), "%s/%s" % (kind, name))


# Configuration

def config_text(value, name, error=ValueError, empty=False):
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise error("%s must be a %sstring" % (name, "" if empty else "nonempty "))
    return value.strip() if not empty else value


def strings(values, name, error=ValueError, optional=False):
    if values is None and optional:
        return None
    if not isinstance(values, (list, tuple)) or not values:
        raise error("%s must be a nonempty list" % name)
    result = [config_text(value, name + " item", error) for value in values]
    if len(set(result)) != len(result):
        raise error("%s contains duplicates" % name)
    return result


def positive_integer(value, name, error=ValueError):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise error("%s must be an integer >= 1" % name)
    return value


def boolean(value, name, error=ValueError):
    if not isinstance(value, bool):
        raise error("%s must be boolean" % name)
    return value


def finite_json(value, error=ValueError, depth=0):
    if depth > 100:
        raise error("configuration JSON nesting exceeds 100 levels")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: finite_json(item, error, depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item, error, depth + 1) for item in value]
    raise error("configuration must contain finite JSON values")


def object_fields(raw, fields, error):
    if not isinstance(raw, dict) or any(not isinstance(key, str) for key in raw):
        raise error("configuration must be an object with string keys")
    unknown = set(raw) - set(fields)
    if unknown:
        raise error("unknown configuration fields: %s" % sorted(unknown))


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON field %r" % key)
            result[key] = value
        return result

    def constant(value):
        raise ValueError("nonfinite JSON value %s" % value)

    return finite_json(json.loads(raw, object_pairs_hook=pairs, parse_constant=constant))


def _config_location(kind, source):
    path_argument = isinstance(source, Path)
    value = str(source) if path_argument else config_text(source, "config source")
    path = Path(value)
    explicit = path_argument or path.is_absolute() or "/" in value or "\\" in value
    name = value if value.endswith(".yaml") else value + ".yaml"
    return value, None if not explicit and name in resource_names(kind) else path


def load_config(kind, source, error):
    try:
        source, path = _config_location(kind, source)
        raw = resource_text(kind, source) if path is None else path.read_text(encoding="utf-8")
        return decode_yaml(raw, source)
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        raise error("%s: %s" % (source, exc)) from None


def config_input_path(kind, source):
    _, path = _config_location(kind, source)
    if path is None:
        return None
    try:
        return path.resolve()
    except (OSError, RuntimeError) as exc:
        raise ValueError("invalid config path: %s" % exc) from None


def parse_config(raw, cls, error, **defaults):
    object_fields(raw, cls.__dataclass_fields__, error)
    try:
        return cls(**dict(defaults, **raw))
    except TypeError as exc:
        raise error(str(exc)) from None


def load_config_object(kind, source, parser, error):
    config = parser(load_config(kind, source, error))
    object.__setattr__(config, "_source_path", config_input_path(kind, source))
    return config


# Families

@dataclass(frozen=True)
class FamilyIdentity:
    provider: str
    model: str
    family: str
    resolved: bool
    basis: str


def resolve_family(reader_spec: str) -> FamilyIdentity:
    if isinstance(reader_spec, str):
        provider, separator, persona = reader_spec.strip().partition(":")
        if separator and provider.lower() == "simulated" and persona in SIMULATED_PERSONAS:
            return FamilyIdentity("simulated", persona, "simulated", True, "simulated-fixture")
    return FamilyIdentity("unknown", "unknown", "unknown", False, "unknown")


# Compute

Number = Union[int, float]
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_COUNT_OPS = {
    "count_ge": operator.ge,
    "count_le": operator.le,
    "count_gt": operator.gt,
    "count_lt": operator.lt,
}
_OPS = {"argmin", "argmax", "rank"} | _COUNT_OPS.keys()


class ExpressionError(ValueError):
    pass


def _number(value: Any, label: str) -> Number:
    if not _finite(value):
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


def apply_op(
    compute_spec: Mapping[str, Any], values: Mapping[str, Any], order: List[str]
) -> Any:
    op = compute_spec["op"]
    if op in ("argmin", "argmax"):
        chooser = min if op == "argmin" else max
        return chooser(order, key=lambda item_id: values[item_id])
    if op == "rank":
        k = int(compute_spec["k"])
        reverse = compute_spec.get("order", "asc") == "desc"
        ordered = sorted(order, key=lambda item_id: values[item_id], reverse=reverse)
        k = max(1, min(k, len(ordered)))
        return ordered[k - 1]
    if op.startswith("count_"):
        threshold = compute_spec["threshold"]
        compare = _COUNT_OPS[op]
        return sum(compare(values[item_id], threshold) for item_id in order)
    raise ValueError("unknown op %r" % op)


def compute_answer_value(data: Mapping[str, Any], spec: Mapping[str, Any]) -> Any:
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
    order = [item["id"] for item in items]
    values = dict(zip(order, _numeric(items, spec["field"], derived)))
    return apply_op(spec, values, order)


def recompute(data: Mapping[str, Any], question: Mapping[str, Any]) -> str:
    if not isinstance(question, Mapping) or not isinstance(question.get("choices"), list):
        raise ValueError("question requires a compute spec and choices list")
    value = compute_answer_value(data, question.get("compute"))
    matches = []
    choice_ids = set()
    for choice in question["choices"]:
        if not isinstance(choice, Mapping):
            raise ValueError("question choices must be mappings")
        choice_id = choice.get("id")
        if not isinstance(choice_id, str) or not choice_id.strip() or choice_id in choice_ids:
            raise ValueError("question choices require unique nonempty string ids")
        choice_ids.add(choice_id)
        if choice.get("value") == value:
            matches.append(choice_id)
    if len(matches) != 1:
        raise ValueError(
            "computed answer %r for question %r must match exactly one choice"
            % (value, question.get("id"))
        )
    return matches[0]


# Schema

SCENARIO_HASH_FORMAT = "glassbox.scenario.v1"


class SchemaError(ValueError):
    pass


@dataclass(frozen=True)
class PresentationHints:
    primary_metric: str
    primary_extreme: str
    headline_field: Optional[str]
    cards_fields: List[str]
    detail_fields: List[str]


@dataclass(frozen=True)
class Choice:
    id: str
    text: str
    value: Any


@dataclass(frozen=True)
class Question:
    id: str
    stem: str
    compute: Dict[str, Any]
    choices: List[Choice]
    answer: str

    @property
    def target_field(self) -> Optional[str]:
        return self.compute.get("field")

    @property
    def correct_choice(self) -> Choice:
        for choice in self.choices:
            if choice.id == self.answer:
                return choice
        raise SchemaError("question %r answer %r is not among its choices" % (self.id, self.answer))


@dataclass(frozen=True)
class Scenario:
    id: str
    title: str
    domain: str
    description: str
    data: Dict[str, Any]
    questions: List[Question]

    @property
    def items(self) -> List[Dict[str, Any]]:
        return self.data["items"]

    @property
    def derived(self) -> Dict[str, str]:
        return self.data.get("derived", {})

    @property
    def presentation(self) -> PresentationHints:
        pres = self.data.get("presentation", {})
        raw_fields = sorted(raw_field_names(self.items))
        fields = sorted(self.derived)
        default_metric = fields[0] if fields else (raw_fields[0] if raw_fields else "")
        return PresentationHints(
            primary_metric=pres.get("primary_metric", default_metric),
            primary_extreme=pres.get("primary_extreme", "argmin"),
            headline_field=pres.get("headline_field"),
            cards_fields=list(pres.get("cards_fields", [])),
            detail_fields=list(pres.get("detail_fields", raw_fields)),
        )

    def item(self, item_id: str) -> Dict[str, Any]:
        for item in self.items:
            if item["id"] == item_id:
                return item
        raise KeyError("no item %r in scenario %r" % (item_id, self.id))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SchemaError(message)


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _finite(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _json_values(value: Any, depth: int = 0) -> None:
    _require(depth <= 100, "scenario JSON nesting exceeds 100 levels")
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, (int, float)):
        _require(_finite(value), "scenario numeric values must be finite")
    elif isinstance(value, dict):
        _require(all(isinstance(key, str) for key in value), "scenario object keys must be strings")
        for item in value.values():
            _json_values(item, depth + 1)
    elif isinstance(value, list):
        for item in value:
            _json_values(item, depth + 1)
    else:
        raise SchemaError("scenario values must contain JSON data")


def raw_field_names(items: List[Dict[str, Any]]) -> set:
    return {key for item in items for key in item if key not in ("id", "label")}


def _validate_presentation(
    pres: Dict[str, Any], sid: str, raw_fields: set, derived_names: set
) -> None:
    _require(isinstance(pres, dict), "%s: presentation must be an object" % sid)
    metric = pres.get("primary_metric")
    if "primary_metric" in pres:
        _require(
            _text(metric) and metric in raw_fields | derived_names,
            "%s: presentation.primary_metric is not a field" % sid,
        )
    _require(
        pres.get("primary_extreme", "argmin") in ("argmin", "argmax"),
        "%s: presentation.primary_extreme must be argmin/argmax" % sid,
    )
    headline = pres.get("headline_field")
    if headline is not None:
        _require(
            _text(headline) and headline in raw_fields,
            "%s: presentation.headline_field is not a raw field" % sid,
        )
    for key in ("cards_fields", "detail_fields"):
        value = pres.get(key, [])
        _require(isinstance(value, list), "%s: presentation.%s must be a list" % (sid, key))
        _require(
            all(_text(name) and name in raw_fields for name in value),
            "%s: presentation.%s references a non-raw field" % (sid, key),
        )
        _require(len(set(value)) == len(value), "%s: presentation.%s has duplicates" % (sid, key))
    display = pres.get("field_display", {})
    _require(isinstance(display, dict), "%s: field_display must be an object" % sid)
    for name, hints in display.items():
        _require(
            name in raw_fields | derived_names and isinstance(hints, dict),
            "%s: invalid field_display entry" % sid,
        )
        _require(
            all(isinstance(hints.get(key, ""), str) for key in ("unit", "value_label")),
            "%s: field display labels/units must be strings" % sid,
        )


def _parse_question(
    raw: Dict[str, Any],
    scenario_id: str,
    item_labels: Dict[str, str],
    raw_fields: set,
    derived_names: set,
) -> Question:
    qid = raw.get("id")
    _require(_text(qid), "%s: a question has no id" % scenario_id)
    where = "%s/%s" % (scenario_id, qid)
    _require(_text(raw.get("stem")), "%s: missing stem" % where)
    spec = raw.get("compute")
    _require(isinstance(spec, dict), "%s: missing compute spec" % where)
    try:
        validate_compute_spec(spec, len(item_labels))
    except ValueError as error:
        raise SchemaError("%s: %s" % (where, error)) from None
    op = spec["op"]
    _require(
        spec["field"] in raw_fields | derived_names,
        "%s: compute field is not a raw or derived field" % where,
    )
    raw_choices = raw.get("choices")
    _require(
        isinstance(raw_choices, list) and 2 <= len(raw_choices) <= 26,
        "%s: needs 2 to 26 choices for A-Z answer letters" % where,
    )
    choices: List[Choice] = []
    seen_ids, seen_text, seen_values = set(), set(), set()
    for rc in raw_choices:
        _require(isinstance(rc, dict), "%s: a choice is not an object" % where)
        cid, text, value = rc.get("id"), rc.get("text"), rc.get("value")
        _require(_text(cid) and cid not in seen_ids, "%s: missing or duplicate choice id" % where)
        _require(_text(text), "%s: choice %r has no text" % (where, cid))
        visible = " ".join(text.split()).casefold()
        _require(visible not in seen_text, "%s: duplicate visible choice answer" % where)
        if op.startswith("count_"):
            _require(
                type(value) is int and value >= 0,
                "%s: count choice values must be nonnegative integers" % where,
            )
            _require(
                re.fullmatch(r"0|[1-9][0-9]*", text.strip()) is not None
                and text.strip() == str(value),
                "%s: count choice text must display its numeric value" % where,
            )
        else:
            _require(
                _text(value) and value in item_labels, "%s: choice value is not an item id" % where
            )
            _require(
                item_labels[value].casefold() in text.casefold(),
                "%s: choice text does not name its item's label" % where,
            )
        _require(value not in seen_values, "%s: duplicate choice value" % where)
        seen_ids.add(cid)
        seen_text.add(visible)
        seen_values.add(value)
        choices.append(Choice(cid, text, value))
    answer = raw.get("answer")
    _require(
        isinstance(answer, str) and answer in seen_ids, "%s: answer is not among choices" % where
    )
    return Question(qid, raw["stem"], copy.deepcopy(spec), choices, answer)


def parse_scenario(raw: Dict[str, Any]) -> Scenario:
    _require(isinstance(raw, dict), "scenario must be an object")
    _json_values(raw)
    sid = raw.get("id")
    _require(_text(sid), "scenario has no id")
    for key in ("title", "domain", "description"):
        _require(_text(raw.get(key)), "%s: missing %s" % (sid, key))
    data = raw.get("data")
    _require(isinstance(data, dict), "%s: missing data" % sid)
    items = data.get("items")
    _require(isinstance(items, list) and items, "%s: needs items" % sid)
    item_labels: Dict[str, str] = {}
    labels = set()
    for item in items:
        _require(
            isinstance(item, dict) and _text(item.get("id")),
            "%s: every item needs a string id" % sid,
        )
        _require(item["id"] not in item_labels, "%s: duplicate item id" % sid)
        _require(_text(item.get("label")), "%s: every item needs a label" % sid)
        label = " ".join(item["label"].split()).casefold()
        _require(label not in labels, "%s: duplicate item label" % sid)
        item_labels[item["id"]] = item["label"]
        labels.add(label)
    for key in ("unit", "value_label"):
        _require(isinstance(data.get(key, ""), str), "%s: %s must be a string" % (sid, key))
    derived = data.get("derived", {})
    _require(isinstance(derived, dict), "%s: derived must be an object" % sid)
    _require(
        all(_text(name) and _text(expr) for name, expr in derived.items()),
        "%s: derived names/expressions must be nonempty strings" % sid,
    )
    raw_fields = raw_field_names(items)
    _require(not raw_fields.intersection(derived), "%s: derived names duplicate raw fields" % sid)
    if "presentation" in data:
        _validate_presentation(data["presentation"], sid, raw_fields, set(derived))
    raw_questions = raw.get("questions")
    _require(isinstance(raw_questions, list) and raw_questions, "%s: needs questions" % sid)
    questions: List[Question] = []
    seen = set()
    for raw_question in raw_questions:
        _require(isinstance(raw_question, dict), "%s: a question is not an object" % sid)
        question = _parse_question(raw_question, sid, item_labels, raw_fields, set(derived))
        _require(question.id not in seen, "%s: duplicate question id" % sid)
        seen.add(question.id)
        questions.append(question)
        if question.target_field in raw_fields:
            _require(
                all(_finite(item.get(question.target_field)) for item in items),
                "%s/%s: target values must be finite numeric" % (sid, question.id),
            )
    scenario = Scenario(
        sid, raw["title"], raw["domain"], raw["description"], copy.deepcopy(data), questions
    )
    metric = scenario.presentation.primary_metric
    if metric in raw_fields:
        _require(
            all(_finite(item.get(metric)) for item in items),
            "%s: presentation.primary_metric %r must be numeric for every item" % (sid, metric),
        )
    return scenario


def validate_structure(scenario: Scenario) -> None:
    _require(isinstance(scenario, Scenario), "expected a Scenario")
    try:
        parse_scenario(asdict(scenario))
    except (TypeError, RecursionError) as exc:
        raise SchemaError("malformed scenario structure: %s" % exc) from exc


def scenario_payload(scenario: Scenario) -> Dict[str, Any]:
    _require(isinstance(scenario, Scenario), "expected a Scenario")
    try:
        payload = asdict(scenario)
        payload["resolved_presentation"] = asdict(scenario.presentation)
        _json_values(payload)
        return payload
    except (TypeError, KeyError, AttributeError, RecursionError) as exc:
        raise SchemaError("cannot snapshot malformed scenario: %s" % exc) from exc


def scenario_sha256(scenario: Scenario) -> str:
    encoded = json.dumps(
        scenario_payload(scenario),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_scenario(path: str) -> Scenario:
    try:
        with open(path, "r", encoding="utf-8") as stream:
            raw = decode_yaml(stream.read(), str(path))
        return parse_scenario(raw)
    except (OSError, UnicodeError, ResourceError) as exc:
        raise SchemaError("%s: %s" % (path, exc)) from exc


def load_all_scenarios(directory: Optional[str] = None) -> Dict[str, Scenario]:
    out: Dict[str, Scenario] = {}
    try:
        names = resource_names("scenarios") if directory is None else sorted(os.listdir(directory))
        for name in names:
            if not name.endswith(".yaml"):
                continue
            scenario = (
                parse_scenario(resource_yaml("scenarios", name))
                if directory is None
                else load_scenario(os.path.join(directory, name))
            )
            _require(scenario.id not in out, "duplicate scenario id %r" % scenario.id)
            out[scenario.id] = scenario
    except (OSError, ResourceError) as exc:
        raise SchemaError("cannot load scenarios: %s" % exc) from exc
    return dict(sorted(out.items()))


# Validate

def validate_scenario(scenario: Scenario) -> List[str]:
    errors: List[str] = []
    try:
        validate_structure(scenario)
    except SchemaError as exc:
        return [str(exc)]

    for name, expression in scenario.derived.items():
        for item in scenario.items:
            try:
                evaluate_expr(expression, item)
            except ExpressionError as exc:
                errors.append(
                    "%s: derived %r fails on item %r: %s" % (scenario.id, name, item["id"], exc)
                )

    for question in scenario.questions:
        try:
            recomputed = recompute(scenario.data, asdict(question))
        except (ValueError, KeyError) as exc:
            errors.append("%s/%s: cannot recompute answer: %s" % (scenario.id, question.id, exc))
            continue
        if recomputed != question.answer:
            errors.append(
                "%s/%s: authored answer %r but source data recomputes to %r"
                % (scenario.id, question.id, question.answer, recomputed)
            )
    return errors


def validate_all(directory: Optional[str] = None) -> Tuple[bool, List[str]]:
    scenarios: Dict[str, Scenario] = load_all_scenarios(directory)
    if not scenarios:
        return False, [
            "no scenarios found in %s"
            % (directory if directory is not None else "packaged scenarios")
        ]

    report: List[str] = []
    all_errors: List[str] = []
    for scenario_id, scenario in scenarios.items():
        errors = validate_scenario(scenario)
        all_errors.extend(errors)
        status = "ok" if not errors else "FAIL"
        report.append(
            "%-16s %s  (%d question%s)"
            % (scenario_id, status, len(scenario.questions), "" if len(scenario.questions) == 1 else "s")
        )
        report.extend("    - " + error for error in errors)
    return (not all_errors), report


# Output paths

def _within(path: Path, parent: Path) -> bool:
    child = tuple(part.casefold() for part in path.parts)
    ancestor = tuple(part.casefold() for part in parent.parts)
    return child[: len(ancestor)] == ancestor


def validate_output_dir(out_dir, *, protected_paths=()) -> Path:
    try:
        if not isinstance(out_dir, (str, Path)) or not str(out_dir).strip():
            raise ValueError("output directory must be a nonempty path")
        original = Path(out_dir)
        if original.is_symlink():
            raise ValueError("output directory must not be a symlink")
        output = original.resolve()
        if output.exists() and not output.is_dir():
            raise ValueError("output directory must be a directory")
        package = Path(__file__).resolve().parent
        roots = [package]
        repository = package.parent.parent
        if package.parent.name == "src":
            roots.extend(
                repository / name
                for name in ("src", "data", "docs", "tests", "scripts", ".github")
            )
        roots.extend(
            Path(path).resolve() for path in protected_paths if Path(path).is_dir()
        )
        if any(_within(output, root) or _within(root, output) for root in roots):
            raise ValueError("output directory intersects package or input sources")
        return output
    except (OSError, RuntimeError, TypeError) as error:
        raise ValueError("cannot resolve output directory: %s" % error) from error


def validate_output_files(out_dir, names, *, protected_paths=()) -> tuple:
    try:
        inputs = tuple(Path(path) for path in protected_paths)
        directory = validate_output_dir(out_dir, protected_paths=inputs)
        if isinstance(names, (str, bytes)):
            raise ValueError("output names must be a sequence of basenames")
        names = tuple(names)
        if not names or any(
            not isinstance(name, str)
            or not name
            or name in (".", "..")
            or any(c in name for c in ("/", "\\", "\0"))
            for name in names
        ):
            raise ValueError("output names must be nonempty basenames")
        if len({name.casefold() for name in names}) != len(names):
            raise ValueError("output filenames alias each other")
        destinations = tuple(directory / name for name in names)
        resolved_inputs = tuple(path.resolve() for path in inputs)
        for destination in destinations:
            if destination.is_symlink():
                raise ValueError(
                    "output artifact must not be a symlink: %s" % destination
                )
            if destination.exists() and (
                not destination.is_file() or destination.stat().st_nlink > 1
            ):
                raise ValueError("output artifact must be a regular unlinked file")
            for source, resolved in zip(inputs, resolved_inputs):
                alias = str(destination).casefold() == str(resolved).casefold()
                if not alias and destination.exists() and source.exists():
                    alias = os.path.samefile(destination, source)
                if alias:
                    raise ValueError("output artifact aliases input: %s" % source)
        return destinations
    except (OSError, RuntimeError, TypeError) as error:
        raise ValueError("cannot resolve output artifacts: %s" % error) from error


# Interfaces

VARIANT_FEATURES: Dict[str, FrozenSet[str]] = {
    "raw": frozenset({"all_raw"}),
    "baseline": frozenset({"all_raw", "as_text"}),
    "cards": frozenset({"headline", "cards_context", "polished"}),
    "table": frozenset({"detail", "derived"}),
    "annotated": frozenset({"detail", "derived", "highlight", "sorted", "polished"}),
}

REFERENCE_VARIANTS = ("raw", "baseline")
INTERFACE_VARIANTS = ("cards", "table", "annotated")
ALL_VARIANTS = REFERENCE_VARIANTS + INTERFACE_VARIANTS


@dataclass(frozen=True)
class Presentation:
    scenario_id: str
    variant: str
    features: FrozenSet[str]
    order: List[str]
    shown: Dict[str, Dict[str, Any]]
    emphasized_field: Optional[str]
    highlight_item: Optional[str]
    primary_metric: str
    primary_extreme: str
    unit: str = ""
    value_label: str = ""
    title: str = ""
    description: str = ""
    labels: Dict[str, str] = field(default_factory=dict)
    field_units: Dict[str, str] = field(default_factory=dict)
    field_labels: Dict[str, str] = field(default_factory=dict)
    highlight_label: str = "best"
    target_question_id: Optional[str] = None
    target_operation: Optional[str] = None
    target_stem: str = ""

    def unit_for(self, name: str) -> str:
        return self.field_units.get(name, self.unit)

    def label_for(self, name: str) -> str:
        label = self.value_label if name == self.primary_metric else ""
        if not label.strip():
            label = self.field_labels.get(name, "")
        return label if label.strip() else _field_label(name)

    @property
    def polished(self) -> bool:
        return "polished" in self.features

    def shows_field(self, item_id: str, field_name: str) -> bool:
        return field_name in self.shown.get(item_id, {})

    def to_html(self) -> str:
        return render_html(self)

    def to_text(self) -> str:
        return render_text(self)


def _num(value: Any) -> str:
    if isinstance(value, float):
        representation = repr(value)
        if not value.is_integer() or "e" in representation or representation == "-0.0":
            return format(value, ",")
        value = int(value)
    if isinstance(value, int):
        return "{:,}".format(value)
    return str(value)


def format_value(v: Any, unit: str) -> str:
    number = _num(v)
    if unit == "$":
        return "$" + number
    if unit == "$k":
        return "$" + number + "k"
    return number + " " + unit if unit else number


def validate_display_values(presentation: Presentation) -> None:
    for item_id in presentation.order:
        for name, value in presentation.shown[item_id].items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            message = "numeric display loses precision for %s/%s" % (item_id, name)
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(message)
            units = {presentation.field_units.get(name, "")}
            if name in (presentation.primary_metric, presentation.emphasized_field):
                units.add(presentation.unit_for(name))
            for unit in units:
                formatted = format_value(value, unit)
                prefix = "$" if unit in ("$", "$k") else ""
                suffix = "k" if unit == "$k" else (" " + unit if unit and unit != "$" else "")
                if (not isinstance(formatted, str) or not formatted.startswith(prefix)
                        or not formatted.endswith(suffix)):
                    raise ValueError(message)
                token = formatted[len(prefix):len(formatted) - len(suffix) if suffix else None]
                try:
                    parsed = (float if isinstance(value, float) else int)(token.replace(",", ""))
                except (ValueError, OverflowError) as error:
                    raise ValueError(message) from error
                if parsed != value or (isinstance(value, float) and value == 0
                                       and math.copysign(1, parsed) != math.copysign(1, value)):
                    raise ValueError(message)


def build_presentation(
    scenario: Scenario,
    features: FrozenSet[str],
    variant: str = "custom",
    target: Optional[Question] = None,
) -> Presentation:
    validate_structure(scenario)
    if target is not None and (
        not isinstance(target, Question) or target not in scenario.questions
    ):
        raise ValueError("target must be a question from this scenario")
    if not isinstance(variant, str) or not variant.strip():
        raise ValueError("presentation variant must be nonempty text")
    if not isinstance(features, (set, frozenset)) or any(
        not isinstance(name, str) for name in features
    ):
        raise ValueError("presentation features must be a set of strings")
    pres = scenario.presentation
    metric = target.target_field if target is not None else pres.primary_metric
    op = target.compute["op"] if target is not None else pres.primary_extreme
    direction = op if op in ("argmin", "argmax") else "argmin"
    if op == "rank":
        direction = "argmax" if target.compute.get("order", "asc") == "desc" else "argmin"
    details = pres.detail_fields
    if target is not None:
        inputs = (
            expr_fields(scenario.derived[metric])
            if metric in scenario.derived
            else {metric}
        )
        details = [name for name in pres.detail_fields if name in inputs]
        details += sorted(inputs - set(details))
    visible = []
    if "all_raw" in features:
        visible.extend(sorted(raw_field_names(scenario.items)))
    if "detail" in features:
        visible.extend(details)
    if "cards_context" in features:
        visible.extend(pres.cards_fields)
    if "headline" in features and pres.headline_field:
        visible.append(pres.headline_field)
    visible = list(dict.fromkeys(visible))
    show_metric = "derived" in features
    values = dict(
        zip(
            (item["id"] for item in scenario.items),
            _numeric(scenario.items, metric, scenario.derived),
        )
    )
    shown: Dict[str, Dict[str, Any]] = {}
    for item in scenario.items:
        row = {name: item[name] for name in visible if name in item}
        if show_metric:
            row[metric] = values[item["id"]]
        shown[item["id"]] = row
    if show_metric:
        emphasized = metric
    elif ("headline" in features or "all_raw" in features) and pres.headline_field:
        emphasized = pres.headline_field
    else:
        emphasized = None
    highlight_item, highlight_label = None, "best"
    if "highlight" in features and op in ("argmin", "argmax", "rank"):
        if target is not None:
            highlight_item = compute_answer_value(scenario.data, target.compute)
        else:
            chooser = min if direction == "argmin" else max
            highlight_item = chooser(values, key=lambda iid: values[iid])
        if op == "rank":
            highlight_label = "rank %d" % target.compute["k"]
    order = [item["id"] for item in scenario.items]
    if "sorted" in features:
        order = sorted(order, key=lambda iid: values[iid], reverse=direction == "argmax")
    unit = scenario.data.get("unit", "") if metric == pres.primary_metric else ""
    label = scenario.data.get("value_label", "") if metric == pres.primary_metric else ""
    if not label.strip():
        label = _field_label(metric)
    display = scenario.data.get("presentation", {}).get("field_display", {})
    units = {metric: unit}
    if pres.headline_field and pres.headline_field != metric:
        units[pres.headline_field] = scenario.data.get("unit", "")
    labels = {metric: label}
    for name, hints in display.items():
        units[name] = hints.get("unit", units.get(name, ""))
        field_label = hints.get("value_label", "")
        labels[name] = field_label if field_label.strip() else _field_label(name)
    unit, label = units[metric], labels[metric]
    return Presentation(
        scenario_id=scenario.id,
        variant=variant,
        features=frozenset(features),
        order=order,
        shown=shown,
        emphasized_field=emphasized,
        highlight_item=highlight_item,
        primary_metric=metric,
        primary_extreme=direction,
        unit=unit,
        value_label=label,
        title=scenario.title,
        description=scenario.description,
        labels={item["id"]: item["label"] for item in scenario.items},
        field_units=units,
        field_labels=labels,
        highlight_label=highlight_label,
        target_question_id=target.id if target is not None else None,
        target_operation=op if target is not None else None,
        target_stem=target.stem if target is not None else "",
    )


def build_variant(scenario: Scenario, name: str, target: Optional[Question] = None) -> Presentation:
    if name not in VARIANT_FEATURES:
        raise KeyError("unknown interface variant %r" % name)
    return build_presentation(scenario, VARIANT_FEATURES[name], variant=name, target=target)


def with_features(
    scenario: Scenario,
    base: Presentation,
    added: FrozenSet[str],
    new_name: Optional[str] = None,
    target: Optional[Question] = None,
) -> Presentation:
    if target is None and base.target_question_id is not None:
        target = next((q for q in scenario.questions if q.id == base.target_question_id), None)
        if target is None:
            raise ValueError("presentation target question no longer exists")
    combined = frozenset(base.features | added)
    name = new_name or (base.variant + "+" + "+".join(sorted(added)))
    return build_presentation(scenario, combined, variant=name, target=target)


_CSS = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
       margin: 0; padding: 28px; background: #ffffff; color: #16181d; width: 760px; }
h1 { font-size: 22px; margin: 0 0 4px; }
p.desc { color: #55606e; margin: 0 0 20px; font-size: 14px; max-width: 60ch; }
table { border-collapse: collapse; table-layout: fixed; width: 100%; font-size: 15px; }
th, td { text-align: left; padding: 10px 14px; border-bottom: 1px solid #e6e8ec; }
th { font-weight: 600; color: #55606e; font-size: 13px; text-transform: uppercase; letter-spacing: .03em;
     white-space: normal; overflow-wrap: break-word; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.cards { display: flex; gap: 16px; flex-wrap: wrap; }
.card { flex: 1 1 210px; border: 1px solid #e6e8ec; border-radius: 14px; padding: 18px 20px;
        box-shadow: 0 1px 3px rgba(16,24,40,.06); background: #fbfcfe; }
.card h2 { font-size: 16px; margin: 0 0 12px; }
.card .headline { font-size: 26px; font-weight: 700; margin: 0 0 4px; }
.card .headline-label { font-size: 12px; color: #55606e; text-transform: uppercase; letter-spacing: .03em; }
.card ul { list-style: none; padding: 0; margin: 14px 0 0; font-size: 13px; color: #55606e; }
.card li { display: flex; justify-content: space-between; padding: 2px 0; }
.mark { outline: 2px solid #1f7a45; background: #eafaf0; }
.badge { display: inline-block; margin-left: 8px; font-size: 11px; font-weight: 600; color: #1f7a45; }
pre.plain { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 14px;
            white-space: pre-wrap; line-height: 1.5; color: inherit; }
@media (prefers-color-scheme: dark) {
  body { background: #16181d; color: #e6e8ec; }
  p.desc, th, .card .headline-label, .card ul { color: #9aa4b2; }
  th, td { border-bottom-color: #2a2e37; }
  .card { background: #1c1f26; border-color: #2a2e37; }
  .mark { background: #133024; outline-color: #3ecf8e; }
  .badge { color: #3ecf8e; }
}
"""


def _doc(title: str, body: str) -> str:
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=760">'
        "<title>" + html.escape(title) + "</title><style>" + _CSS + "</style></head>"
        "<body>" + body + "</body></html>"
    )


def _field_label(name: str) -> str:
    labels = {
        "apr_pct": "APR (%)",
        "stated_total": "advertised total",
        "term_months": "term (months)",
        "included_gb": "included data (GB)",
        "usage_gb": "data usage (GB)",
        "overage_per_gb": "overage per GB",
    }
    return labels.get(name, name.replace("_", " "))


def render_html(p: Presentation) -> str:
    header = f'<h1>{html.escape(p.title)}</h1><p class="desc">{html.escape(p.description)}</p>'
    if p.target_stem:
        header += f'<p class="desc">Target: {html.escape(p.target_stem)}</p>'
    if "as_text" in p.features:
        body = f'<pre class="plain">{html.escape(render_text(p, include_header=False))}</pre>'
    else:
        body = _render_cards(p) if p.polished else _render_table(p)
    return _doc(p.title, header + body)


def _render_cards(p: Presentation) -> str:
    cards = []
    for item_id in p.order:
        row = p.shown[item_id]
        highlighted = item_id == p.highlight_item
        marked = " mark" if highlighted else ""
        badge = f'<span class="badge">{html.escape(p.highlight_label)}</span>' if highlighted else ""
        headline = ""
        field = p.emphasized_field
        if field and field in row:
            value = html.escape(format_value(row[field], p.unit_for(field)))
            headline = (
                f'<div class="headline">{value}</div>'
                f'<div class="headline-label">{html.escape(p.label_for(field))}</div>'
            )
        details = "".join(
            f'<li><span>{html.escape(p.label_for(name))}</span>'
            f'<span>{html.escape(format_value(value, p.field_units.get(name, "")))}</span></li>'
            for name, value in row.items() if name != field
        )
        cards.append(
            f'<div class="card{marked}"><h2>{html.escape(p.labels.get(item_id, item_id))}'
            f'{badge}</h2>{headline}<ul>{details}</ul></div>'
        )
    return '<div class="cards">' + "".join(cards) + '</div>'


def _render_table(p: Presentation) -> str:
    columns = list(dict.fromkeys(
        name for item_id in p.order for name in p.shown[item_id] if name != p.primary_metric
    ))
    if any(p.primary_metric in p.shown[item_id] for item_id in p.order):
        columns.append(p.primary_metric)
    header = '<tr><th>option</th>' + ''.join(
        f'<th>{html.escape(p.label_for(name))}</th>' for name in columns
    ) + '</tr>'
    rows = []
    for item_id in p.order:
        row = p.shown[item_id]
        marked = ' class="mark"' if item_id == p.highlight_item else ""
        cells = [f'<td>{html.escape(p.labels.get(item_id, item_id))}</td>']
        for name in columns:
            value = ""
            primary = name == p.primary_metric
            if name in row and (not primary or row[name] is not None):
                unit = p.unit_for(name) if primary else p.field_units.get(name, "")
                value = html.escape(format_value(row[name], unit))
            cells.append(f'<td class="num">{value}</td>')
        rows.append(f'<tr{marked}>{"".join(cells)}</tr>')
    return '<table>' + header + ''.join(rows) + '</table>'


def render_text(p: Presentation, include_header: bool = True) -> str:
    lines = []
    if include_header:
        lines.extend((p.title, p.description))
        if p.target_stem:
            lines.append("Target: " + p.target_stem)
        lines.append("")
    for item_id in p.order:
        parts = []
        for name, value in p.shown[item_id].items():
            primary = name == p.primary_metric
            emphasized = name == p.emphasized_field
            unit = p.unit_for(name) if primary or emphasized else p.field_units.get(name, "")
            separator = " = " if primary else (": " if emphasized else " ")
            parts.append(p.label_for(name) + separator + format_value(value, unit))
        marker = "  <- highlighted as " + p.highlight_label if item_id == p.highlight_item else ""
        lines.append(f'- {p.labels.get(item_id, item_id)}: {", ".join(parts)}{marker}')
    return "\n".join(lines)


# Stimuli

@dataclass(frozen=True)
class Stimulus:
    presentation: Presentation
    kind: str
    text: str
    image_path: Optional[str] = None
    image_sha256: Optional[str] = None
    render_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.presentation, Presentation):
            raise ValueError("stimulus requires a Presentation")
        if self.kind not in ("image", "spec") or not isinstance(self.text, str):
            raise ValueError("stimulus requires an image/spec kind and text")
        if self.kind == "image":
            if not isinstance(self.image_path, str) or not self.image_path.strip():
                raise ValueError("image stimulus requires a nonempty image path")
        elif self.image_path is not None or self.image_sha256 is not None:
            raise ValueError("spec stimulus must not include image fields")
        if not isinstance(self.render_metadata, dict):
            raise ValueError("render_metadata must be a dict")
        object.__setattr__(self, "presentation", copy.deepcopy(self.presentation))
        object.__setattr__(self, "render_metadata", copy.deepcopy(self.render_metadata))

    @property
    def text_sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


# Render

DEFAULT_CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
_WINDOW = "820,620"
_SCALE = "2"
_RENDER_TIMEOUT = 15.0
_EXIT_GRACE = 1.0
_STOP_TIMEOUT = 2.0
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_FILENAME_COMPONENT_LIMIT = 60


class RenderError(RuntimeError):
    pass


def chrome_available() -> bool:
    return os.path.isfile(DEFAULT_CHROME) and os.access(DEFAULT_CHROME, os.X_OK)


def renderer_version(skip: bool = False) -> str:
    if skip or not chrome_available():
        return "skip-render"
    try:
        result = subprocess.run(
            [DEFAULT_CHROME, "--version"], capture_output=True, text=True, timeout=20
        )
        return (
            result.stdout.strip()
            if result.returncode == 0 and result.stdout.strip()
            else "chrome (version unknown)"
        )
    except (OSError, subprocess.SubprocessError):
        return "chrome (version unavailable)"


def _validate_png(data: bytes) -> Dict[str, int]:
    if not data.startswith(_PNG_SIGNATURE) or len(data) > 64 * 1024 * 1024:
        raise RenderError("chrome output is not a bounded PNG")
    offset = 8
    header = None
    image_data = bytearray()
    ended = saw_data = False
    palette = None
    data_ended = False
    while offset < len(data):
        if offset + 12 > len(data):
            raise RenderError("PNG chunk is truncated")
        size = struct.unpack(">I", data[offset : offset + 4])[0]
        kind = data[offset + 4 : offset + 8]
        if not re.fullmatch(b"[A-Za-z]{4}", kind):
            raise RenderError("PNG chunk type is invalid")
        end = offset + size + 12
        if end > len(data):
            raise RenderError("PNG chunk is truncated")
        content = data[offset + 8 : end - 4]
        crc = struct.unpack(">I", data[end - 4 : end])[0]
        if (zlib.crc32(kind + content) & 0xFFFFFFFF) != crc:
            raise RenderError("PNG chunk CRC is invalid")
        if header is None and kind != b"IHDR":
            raise RenderError("PNG must start with IHDR")
        if kind == b"IHDR":
            if header is not None or size != 13:
                raise RenderError("PNG IHDR is invalid")
            header = struct.unpack(">IIBBBBB", content)
        elif kind == b"PLTE":
            if saw_data or palette is not None or size == 0 or size > 768 or size % 3:
                raise RenderError("PNG palette is invalid")
            palette = size // 3
        elif kind == b"IDAT":
            if data_ended:
                raise RenderError("PNG image chunks must be consecutive")
            saw_data = True
            image_data.extend(content)
        elif kind == b"IEND":
            if size != 0 or end != len(data):
                raise RenderError("PNG IEND is invalid")
            ended = True
            break
        elif not kind[0] & 32:
            raise RenderError("PNG has an unknown critical chunk")
        if saw_data and kind != b"IDAT":
            data_ended = True
        offset = end
    if header is None or not saw_data or not ended:
        raise RenderError("PNG lacks required image chunks")
    width, height, depth, color, compression, filtering, interlace = header
    depths = {0: (1, 2, 4, 8, 16), 2: (8, 16), 3: (1, 2, 4, 8), 4: (8, 16), 6: (8, 16)}
    if (
        not width
        or not height
        or width * height > 32_000_000
        or color not in depths
        or depth not in depths[color]
        or (compression, filtering, interlace) != (0, 0, 0)
    ):
        raise RenderError("PNG IHDR dimensions/encoding are invalid")
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[color]
    if color == 3 and (palette is None or palette > 2**depth):
        raise RenderError("indexed PNG requires a valid palette")
    row_size = (width * channels * depth + 7) // 8 + 1
    expected = row_size * height
    if expected > 128 * 1024 * 1024:
        raise RenderError("PNG decoded data exceeds the screenshot limit")
    try:
        decoder = zlib.decompressobj()
        rows = decoder.decompress(bytes(image_data), expected + 1)
        if (
            len(rows) != expected
            or not decoder.eof
            or decoder.unused_data
            or decoder.unconsumed_tail
            or any(rows[index] > 4 for index in range(0, len(rows), row_size))
        ):
            raise RenderError("PNG image rows are invalid")
    except zlib.error as exc:
        raise RenderError("PNG image compression is invalid") from exc
    return {"width": width, "height": height}


def _cleanup_chrome(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=_STOP_TIMEOUT)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=_STOP_TIMEOUT)
    if os.name == "posix":
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                break


def _capture(command, staged: Path, log_path: Path) -> Dict[str, Any]:
    process = None
    started = time.monotonic()
    confirmed_at = confirmed_data = None
    try:
        with log_path.open("wb") as log:
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=os.name == "posix",
            )
            while True:
                status = process.poll()
                now = time.monotonic()
                messages = log_path.read_bytes()
                if status is not None and status != 0:
                    raise RenderError(
                        "chrome exited %s: %s"
                        % (status, messages.decode("utf-8", errors="replace")[-400:])
                    )
                data, dimensions = None, None
                if staged.is_file():
                    data = staged.read_bytes()
                    try:
                        dimensions = _validate_png(data)
                    except RenderError:
                        if status is not None:
                            raise
                marker = (
                    ("%d bytes written to file %s" % (len(data), staged)).encode("utf-8")
                    if dimensions is not None
                    else None
                )
                confirmed = marker is not None and marker in messages.splitlines()
                if status == 0:
                    if dimensions is None:
                        raise RenderError("chrome exited without a complete screenshot")
                    return {
                        **dimensions,
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "completion": "natural-exit",
                    }
                if now - started >= _RENDER_TIMEOUT:
                    raise RenderError(
                        "chrome did not finish capture within %g seconds" % _RENDER_TIMEOUT
                    )
                if confirmed:
                    if confirmed_at is None:
                        confirmed_at, confirmed_data = now, data
                    if data != confirmed_data:
                        raise RenderError("chrome screenshot changed after capture confirmation")
                    if now - confirmed_at >= _EXIT_GRACE:
                        status = process.poll()
                        if status is not None:
                            continue
                        process.terminate()
                        status = process.wait(timeout=_STOP_TIMEOUT)
                        if status != 0:
                            raise RenderError("owned chrome shutdown exited %s" % status)
                        if staged.read_bytes() != confirmed_data:
                            raise RenderError("chrome screenshot changed during shutdown")
                        return {
                            **dimensions,
                            "sha256": hashlib.sha256(data).hexdigest(),
                            "completion": "validated-capture-controlled-shutdown",
                        }
                time.sleep(0.05)
    finally:
        if process is not None:
            _cleanup_chrome(process)


def _screenshot(html: str, png_path: str) -> Dict[str, Any]:
    destination = Path(png_path)
    image_path = None
    try:
        with tempfile.TemporaryDirectory(prefix="glassbox_capture_", dir=destination.parent) as tmp:
            staging = Path(tmp)
            descriptor, temporary = tempfile.mkstemp(suffix=".png", dir=destination.parent)
            os.close(descriptor)
            image_path = Path(temporary)
            image_path.unlink()
            html_path, profile = staging / "view.html", staging / "profile"
            html_path.write_text(html, encoding="utf-8")
            profile.mkdir()
            command = [
                DEFAULT_CHROME,
                "--headless=new",
                "--disable-gpu",
                "--hide-scrollbars",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-background-networking",
                "--disable-component-update",
                "--disable-sync",
                "--disable-extensions",
                "--password-store=basic",
                "--use-mock-keychain",
                "--user-data-dir=" + str(profile),
                "--force-device-scale-factor=" + _SCALE,
                "--window-size=" + _WINDOW,
                "--default-background-color=FFFFFFFF",
                "--screenshot=" + str(image_path),
                html_path.resolve().as_uri(),
            ]
            image = _capture(command, image_path, staging / "capture.log")
        data = image_path.read_bytes()
        _validate_png(data)
        if hashlib.sha256(data).hexdigest() != image["sha256"]:
            raise RenderError("chrome screenshot changed before replacement")
        os.replace(str(image_path), str(destination))
        return image
    except (OSError, subprocess.SubprocessError) as exc:
        raise RenderError("chrome render failed: %s" % exc) from exc
    finally:
        if image_path is not None:
            image_path.unlink(missing_ok=True)


def _component(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", value).strip("_") or "view"
    if safe != value or value != value.casefold() or len(safe) > _FILENAME_COMPONENT_LIMIT:
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
        safe = safe[:_FILENAME_COMPONENT_LIMIT] + "_" + digest
    return safe


def render_stimulus(
    presentation: Presentation, out_dir: Optional[str] = None, skip: bool = False
) -> Stimulus:
    presentation = copy.deepcopy(presentation)
    validate_display_values(presentation)
    text = presentation.to_text()
    if skip or not chrome_available():
        return Stimulus(
            presentation, "spec", text, render_metadata={"mode": "spec", "name": "skip-render"}
        )
    directory = (
        Path(out_dir) if out_dir is not None else Path(tempfile.mkdtemp(prefix="glassbox_render_"))
    )
    try:
        stem = _component(presentation.scenario_id) + "__" + _component(presentation.variant)
        if presentation.target_question_id is not None:
            stem += "__" + _component(presentation.target_question_id)
        destination = validate_output_files(directory, [stem + ".png"])[0]
        destination.parent.mkdir(parents=True, exist_ok=True)
        image = _screenshot(presentation.to_html(), str(destination))
    except OSError as exc:
        raise RenderError("cannot create screenshot output: %s" % exc) from exc
    metadata = {
        "mode": "image",
        "name": "chrome",
        "version": renderer_version(),
        "window": [int(value) for value in _WINDOW.split(",")],
        "scale": int(_SCALE),
        "completion": image["completion"],
        "color_scheme": "system",
        "font_stack": "system",
        "png_dimensions": [image["width"], image["height"]],
    }
    return Stimulus(
        presentation,
        "image",
        text,
        image_path=str(destination),
        image_sha256=image["sha256"],
        render_metadata=metadata,
    )


# Readers

@dataclass(frozen=True)
class Answer:
    choice_id: Optional[str]
    method: str = ""
    raw: str = ""


class Reader:
    name: str = "reader"
    family: str = ""
    simulated: bool = False
    deterministic: Optional[bool] = None
    provider: str = "unknown"
    model_family: str = "unknown"
    family_resolved: bool = False
    family_basis: str = "unknown"

    def answer(
        self, scenario: Scenario, question: Question, stimulus: Stimulus
    ) -> Answer:
        raise NotImplementedError


def value_to_choice(question: Question, value: Any) -> Optional[str]:
    for choice in question.choices:
        if choice.value == value:
            return choice.id
    return None


# Simulated readers

SIMULATED_PERSONAS = ("literal", "diligent", "careless")
CLOSE_MARGIN = 0.05


class SimulatedReader(Reader):
    simulated = True
    deterministic = True
    provider = "simulated"
    model_family = "simulated"
    family_resolved = True
    family_basis = "simulated-fixture"

    def __init__(self, persona: str):
        if persona not in SIMULATED_PERSONAS:
            raise ValueError(
                "unknown simulated persona %r (have %s)"
                % (persona, ", ".join(SIMULATED_PERSONAS))
            )
        self.persona = persona
        self.family = "simulated"
        self.name = "simulated:" + persona

    def answer(
        self, scenario: Scenario, question: Question, stimulus: Stimulus
    ) -> Answer:
        presentation = stimulus.presentation
        values, method = self._value_source(scenario, question, presentation)
        if values is None:
            return Answer(choice_id=question.choices[0].id, method="guess")

        answer_value = apply_op(question.compute, values, presentation.order)
        if self.persona == "careless" and question.compute["op"] in (
            "argmin",
            "argmax",
        ):
            answer_value = self._careless_adjust(
                question, values, presentation, answer_value
            )

        choice = value_to_choice(question, answer_value)
        if choice is None:
            choice, method = question.choices[0].id, "guess"
        return Answer(choice_id=choice, method=method)

    def _value_source(
        self, scenario: Scenario, question: Question, presentation: Presentation
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        target = question.compute.get("field")
        order = presentation.order

        if target and all(
            presentation.shows_field(item_id, target) for item_id in order
        ):
            return {
                item_id: presentation.shown[item_id][target] for item_id in order
            }, "read"

        if self.persona in ("diligent", "careless") and target in scenario.derived:
            expression = scenario.derived[target]
            inputs = expr_fields(expression)
            if all(
                field in presentation.shown[item_id]
                for item_id in order
                for field in inputs
            ):
                return (
                    {
                        item_id: evaluate_expr(
                            expression, presentation.shown[item_id]
                        )
                        for item_id in order
                    },
                    "computed",
                )

        emphasized = presentation.emphasized_field
        if emphasized and all(
            presentation.shows_field(item_id, emphasized) for item_id in order
        ):
            return {
                item_id: presentation.shown[item_id][emphasized] for item_id in order
            }, "headline"

        return None, "guess"

    def _careless_adjust(
        self,
        question: Question,
        values: Dict[str, Any],
        presentation: Presentation,
        answer_value: Any,
    ) -> Any:
        op = question.compute["op"]
        order = presentation.order
        if len(order) < 2:
            return answer_value
        ranked = sorted(
            order, key=lambda item_id: values[item_id], reverse=(op == "argmax")
        )
        winner, runner_up = ranked[:2]
        first_value, second_value = values[winner], values[runner_up]
        gap = abs(first_value - second_value) / max(
            abs(first_value), abs(second_value), 1
        )
        if gap >= CLOSE_MARGIN:
            return answer_value
        if presentation.highlight_item == winner:
            return winner
        return runner_up


# Reader selection

def validate_reader_spec(spec: str, *, shorthand: bool = False) -> str:
    spec = config_text(spec, "reader spec")
    if shorthand and spec == "simulated":
        return spec
    provider, separator, persona = spec.partition(":")
    if not separator or provider.lower() != "simulated":
        raise ValueError("unsupported reader %r; use simulated:<persona>" % spec)
    persona = persona.strip()
    if persona not in SIMULATED_PERSONAS:
        raise ValueError(
            "unknown simulated persona %r (have %s)" % (persona, ", ".join(SIMULATED_PERSONAS))
        )
    return "simulated:" + persona


def build_reader(spec: str) -> Reader:
    return SimulatedReader(validate_reader_spec(spec).split(":", 1)[1])


def expand_reader_specs(specs: List[str]) -> List[str]:
    specs = strings(specs, "readers")
    expanded = []
    for spec in specs:
        spec = validate_reader_spec(spec, shorthand=True)
        if spec == "simulated":
            expanded.extend("simulated:" + persona for persona in SIMULATED_PERSONAS)
        else:
            expanded.append(spec)
    if len(set(expanded)) != len(expanded):
        raise ValueError("readers contain duplicate expanded specs")
    return expanded


def build_readers(specs: List[str]) -> List[Reader]:
    return [build_reader(spec) for spec in expand_reader_specs(specs)]
