"""Load scenarios, validate task structure and snapshot their complete content."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from .resources import ResourceError, decode_json, resource_json, resource_names

_KNOWN_OPS = {"argmin", "argmax", "rank", "count_ge", "count_le", "count_gt", "count_lt"}
SCENARIO_HASH_FORMAT = "glassbox.scenario.v1"


class SchemaError(ValueError):
    """A scenario is structurally malformed."""


@dataclass(frozen=True)
class PresentationHints:
    """Default display fields and the highlighted primary extreme."""

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
    if metric is not None:
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
    op = spec.get("op")
    _require(isinstance(op, str) and op in _KNOWN_OPS, "%s: unknown compute op %r" % (where, op))
    _require(
        _text(spec.get("field")) and spec["field"] in raw_fields | derived_names,
        "%s: compute field is not a raw or derived field" % where,
    )
    if op == "rank":
        k = spec.get("k")
        _require(
            type(k) is int and 1 <= k <= len(item_labels),
            "%s: rank needs an integer k within the item range" % where,
        )
        _require(
            spec.get("order", "asc") in ("asc", "desc"), "%s: rank order must be asc/desc" % where
        )
    if op.startswith("count_"):
        _require(
            _finite(spec.get("threshold")), "%s: count threshold must be finite numeric" % where
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
    return Scenario(
        sid, raw["title"], raw["domain"], raw["description"], copy.deepcopy(data), questions
    )


def validate_structure(scenario: Scenario) -> None:
    """Apply parsed-shape guards to direct public Scenario instances."""
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
            raw = decode_json(stream.read(), str(path))
        return parse_scenario(raw)
    except (OSError, UnicodeError, ResourceError) as exc:
        raise SchemaError("%s: %s" % (path, exc)) from exc


def scenarios_dir(base: Optional[str] = None) -> str:
    """Retain the directory helper for explicit filesystem callers."""
    if base is not None:
        return os.path.join(base, "data", "scenarios")
    return os.path.join(os.path.dirname(__file__), "_data", "scenarios")


def load_all_scenarios(directory: Optional[str] = None) -> Dict[str, Scenario]:
    out: Dict[str, Scenario] = {}
    try:
        names = resource_names("scenarios") if directory is None else sorted(os.listdir(directory))
        for name in names:
            if not name.endswith(".json"):
                continue
            scenario = (
                parse_scenario(resource_json("scenarios", name))
                if directory is None
                else load_scenario(os.path.join(directory, name))
            )
            _require(scenario.id not in out, "duplicate scenario id %r" % scenario.id)
            out[scenario.id] = scenario
    except (OSError, ResourceError) as exc:
        raise SchemaError("cannot load scenarios: %s" % exc) from exc
    return dict(sorted(out.items()))
