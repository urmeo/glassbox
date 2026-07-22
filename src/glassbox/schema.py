"""Scenario schema — typed load + structural validation of ``data/scenarios/*.json``.

A scenario is source data plus multiple-choice questions whose correct answer is
*recomputable* from that data. This module guards the file's
*shape*; :mod:`glassbox.validate` guards its *answers* (recomputation). Keeping
them separate means a malformed file fails loudly here, a wrong answer key fails
loudly there, and neither can masquerade as the other.

Scenario JSON shape::

    {
      "id": "loans",
      "title": "Three loan offers",
      "domain": "choosing between options",
      "description": "...",
      "data": {
        "unit": "$",                       # optional, for display
        "value_label": "total cost",       # optional, for display
        "items": [
          {"id": "A", "label": "Offer A", "monthly_payment": 305.0,
           "term_months": 36, "fees": 300, "stated_total": 11280, "apr_pct": 7.1},
          ...
        ],
        "derived": {"total_cost": "monthly_payment * term_months + fees"}
      },
      "questions": [
        {
          "id": "cheapest_total",
          "stem": "Which offer costs the least in total over its full term?",
          "compute": {"op": "argmin", "field": "total_cost"},
          "choices": [{"id": "a", "text": "Offer A", "value": "A"}, ...],
          "answer": "a"
        }
      ]
    }
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

_KNOWN_OPS = {"argmin", "argmax", "rank", "count_ge", "count_le", "count_gt", "count_lt"}


class SchemaError(ValueError):
    """A scenario file is structurally malformed."""


@dataclass(frozen=True)
class PresentationHints:
    """How interfaces should present this scenario (config, not content).

    ``primary_metric`` is the derived value the honest interfaces (table, annotated)
    display and the analysis treats as "the answer number". ``headline_field`` is the
    raw field the polished ``cards`` interface emphasizes instead — the field that can
    mislead a reader who takes it at face value. ``primary_extreme`` (argmin/argmax)
    is which end of ``primary_metric`` the annotated interface highlights.
    """
    primary_metric: str
    primary_extreme: str  # "argmin" or "argmax"
    headline_field: Optional[str]
    cards_fields: List[str]
    detail_fields: List[str]


@dataclass(frozen=True)
class Choice:
    id: str
    text: str
    value: Any  # an item id (str) for item ops, or an int for count ops


@dataclass(frozen=True)
class Question:
    id: str
    stem: str
    compute: Dict[str, Any]
    choices: List[Choice]
    answer: str  # a choice id

    @property
    def target_field(self) -> Optional[str]:
        """The data field this question hinges on (used by readers/interfaces)."""
        return self.compute.get("field")

    @property
    def correct_choice(self) -> Choice:
        for c in self.choices:
            if c.id == self.answer:
                return c
        raise SchemaError("question %r answer %r is not among its choices"
                          % (self.id, self.answer))


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
        """Interface presentation hints, with sensible defaults if unspecified."""
        pres = self.data.get("presentation", {})
        raw_fields = sorted(raw_field_names(self.items))
        default_metric = next(iter(self.derived), (raw_fields[0] if raw_fields else ""))
        return PresentationHints(
            primary_metric=pres.get("primary_metric", default_metric),
            primary_extreme=pres.get("primary_extreme", "argmin"),
            headline_field=pres.get("headline_field"),
            cards_fields=list(pres.get("cards_fields", [])),
            detail_fields=list(pres.get("detail_fields", raw_fields)),
        )

    def item(self, item_id: str) -> Dict[str, Any]:
        for it in self.items:
            if it["id"] == item_id:
                return it
        raise KeyError("no item %r in scenario %r" % (item_id, self.id))


def _require(cond: bool, message: str) -> None:
    if not cond:
        raise SchemaError(message)


def raw_field_names(items: List[Dict[str, Any]]) -> set:
    """The union of every raw field name across items (excluding id and label)."""
    names = set()
    for it in items:
        names.update(k for k in it if k not in ("id", "label"))
    return names


def _validate_presentation(pres: Dict[str, Any], sid: str,
                           raw_fields: set, derived_names: set) -> None:
    _require(isinstance(pres, dict), "%s: presentation must be an object" % sid)
    metric = pres.get("primary_metric")
    if metric is not None:
        _require(metric in raw_fields or metric in derived_names,
                 "%s: presentation.primary_metric %r is not a field" % (sid, metric))
    extreme = pres.get("primary_extreme", "argmin")
    _require(extreme in ("argmin", "argmax"),
             "%s: presentation.primary_extreme must be argmin/argmax" % sid)
    headline = pres.get("headline_field")
    if headline is not None:
        _require(headline in raw_fields,
                 "%s: presentation.headline_field %r is not a raw field" % (sid, headline))
    for key in ("cards_fields", "detail_fields"):
        value = pres.get(key, [])
        _require(isinstance(value, list), "%s: presentation.%s must be a list" % (sid, key))
        for f in value:
            _require(f in raw_fields,
                     "%s: presentation.%s references non-raw field %r" % (sid, key, f))


def _parse_question(raw: Dict[str, Any], scenario_id: str,
                    item_labels: Dict[str, str]) -> Question:
    qid = raw.get("id")
    _require(isinstance(qid, str) and qid, "%s: a question has no id" % scenario_id)
    where = "%s/%s" % (scenario_id, qid)

    _require(isinstance(raw.get("stem"), str) and raw["stem"],
             "%s: missing stem" % where)

    compute = raw.get("compute")
    _require(isinstance(compute, dict), "%s: missing compute spec" % where)
    op = compute.get("op")
    _require(op in _KNOWN_OPS, "%s: unknown compute op %r" % (where, op))
    if op in ("argmin", "argmax", "rank") or op.startswith("count_"):
        _require(isinstance(compute.get("field"), str),
                 "%s: compute op %r needs a 'field'" % (where, op))
    if op == "rank":
        _require(isinstance(compute.get("k"), int) and compute["k"] >= 1,
                 "%s: rank needs integer k >= 1" % where)
    if op.startswith("count_"):
        _require(isinstance(compute.get("threshold"), (int, float)),
                 "%s: %s needs a numeric 'threshold'" % (where, op))

    raw_choices = raw.get("choices")
    _require(isinstance(raw_choices, list) and len(raw_choices) >= 2,
             "%s: needs >= 2 choices" % where)
    choices: List[Choice] = []
    seen_ids = set()
    for rc in raw_choices:
        _require(isinstance(rc, dict), "%s: a choice is not an object" % where)
        cid, text = rc.get("id"), rc.get("text")
        _require(isinstance(cid, str) and cid, "%s: a choice has no id" % where)
        _require(cid not in seen_ids, "%s: duplicate choice id %r" % (where, cid))
        seen_ids.add(cid)
        _require(isinstance(text, str) and text, "%s: choice %r has no text" % (where, cid))
        _require("value" in rc, "%s: choice %r has no 'value'" % (where, cid))
        choices.append(Choice(id=cid, text=text, value=rc["value"]))

    answer = raw.get("answer")
    _require(answer in seen_ids, "%s: answer %r not among choice ids" % (where, answer))

    # For item-referencing ops, every choice value must name a real item, and the
    # choice's visible text must name that same item — a real reader answers from the
    # text it sees, so a text/value mismatch would mark a correct reader wrong.
    if op in ("argmin", "argmax", "rank"):
        for c in choices:
            _require(c.value in item_labels,
                     "%s: choice %r value %r is not an item id" % (where, c.id, c.value))
            label = item_labels[c.value]
            _require(label.lower() in c.text.lower(),
                     "%s: choice %r text %r does not name item %r (label %r)"
                     % (where, c.id, c.text, c.value, label))

    return Question(id=qid, stem=raw["stem"], compute=compute,
                    choices=choices, answer=answer)


def parse_scenario(raw: Dict[str, Any]) -> Scenario:
    """Build and structurally validate a :class:`Scenario` from a parsed dict."""
    sid = raw.get("id")
    _require(isinstance(sid, str) and sid, "scenario has no id")
    for key in ("title", "domain", "description"):
        _require(isinstance(raw.get(key), str) and raw[key],
                 "%s: missing %s" % (sid, key))

    data = raw.get("data")
    _require(isinstance(data, dict), "%s: missing data" % sid)
    items = data.get("items")
    _require(isinstance(items, list) and len(items) >= 1, "%s: needs items" % sid)
    item_labels: Dict[str, str] = {}
    for it in items:
        _require(isinstance(it, dict) and isinstance(it.get("id"), str),
                 "%s: every item needs a string id" % sid)
        _require(it["id"] not in item_labels, "%s: duplicate item id %r" % (sid, it["id"]))
        _require(isinstance(it.get("label"), str) and it["label"],
                 "%s: item %r needs a label" % (sid, it["id"]))
        item_labels[it["id"]] = it["label"]

    derived = data.get("derived", {})
    _require(isinstance(derived, dict), "%s: 'derived' must be an object" % sid)
    for name, expr in derived.items():
        _require(isinstance(expr, str) and expr,
                 "%s: derived %r must be a non-empty expression string" % (sid, name))

    if "presentation" in data:
        _validate_presentation(data["presentation"], sid,
                               raw_field_names(items), set(derived))

    raw_questions = raw.get("questions")
    _require(isinstance(raw_questions, list) and len(raw_questions) >= 1,
             "%s: needs >= 1 question" % sid)
    q_ids = set()
    questions: List[Question] = []
    for rq in raw_questions:
        _require(isinstance(rq, dict), "%s: a question is not an object" % sid)
        q = _parse_question(rq, sid, item_labels)
        _require(q.id not in q_ids, "%s: duplicate question id %r" % (sid, q.id))
        q_ids.add(q.id)
        questions.append(q)

    return Scenario(id=sid, title=raw["title"], domain=raw["domain"],
                    description=raw["description"], data=data, questions=questions)


def load_scenario(path: str) -> Scenario:
    """Load and structurally validate a single scenario JSON file."""
    with open(path, "r", encoding="utf-8") as fh:
        try:
            raw = json.load(fh)
        except json.JSONDecodeError as exc:
            raise SchemaError("%s: invalid JSON: %s" % (path, exc))
    return parse_scenario(raw)


def scenarios_dir(base: Optional[str] = None) -> str:
    """Locate ``data/scenarios`` relative to the repo root (two levels up from src)."""
    if base is None:
        base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(base, "data", "scenarios")


def load_all_scenarios(directory: Optional[str] = None) -> Dict[str, Scenario]:
    """Load every ``*.json`` in the scenarios directory, keyed and sorted by id."""
    directory = directory or scenarios_dir()
    out: Dict[str, Scenario] = {}
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".json"):
            continue
        scenario = load_scenario(os.path.join(directory, name))
        if scenario.id in out:
            raise SchemaError("duplicate scenario id %r (file %s)" % (scenario.id, name))
        out[scenario.id] = scenario
    return dict(sorted(out.items()))
