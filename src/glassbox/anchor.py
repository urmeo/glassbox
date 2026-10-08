"""Declared per-item human accuracy and reader comparison."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

from . import analysis, scoring
from ._config import (
    config_input_path,
    boolean,
    load_config,
    object_fields,
    positive_integer,
    text,
)
from .interfaces import ALL_VARIANTS
from .output_paths import validate_output_files
from .readers.base import Reader
from .schema import load_all_scenarios


class AnchorError(ValueError):
    """An anchor set is malformed."""


@dataclass(frozen=True)
class AnchorItem:
    scenario: str
    question: str
    human_accuracy: float

    def __post_init__(self) -> None:
        for name in ("scenario", "question"):
            object.__setattr__(self, name, text(getattr(self, name), name, AnchorError))
        value = self.human_accuracy
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0 <= value <= 1
            or not math.isfinite(value)
        ):
            raise AnchorError("human_accuracy must be finite and in [0,1]")


@dataclass(frozen=True)
class AnchorSet:
    id: str
    source: str
    note: str
    condition: str
    items: List[AnchorItem]
    is_fixture: bool

    def __post_init__(self) -> None:
        for name in ("id", "source", "note"):
            text(getattr(self, name), name, AnchorError, empty=name != "id")
        text(self.condition, "condition", AnchorError)
        if self.condition not in ALL_VARIANTS:
            raise AnchorError("unknown presentation %r" % self.condition)
        boolean(self.is_fixture, "is_fixture", AnchorError)
        if not isinstance(self.items, (list, tuple)) or not self.items:
            raise AnchorError("items must be a nonempty list of AnchorItem objects")
        identities = []
        scenarios = load_all_scenarios()
        for item in self.items:
            if not isinstance(item, AnchorItem):
                raise AnchorError("items must contain AnchorItem objects")
            item.__post_init__()
            if item.scenario not in scenarios:
                raise AnchorError("unknown scenario %r" % item.scenario)
            if item.question not in {q.id for q in scenarios[item.scenario].questions}:
                raise AnchorError(
                    "unknown question %r in %r" % (item.question, item.scenario)
                )
            identities.append((item.scenario, item.question))
        if len(set(identities)) != len(identities):
            raise AnchorError("duplicate anchor items")


@dataclass(frozen=True)
class AnchorPoint:
    scenario: str
    question: str
    human_accuracy: float
    model_accuracy: float

    @property
    def abs_gap(self) -> float:
        return abs(self.model_accuracy - self.human_accuracy)


@dataclass(frozen=True)
class AnchorResult:
    anchor: AnchorSet
    readers: List[str]
    points: List[AnchorPoint]
    spearman: float
    pearson: float
    kendall: float

    @property
    def n_items(self) -> int:
        return len(self.points)

    def breaks(self, top: int = 3) -> List[AnchorPoint]:
        """The items where model and human disagree most."""
        return sorted(self.points, key=lambda p: -p.abs_gap)[:top]


def parse_anchor_set(raw: Dict) -> AnchorSet:
    object_fields(raw, AnchorSet.__dataclass_fields__, AnchorError)
    raw_items = raw.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise AnchorError("items must be a nonempty list")
    items = []
    for raw_item in raw_items:
        object_fields(raw_item, AnchorItem.__dataclass_fields__, AnchorError)
        try:
            items.append(AnchorItem(**raw_item))
        except TypeError as exc:
            raise AnchorError(str(exc)) from None
    values = dict(raw, items=items)
    values.setdefault("source", "")
    values.setdefault("note", "")
    values.setdefault("condition", "raw")
    values.setdefault("is_fixture", True)
    try:
        return AnchorSet(**values)
    except TypeError as exc:
        raise AnchorError(str(exc)) from None


def load_anchor_set(source: str) -> AnchorSet:
    config = parse_anchor_set(load_config("anchors", source, AnchorError))
    object.__setattr__(config, "_source_path", config_input_path("anchors", source))
    return config


def run_anchor(
    anchor: AnchorSet,
    readers: List[Reader],
    skip_render: bool = False,
    replicates: int = 1,
    out_dir: Optional[str] = None,
) -> AnchorResult:
    """Run the readers on the anchor items and correlate model vs human accuracy."""
    if not isinstance(anchor, AnchorSet):
        raise AnchorError("anchor must be an AnchorSet")
    anchor.__post_init__()
    boolean(skip_render, "skip_render", AnchorError)
    positive_integer(replicates, "replicates", AnchorError)
    if (
        not isinstance(readers, (list, tuple))
        or not readers
        or any(not isinstance(reader, Reader) for reader in readers)
    ):
        raise AnchorError("readers must be a nonempty list of Reader objects")
    names = [reader.name for reader in readers]
    if len(set(names)) != len(names):
        raise AnchorError("duplicate readers")
    if out_dir is not None:
        protected = (
            (anchor._source_path,)
            if getattr(anchor, "_source_path", None) is not None
            else ()
        )
        validate_output_files(
            out_dir, ("anchor.md", "anchor.json"), protected_paths=protected
        )
    scenarios_by_id = load_all_scenarios()
    needed = [
        scenarios_by_id[sid]
        for sid in dict.fromkeys(it.scenario for it in anchor.items)
    ]
    results = scoring.read_all(
        needed,
        readers,
        variants=[anchor.condition],
        skip_render=skip_render,
        out_dir=out_dir,
        replicates=replicates,
    )

    by_item: Dict = {}
    for qr in results:
        by_item.setdefault((qr.scenario_id, qr.question_id), []).append(qr.correct)

    points: List[AnchorPoint] = []
    for it in anchor.items:
        answers = by_item.get((it.scenario, it.question), [])
        model_acc = sum(answers) / len(answers) if answers else 0.0
        points.append(
            AnchorPoint(
                scenario=it.scenario,
                question=it.question,
                human_accuracy=it.human_accuracy,
                model_accuracy=model_acc,
            )
        )

    human = [p.human_accuracy for p in points]
    model = [p.model_accuracy for p in points]
    return AnchorResult(
        anchor=anchor,
        readers=[r.name for r in readers],
        points=points,
        spearman=analysis.spearman(model, human),
        pearson=analysis.pearson(model, human),
        kendall=analysis.kendall_tau(model, human),
    )
