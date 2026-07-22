"""H3 anchoring — how closely a model reader tracks a human reader.

An *anchor set* pairs benchmark questions with the accuracy real people achieved on the
same items (e.g. from CALVI or the Cleveland & McGill perception studies). The harness
runs the model readers on those items, then correlates model per-item accuracy against
human per-item accuracy (Spearman, Pearson, Kendall) and reports where they diverge most
— "where the proxy breaks."

The real human numbers must come from published data; they are never invented. A
*fixture* anchor set carries synthetic human numbers, labeled as such, purely so the
harness is verifiable offline — it can never stand in for a real H3 result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from . import analysis, scoring
from .interfaces import ALL_VARIANTS
from .readers.base import Reader
from .schema import load_all_scenarios


class AnchorError(ValueError):
    """An anchor set is malformed."""


@dataclass(frozen=True)
class AnchorItem:
    scenario: str
    question: str
    human_accuracy: float


@dataclass(frozen=True)
class AnchorSet:
    id: str
    source: str
    note: str
    condition: str            # the presentation readers see (e.g. "raw" — the raw data)
    items: List[AnchorItem]
    is_fixture: bool


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
    if not (isinstance(raw.get("id"), str) and raw["id"]):
        raise AnchorError("anchor set needs a string id")
    source = raw.get("source", "")
    if not isinstance(source, str):
        raise AnchorError("anchor 'source' must be a string")
    condition = raw.get("condition", "raw")
    if condition not in ALL_VARIANTS:
        raise AnchorError("anchor 'condition' %r is not a known presentation" % condition)

    scenarios = load_all_scenarios()
    raw_items = raw.get("items")
    if not (isinstance(raw_items, list) and raw_items):
        raise AnchorError("anchor set needs a non-empty 'items' list")
    items: List[AnchorItem] = []
    for it in raw_items:
        sid, qid = it.get("scenario"), it.get("question")
        if sid not in scenarios:
            raise AnchorError("anchor item references unknown scenario %r" % sid)
        if qid not in {q.id for q in scenarios[sid].questions}:
            raise AnchorError("anchor item references unknown question %r in %r" % (qid, sid))
        ha = it.get("human_accuracy")
        if not (isinstance(ha, (int, float)) and 0.0 <= ha <= 1.0):
            raise AnchorError("anchor item %s/%s needs human_accuracy in [0,1]" % (sid, qid))
        items.append(AnchorItem(scenario=sid, question=qid, human_accuracy=float(ha)))

    # Safe default: an unlabeled set is treated as a fixture (over-warns) rather than
    # emitting an H3 claim with invented numbers. A real set must declare is_fixture: false.
    is_fixture = bool(raw.get("is_fixture", True))
    return AnchorSet(id=raw["id"], source=source, note=raw.get("note", ""),
                     condition=condition, items=items, is_fixture=is_fixture)


def load_anchor_set(path: str) -> AnchorSet:
    import json
    with open(path, "r", encoding="utf-8") as fh:
        try:
            raw = json.load(fh)
        except json.JSONDecodeError as exc:
            raise AnchorError("%s: invalid JSON: %s" % (path, exc))
    return parse_anchor_set(raw)


def run_anchor(anchor: AnchorSet, readers: List[Reader], skip_render: bool = False,
               replicates: int = 1, out_dir: Optional[str] = None) -> AnchorResult:
    """Run the readers on the anchor items and correlate model vs human accuracy."""
    scenarios_by_id = load_all_scenarios()
    needed = [scenarios_by_id[sid] for sid in {it.scenario for it in anchor.items}]
    results = scoring.read_all(needed, readers, variants=[anchor.condition],
                               skip_render=skip_render, out_dir=out_dir, replicates=replicates)

    # Per-item model accuracy, pooled over readers and replicates.
    by_item: Dict = {}
    for qr in results:
        by_item.setdefault((qr.scenario_id, qr.question_id), []).append(qr.correct)

    points: List[AnchorPoint] = []
    for it in anchor.items:
        answers = by_item.get((it.scenario, it.question), [])
        model_acc = sum(answers) / len(answers) if answers else 0.0
        points.append(AnchorPoint(scenario=it.scenario, question=it.question,
                                  human_accuracy=it.human_accuracy, model_accuracy=model_acc))

    human = [p.human_accuracy for p in points]
    model = [p.model_accuracy for p in points]
    return AnchorResult(
        anchor=anchor, readers=[r.name for r in readers], points=points,
        spearman=analysis.spearman(model, human),
        pearson=analysis.pearson(model, human),
        kendall=analysis.kendall_tau(model, human))
