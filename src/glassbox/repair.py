"""Add presentation features for the selected question within a bounded loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from . import interfaces, render
from .interfaces import Presentation
from .readers.base import Reader
from .schema import Question, Scenario

START_VARIANT = "cards"
MAX_TURNS = 4


@dataclass(frozen=True)
class RepairTurn:
    turn: int
    variant: str
    features: List[str]
    chosen: Optional[str]
    correct: bool
    transform: str


@dataclass(frozen=True)
class RepairResult:
    scenario_id: str
    reader: str
    question_id: str
    turns: List[RepairTurn]
    converged: bool
    turns_to_understanding: Optional[int]


def _next_transform(
    scenario: Scenario, presentation: Presentation, question: Optional[Question] = None
):
    metric = question.target_field if question is not None else presentation.primary_metric
    if not all(presentation.shows_field(iid, metric) for iid in presentation.order):
        return "show the target %s" % (presentation.value_label or metric), frozenset(
            {"detail", "derived"}
        )
    op = question.compute["op"] if question is not None else presentation.primary_extreme
    if op.startswith("count_"):
        if "sorted" not in presentation.features:
            return "sort the target values for the count comparison", frozenset({"sorted"})
    elif presentation.highlight_item is None:
        return "highlight and sort by the target %s" % op, frozenset({"highlight", "sorted"})
    return None


def repair(
    scenario: Scenario,
    reader: Reader,
    question: Question,
    max_turns: int = MAX_TURNS,
    skip_render: bool = False,
    out_dir: Optional[str] = None,
) -> RepairResult:
    if type(max_turns) is not int or max_turns < 0:
        raise ValueError("max_turns must be a nonnegative integer")
    presentation = interfaces.variant(scenario, START_VARIANT, target=question)
    turns: List[RepairTurn] = []
    description, converged, ttu = "", False, None
    for turn_index in range(max_turns + 1):
        stimulus = render.render(presentation, out_dir=out_dir, skip=skip_render)
        answer = reader.answer(scenario, question, stimulus)
        correct = answer.choice_id == question.answer
        turns.append(
            RepairTurn(
                turn_index,
                presentation.variant,
                sorted(presentation.features),
                answer.choice_id,
                correct,
                description,
            )
        )
        if correct:
            converged, ttu = True, turn_index
            break
        if turn_index == max_turns:
            break
        next_step = _next_transform(scenario, presentation, question)
        if next_step is None:
            break
        description, added = next_step
        presentation = interfaces.with_features(scenario, presentation, added, target=question)
    return RepairResult(scenario.id, reader.name, question.id, turns, converged, ttu)
