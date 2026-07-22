"""H4 — the live repair loop: regenerate an interface to fix a reader's mistake.

Starting from the misleading polished cards, the reader answers a target question. If
it is wrong, the loop *diagnoses* what the interface lacked and adds the feature that
supplies it — show the honest computed value, then highlight the correct option — and
re-asks, counting **turns to understanding**. Diagnosis is adaptive (it inspects the
current interface), and because repair walks the same feature lattice the variants use,
it converges on the built-in fixtures.
"""

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
    transform: str   # what was changed to produce this interface ("" for the first)


@dataclass(frozen=True)
class RepairResult:
    scenario_id: str
    reader: str
    question_id: str
    turns: List[RepairTurn]
    converged: bool
    turns_to_understanding: Optional[int]  # index of the first correct turn, None if never


def _next_transform(scenario: Scenario, presentation: Presentation):
    """Diagnose the interface's gap and return (description, features-to-add), or None."""
    metric = scenario.presentation.primary_metric
    metric_shown = all(presentation.shows_field(i, metric) for i in presentation.order)
    if not metric_shown:
        label = scenario.data.get("value_label") or metric
        return "show the computed %s" % label, frozenset({"detail", "derived"})
    if presentation.highlight_item is None:
        return "highlight and sort by the correct option", frozenset({"highlight", "sorted"})
    return None


def repair(scenario: Scenario, reader: Reader, question: Question,
           max_turns: int = MAX_TURNS, skip_render: bool = False,
           out_dir: Optional[str] = None) -> RepairResult:
    """Run the repair loop for one reader on one question."""
    presentation = interfaces.variant(scenario, START_VARIANT)
    turns: List[RepairTurn] = []
    transform_desc = ""
    converged = False
    ttu: Optional[int] = None

    for turn_index in range(max_turns + 1):
        stimulus = render.render(presentation, out_dir=out_dir, skip=skip_render)
        ans = reader.answer(scenario, question, stimulus)
        correct = ans.choice_id == question.answer
        turns.append(RepairTurn(
            turn=turn_index, variant=presentation.variant,
            features=sorted(presentation.features), chosen=ans.choice_id,
            correct=correct, transform=transform_desc))
        if correct:
            converged = True
            ttu = turn_index
            break
        nxt = _next_transform(scenario, presentation)
        if nxt is None:
            break  # nothing left to try — did not converge
        transform_desc, added = nxt
        presentation = interfaces.with_features(scenario, presentation, added)

    return RepairResult(scenario_id=scenario.id, reader=reader.name,
                        question_id=question.id, turns=turns, converged=converged,
                        turns_to_understanding=ttu)
