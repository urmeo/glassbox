"""The preference judge — measures what is *liked*, never what is *understood*.

It runs alongside the readers and is deliberately kept out of the score: comprehension
is measured only by readers answering questions. Keeping preference and comprehension
separate is what lets the harness show them diverge.

The v1 judge is a deterministic **rating** judge over aesthetic features — a stand-in
that reproduces the field's "people prefer the polished one" finding offline. It is a
placeholder that a later version upgrades to real pairwise API judges. It looks only at
how an interface *presents*, never at whether its content is correct.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import List, Sequence

from .interfaces import Presentation
from .schema import Scenario


@dataclass(frozen=True)
class Preference:
    scenario_id: str
    variant: str
    score: float
    judge: str
    simulated: bool


class Judge:
    name = "judge"
    simulated = False

    def preference(self, scenario: Scenario, presentation: Presentation) -> float:
        raise NotImplementedError


class SimulatedJudge(Judge):
    """Rates polish from presentation features — the more styled and less busy, the higher."""

    name = "simulated:polish"
    simulated = True

    # Aesthetic weights: polish dominates (the field's 70-83% preference for polished UIs),
    # small bonuses for helpful-looking touches, a penalty for plain text and for clutter.
    def preference(self, scenario: Scenario, presentation: Presentation) -> float:
        f = presentation.features
        score = 0.30
        if "polished" in f:
            score += 0.35
        if "highlight" in f:
            score += 0.08
        if "derived" in f:
            score += 0.05
        if "sorted" in f:
            score += 0.04
        if "as_text" in f:
            score -= 0.15
        if presentation.order:  # busier interfaces read as less clean
            avg_fields = statistics.fmean(len(presentation.shown[i]) for i in presentation.order)
            score -= 0.015 * avg_fields
        return max(0.0, min(1.0, score))


def build_judge(spec: str) -> Judge:
    """Construct a judge from a spec string. ``polish`` is the deterministic v1 judge."""
    if spec in ("polish", "simulated", "simulated:polish"):
        return SimulatedJudge()
    raise ValueError("unknown judge %r (have: polish)" % spec)


def judge_all(judge: Judge, scenario: Scenario,
              presentations: Sequence[Presentation]) -> List[Preference]:
    """Score a set of presentations for one scenario."""
    return [Preference(scenario_id=scenario.id, variant=p.variant,
                       score=judge.preference(scenario, p), judge=judge.name,
                       simulated=judge.simulated)
            for p in presentations]
