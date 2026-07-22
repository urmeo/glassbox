"""Scoring — read every interface with every reader, then compute the metrics.

The reported number is **comprehension lift**: a reader's accuracy on an interface
minus its accuracy on the plain-text baseline, on the same source. **Ceiling control**
compares against the reader's accuracy on the raw data — separating a clear interface
from a reader that is simply good at arithmetic. Aggregation across readers is the
headline; a one-reader result describes that reader, not the interface.
"""

from __future__ import annotations

import os
import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from . import interfaces, render
from .interfaces import INTERFACE_VARIANTS
from .readers.base import Reader
from .schema import Scenario

CEILING_VARIANT = "raw"
BASELINE_VARIANT = "baseline"


@dataclass(frozen=True)
class QuestionResult:
    scenario_id: str
    reader: str
    reader_simulated: bool
    variant: str
    question_id: str
    chosen: Optional[str]
    correct_answer: str
    correct: bool
    method: str


def read_all(scenarios: Sequence[Scenario], readers: Sequence[Reader],
             variants: Optional[Sequence[str]] = None, skip_render: bool = False,
             out_dir: Optional[str] = None) -> List[QuestionResult]:
    """Render each (scenario, variant) once and have every reader answer every question."""
    variants = list(variants) if variants else list(interfaces.ALL_VARIANTS)
    stim_dir = os.path.join(out_dir, "stimuli") if out_dir else None
    results: List[QuestionResult] = []
    for scenario in scenarios:
        for v in variants:
            presentation = interfaces.variant(scenario, v)
            stimulus = render.render(presentation, out_dir=stim_dir, skip=skip_render)
            for reader in readers:
                for q in scenario.questions:
                    ans = reader.answer(scenario, q, stimulus)
                    results.append(QuestionResult(
                        scenario_id=scenario.id, reader=reader.name,
                        reader_simulated=reader.simulated, variant=v, question_id=q.id,
                        chosen=ans.choice_id, correct_answer=q.answer,
                        correct=(ans.choice_id == q.answer), method=ans.method))
    return results


class ScoreBook:
    """Queryable metrics over a list of :class:`QuestionResult`."""

    def __init__(self, results: Sequence[QuestionResult]):
        self.results = list(results)
        self.readers = sorted({r.reader for r in results})
        self.scenarios = sorted({r.scenario_id for r in results})
        self.variants = _ordered_variants({r.variant for r in results})
        self._simulated = {r.reader: r.reader_simulated for r in results}
        self._tally: Dict[Tuple[str, str, str], List[int]] = {}
        for r in results:
            cell = self._tally.setdefault((r.reader, r.scenario_id, r.variant), [0, 0])
            cell[0] += 1 if r.correct else 0
            cell[1] += 1

    # --- per-reader metrics ---
    def accuracy(self, reader: str, scenario: str, variant: str) -> float:
        correct, total = self._tally.get((reader, scenario, variant), (0, 0))
        return correct / total if total else 0.0

    def ceiling(self, reader: str, scenario: str) -> float:
        return self.accuracy(reader, scenario, CEILING_VARIANT)

    def baseline(self, reader: str, scenario: str) -> float:
        return self.accuracy(reader, scenario, BASELINE_VARIANT)

    def lift(self, reader: str, scenario: str, variant: str) -> float:
        return self.accuracy(reader, scenario, variant) - self.baseline(reader, scenario)

    def ceiling_gap(self, reader: str, scenario: str, variant: str) -> float:
        return self.accuracy(reader, scenario, variant) - self.ceiling(reader, scenario)

    # --- cross-reader aggregates (the headline) ---
    def mean_lift(self, scenario: str, variant: str) -> float:
        return statistics.fmean(self.lift(r, scenario, variant) for r in self.readers)

    def mean_accuracy(self, scenario: str, variant: str) -> float:
        return statistics.fmean(self.accuracy(r, scenario, variant) for r in self.readers)

    def interface_variants(self) -> List[str]:
        return [v for v in self.variants if v in INTERFACE_VARIANTS]

    def all_simulated(self) -> bool:
        return all(self._simulated.values())

    def any_real(self) -> bool:
        return any(not sim for sim in self._simulated.values())


def _ordered_variants(present: set) -> List[str]:
    ordered = [v for v in interfaces.ALL_VARIANTS if v in present]
    ordered += sorted(v for v in present if v not in interfaces.ALL_VARIANTS)
    return ordered
