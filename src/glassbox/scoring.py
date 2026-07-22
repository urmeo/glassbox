"""Scoring — read every interface with every reader, then compute the metrics.

The reported number is **comprehension lift**: a reader's accuracy on an interface
minus its accuracy on the plain-text baseline, on the same source. **Ceiling control**
compares against the reader's accuracy on the raw data — separating a clear interface
from a reader that is simply good at arithmetic. Aggregation across readers is the
headline; a one-reader result describes that reader, not the interface.

Real readers are stochastic, so a reading can be *replicated*: each reader answers each
question ``replicates`` times, and the score book exposes both the pooled accuracy and
its spread across replicates (deterministic simulated readers have a spread of zero).
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

Cell = Tuple[str, str, str]  # (reader, scenario, variant)


@dataclass(frozen=True)
class QuestionResult:
    scenario_id: str
    reader: str
    reader_family: str
    reader_simulated: bool
    variant: str
    question_id: str
    chosen: Optional[str]
    correct_answer: str
    correct: bool
    method: str
    replicate: int = 0


def read_all(scenarios: Sequence[Scenario], readers: Sequence[Reader],
             variants: Optional[Sequence[str]] = None, skip_render: bool = False,
             out_dir: Optional[str] = None, replicates: int = 1) -> List[QuestionResult]:
    """Render each (scenario, variant) once; every reader answers every question
    ``replicates`` times. The stimulus is shared across readers and replicates."""
    if replicates < 1:
        raise ValueError("replicates must be >= 1")
    variants = list(variants) if variants else list(interfaces.ALL_VARIANTS)
    stim_dir = os.path.join(out_dir, "stimuli") if out_dir else None
    results: List[QuestionResult] = []
    for scenario in scenarios:
        for v in variants:
            presentation = interfaces.variant(scenario, v)
            stimulus = render.render(presentation, out_dir=stim_dir, skip=skip_render)
            for reader in readers:
                for rep in range(replicates):
                    for q in scenario.questions:
                        ans = reader.answer(scenario, q, stimulus)
                        results.append(QuestionResult(
                            scenario_id=scenario.id, reader=reader.name,
                            reader_family=reader.family, reader_simulated=reader.simulated,
                            variant=v, question_id=q.id, chosen=ans.choice_id,
                            correct_answer=q.answer, correct=(ans.choice_id == q.answer),
                            method=ans.method, replicate=rep))
    return results


class ScoreBook:
    """Queryable metrics over a list of :class:`QuestionResult`."""

    def __init__(self, results: Sequence[QuestionResult]):
        self.results = list(results)
        self.readers = sorted({r.reader for r in results})
        self.scenarios = sorted({r.scenario_id for r in results})
        self.variants = _ordered_variants({r.variant for r in results})
        self._simulated = {r.reader: r.reader_simulated for r in results}
        self._family = {r.reader: r.reader_family for r in results}
        self._cell: Dict[Cell, List[QuestionResult]] = {}
        for r in results:
            self._cell.setdefault((r.reader, r.scenario_id, r.variant), []).append(r)

    # --- per-reader metrics ---
    def accuracy(self, reader: str, scenario: str, variant: str) -> float:
        cell = self._cell.get((reader, scenario, variant), [])
        return statistics.fmean(qr.correct for qr in cell) if cell else 0.0

    def ceiling(self, reader: str, scenario: str) -> float:
        return self.accuracy(reader, scenario, CEILING_VARIANT)

    def baseline(self, reader: str, scenario: str) -> float:
        return self.accuracy(reader, scenario, BASELINE_VARIANT)

    def lift(self, reader: str, scenario: str, variant: str) -> float:
        return self.accuracy(reader, scenario, variant) - self.baseline(reader, scenario)

    def ceiling_gap(self, reader: str, scenario: str, variant: str) -> float:
        return self.accuracy(reader, scenario, variant) - self.ceiling(reader, scenario)

    # --- replicate spread (matters for stochastic real readers) ---
    def accuracy_by_replicate(self, reader: str, scenario: str, variant: str) -> List[float]:
        cell = self._cell.get((reader, scenario, variant), [])
        by_rep: Dict[int, List[bool]] = {}
        for qr in cell:
            by_rep.setdefault(qr.replicate, []).append(qr.correct)
        return [statistics.fmean(v) for _, v in sorted(by_rep.items())]

    def accuracy_std(self, reader: str, scenario: str, variant: str) -> float:
        accs = self.accuracy_by_replicate(reader, scenario, variant)
        return statistics.pstdev(accs) if len(accs) > 1 else 0.0

    def answer_stability(self, reader: str, scenario: str, variant: str) -> float:
        """Mean fraction of replicates that gave the modal answer, over questions.

        1.0 means perfectly consistent (every simulated reader); lower means the
        reader flips answers across replicates."""
        cell = self._cell.get((reader, scenario, variant), [])
        by_q: Dict[str, List[Optional[str]]] = {}
        for qr in cell:
            by_q.setdefault(qr.question_id, []).append(qr.chosen)
        fractions = []
        for _, chosen in by_q.items():
            modal = max(set(chosen), key=chosen.count)
            fractions.append(chosen.count(modal) / len(chosen))
        return statistics.fmean(fractions) if fractions else 1.0

    # --- cross-reader aggregates (the headline) ---
    def mean_lift(self, scenario: str, variant: str) -> float:
        return statistics.fmean(self.lift(r, scenario, variant) for r in self.readers)

    def mean_accuracy(self, scenario: str, variant: str) -> float:
        return statistics.fmean(self.accuracy(r, scenario, variant) for r in self.readers)

    # --- families / structure ---
    def family_of(self, reader: str) -> str:
        return self._family.get(reader, "")

    def families(self) -> List[str]:
        return sorted(set(self._family.values()))

    def readers_in_family(self, family: str) -> List[str]:
        return [r for r in self.readers if self._family.get(r) == family]

    def interface_variants(self) -> List[str]:
        return [v for v in self.variants if v in INTERFACE_VARIANTS]

    def replicate_count(self) -> int:
        return max((qr.replicate for qr in self.results), default=0) + 1

    def all_simulated(self) -> bool:
        return all(self._simulated.values())

    def any_real(self) -> bool:
        return any(not sim for sim in self._simulated.values())


def _ordered_variants(present: set) -> List[str]:
    ordered = [v for v in interfaces.ALL_VARIANTS if v in present]
    ordered += sorted(v for v in present if v not in interfaces.ALL_VARIANTS)
    return ordered
