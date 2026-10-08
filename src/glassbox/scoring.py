"""Accuracy, plain-text lift and raw-data gap over scored answers."""

from __future__ import annotations

import os
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import interfaces, render
from .interfaces import INTERFACE_VARIANTS
from .readers.base import Reader
from .families import canonical_family, resolve_family
from .schema import Scenario

CEILING_VARIANT = "raw"
BASELINE_VARIANT = "baseline"

Cell = Tuple[str, str, str]


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
    reader_provider: str = ""
    reader_model: str = ""
    reader_model_family: str = "unknown"
    reader_family_resolved: bool = False
    reader_family_basis: str = "unknown"
    reader_deterministic: Optional[bool] = None
    stimulus_kind: str = "unknown"
    stimulus_text_sha256: Optional[str] = None
    stimulus_image_sha256: Optional[str] = None
    render_metadata: Dict[str, Any] = field(default_factory=dict)
    reader_prompt_sha256: Optional[str] = None
    raw_answer: str = ""


def read_all(
    scenarios: Sequence[Scenario],
    readers: Sequence[Reader],
    variants: Optional[Sequence[str]] = None,
    skip_render: bool = False,
    out_dir: Optional[str] = None,
    replicates: int = 1,
) -> List[QuestionResult]:
    """Share each stimulus across readers and replicates."""
    if (
        isinstance(replicates, bool)
        or not isinstance(replicates, int)
        or replicates < 1
    ):
        raise ValueError("replicates must be an integer >= 1")
    if not isinstance(skip_render, bool):
        raise ValueError("skip_render must be a boolean")
    scenarios, readers = list(scenarios), list(readers)
    if not scenarios or any(
        not isinstance(s, Scenario) or not s.questions for s in scenarios
    ):
        raise ValueError("scenarios must be nonempty Scenario values with questions")
    if len({s.id for s in scenarios}) != len(scenarios):
        raise ValueError("scenario IDs must be distinct")
    if not readers or any(not isinstance(r, Reader) for r in readers):
        raise ValueError("readers must be nonempty Reader values")
    if len({r.name for r in readers}) != len(readers):
        raise ValueError("reader names must be distinct")
    if isinstance(variants, (str, bytes)):
        raise ValueError("variants must be a sequence")
    variants = list(interfaces.ALL_VARIANTS) if variants is None else list(variants)
    if (
        not variants
        or any(
            not isinstance(v, str) or v not in interfaces.ALL_VARIANTS for v in variants
        )
        or len(set(variants)) != len(variants)
    ):
        raise ValueError("variants must be nonempty, known and distinct")
    stim_dir = os.path.join(out_dir, "stimuli") if out_dir else None
    results: List[QuestionResult] = []
    for scenario in scenarios:
        for v in variants:
            presentation = interfaces.variant(scenario, v)
            stimulus = render.render(presentation, out_dir=stim_dir, skip=skip_render)
            for reader in readers:
                identity = resolve_family(reader.name)
                model_family = getattr(reader, "model_family", identity.family)
                resolved = getattr(reader, "family_resolved", identity.resolved)
                basis = getattr(reader, "family_basis", identity.basis)
                for rep in range(replicates):
                    for q in scenario.questions:
                        ans = reader.answer(scenario, q, stimulus)
                        results.append(
                            QuestionResult(
                                scenario_id=scenario.id,
                                reader=reader.name,
                                reader_family=reader.family,
                                reader_simulated=reader.simulated,
                                variant=v,
                                question_id=q.id,
                                chosen=ans.choice_id,
                                correct_answer=q.answer,
                                correct=(ans.choice_id == q.answer),
                                method=ans.method,
                                replicate=rep,
                                reader_provider=getattr(
                                    reader, "provider", identity.provider
                                ),
                                reader_model=getattr(reader, "model", identity.model),
                                reader_model_family=model_family,
                                reader_family_resolved=resolved,
                                reader_family_basis=basis,
                                reader_deterministic=getattr(
                                    reader, "deterministic", None
                                ),
                                stimulus_kind=stimulus.kind,
                                stimulus_text_sha256=stimulus.text_sha256,
                                stimulus_image_sha256=stimulus.image_sha256,
                                render_metadata=dict(stimulus.render_metadata),
                                reader_prompt_sha256=getattr(
                                    ans, "prompt_sha256", None
                                ),
                                raw_answer=ans.raw,
                            )
                        )
    return results


class ScoreBook:
    """Queryable metrics over a list of :class:`QuestionResult`."""

    def __init__(self, results: Sequence[QuestionResult]):
        self.results = list(results)
        self.readers = sorted({r.reader for r in results})
        self.scenarios = sorted({r.scenario_id for r in results})
        self.variants = _ordered_variants({r.variant for r in results})
        self._simulated = {r.reader: r.reader_simulated for r in results}
        self._family = {
            r.reader: r.reader_model_family
            for r in results
            if r.reader_family_resolved is True
            and r.reader_model_family != "unknown"
            and (
                canonical_family(r.reader_model_family) != "unknown"
                or (
                    r.reader_simulated is True
                    and r.reader_family_basis == "synthetic-fixture"
                )
            )
        }
        self._deterministic = {r.reader: r.reader_deterministic for r in results}
        for name in self.readers:
            rows = [r for r in self.results if r.reader == name]
            identities = {
                (
                    r.reader_family,
                    r.reader_simulated,
                    r.reader_provider,
                    r.reader_model,
                    r.reader_model_family,
                    r.reader_family_resolved,
                    r.reader_family_basis,
                    r.reader_deterministic,
                )
                for r in rows
            }
            if len(identities) != 1:
                raise ValueError("inconsistent identity for reader %r" % name)
        self._cell: Dict[Cell, List[QuestionResult]] = {}
        for r in results:
            self._cell.setdefault((r.reader, r.scenario_id, r.variant), []).append(r)

    def accuracy(self, reader: str, scenario: str, variant: str) -> float:
        cell = self._cell.get((reader, scenario, variant), [])
        return statistics.fmean(qr.correct for qr in cell) if cell else float("nan")

    def ceiling(self, reader: str, scenario: str) -> float:
        return self.accuracy(reader, scenario, CEILING_VARIANT)

    def baseline(self, reader: str, scenario: str) -> float:
        return self.accuracy(reader, scenario, BASELINE_VARIANT)

    def lift(self, reader: str, scenario: str, variant: str) -> float:
        return self.accuracy(reader, scenario, variant) - self.baseline(
            reader, scenario
        )

    def ceiling_gap(self, reader: str, scenario: str, variant: str) -> float:
        return self.accuracy(reader, scenario, variant) - self.ceiling(reader, scenario)

    def accuracy_by_replicate(
        self, reader: str, scenario: str, variant: str
    ) -> List[float]:
        cell = self._cell.get((reader, scenario, variant), [])
        by_rep: Dict[int, List[bool]] = {}
        for qr in cell:
            by_rep.setdefault(qr.replicate, []).append(qr.correct)
        return [statistics.fmean(v) for _, v in sorted(by_rep.items())]

    def accuracy_std(self, reader: str, scenario: str, variant: str) -> float:
        accs = self.accuracy_by_replicate(reader, scenario, variant)
        if not accs:
            return float("nan")
        return statistics.pstdev(accs) if len(accs) > 1 else 0.0

    def answer_stability(self, reader: str, scenario: str, variant: str) -> float:
        """Mean modal-answer fraction over questions."""
        cell = self._cell.get((reader, scenario, variant), [])
        by_q: Dict[str, List[Optional[str]]] = {}
        for qr in cell:
            by_q.setdefault(qr.question_id, []).append(qr.chosen)
        fractions = []
        for _, chosen in by_q.items():
            modal = max(set(chosen), key=chosen.count)
            fractions.append(chosen.count(modal) / len(chosen))
        return (
            statistics.fmean(fractions) if fractions else float("nan")
        )

    def mean_lift(self, scenario: str, variant: str) -> float:
        return (
            statistics.fmean(self.lift(r, scenario, variant) for r in self.readers)
            if self.readers
            else float("nan")
        )

    def mean_accuracy(self, scenario: str, variant: str) -> float:
        return (
            statistics.fmean(self.accuracy(r, scenario, variant) for r in self.readers)
            if self.readers
            else float("nan")
        )

    def family_of(self, reader: str) -> str:
        return self._family.get(reader, "unknown")

    def families(self) -> List[str]:
        return sorted(set(self._family.values()))

    def readers_in_family(self, family: str) -> List[str]:
        return [r for r in self.readers if self._family.get(r) == family]

    def interface_variants(self) -> List[str]:
        return [v for v in self.variants if v in INTERFACE_VARIANTS]

    def replicate_count(self) -> int:
        return max((qr.replicate for qr in self.results), default=-1) + 1

    def all_simulated(self) -> bool:
        return bool(self.readers) and all(self._simulated.values())

    def any_simulated(self) -> bool:
        return any(self._simulated.values())

    def all_deterministic(self) -> bool:
        return bool(self.readers) and all(
            v is True for v in self._deterministic.values()
        )

    def unresolved_readers(self) -> List[str]:
        return [r for r in self.readers if r not in self._family]

    def any_real(self) -> bool:
        return any(not sim for sim in self._simulated.values())


def _ordered_variants(present: set) -> List[str]:
    ordered = [v for v in interfaces.ALL_VARIANTS if v in present]
    ordered += sorted(v for v in present if v not in interfaces.ALL_VARIANTS)
    return ordered
